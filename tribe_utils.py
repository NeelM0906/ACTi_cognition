"""
tribe_utils.py — Core inference, comparison, and visualization for the A/B Test Simulator.

Supports all TRIBE v2 input patterns (video, audio, text, image→video, multimodal) and
duration-aware comparison when A and B produce different-length predictions.

Architecture:
  - StimulusSpec / resolve_stimulus_for_tribe — modality routing, temp files, image→MP4
  - TribeInferenceEngine — TribeModel wrapper + disk cache + demo mode
  - ABComparator — Welch t-test + aligned time curves + ROI aggregates
  - BrainVisualizer — Plotly + nilearn surfaces
  - EngagementScorer — proxy scores + modality-specific interpretation helpers
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import diskcache
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for Gradio workers
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# TRIBE v2 surface outputs are typically at 2 Hz TR in training literature; keep in sync with UI
DEFAULT_TR_SEC = 0.5

# Static image shown as repeated frames (TRIBE expects a temporal video stream)
IMAGE_VIDEO_DURATION_SEC = 8.0
IMAGE_VIDEO_FPS = 24


# ---------------------------------------------------------------------------
# Modality modes (matches Gradio radio: lowercase values)
# ---------------------------------------------------------------------------


class ModalityMode(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


# ---------------------------------------------------------------------------
# Brain region bands — coarse fsaverage5 proxies (see prior project note)
# ---------------------------------------------------------------------------

BRAIN_REGIONS: dict[str, tuple[int, int]] = {
    "Visual Cortex (V1/V2)": (0, 1200),
    "Auditory Cortex": (1200, 2000),
    "Prefrontal Cortex": (2000, 3500),
    "Motor Cortex": (3500, 4800),
    "Temporal Lobe": (4800, 6800),
    "Parietal Lobe": (6800, 8500),
    "Occipital Lobe": (8500, 10000),
    "Limbic / Emotional": (10000, 12000),
    "Default Mode Network": (12000, 14000),
    "Attention Network": (14000, 16000),
    "Language Areas": (16000, 18000),
    "Supplementary Motor": (18000, 20484),
}

ENGAGEMENT_WEIGHTS: dict[str, float] = {
    "Visual Cortex (V1/V2)": 0.15,
    "Auditory Cortex": 0.10,
    "Prefrontal Cortex": 0.20,
    "Motor Cortex": 0.05,
    "Temporal Lobe": 0.10,
    "Parietal Lobe": 0.05,
    "Occipital Lobe": 0.08,
    "Limbic / Emotional": 0.12,
    "Default Mode Network": -0.05,
    "Attention Network": 0.18,
    "Language Areas": 0.10,
    "Supplementary Motor": 0.02,
}

ATTENTION_WEIGHTS: dict[str, float] = {
    "Visual Cortex (V1/V2)": 0.20,
    "Auditory Cortex": 0.15,
    "Prefrontal Cortex": 0.25,
    "Motor Cortex": 0.02,
    "Temporal Lobe": 0.08,
    "Parietal Lobe": 0.10,
    "Occipital Lobe": 0.05,
    "Limbic / Emotional": 0.03,
    "Default Mode Network": -0.10,
    "Attention Network": 0.25,
    "Language Areas": 0.10,
    "Supplementary Motor": 0.02,
}

EMOTION_WEIGHTS: dict[str, float] = {
    "Visual Cortex (V1/V2)": 0.05,
    "Auditory Cortex": 0.10,
    "Prefrontal Cortex": 0.15,
    "Motor Cortex": 0.03,
    "Temporal Lobe": 0.15,
    "Parietal Lobe": 0.05,
    "Occipital Lobe": 0.05,
    "Limbic / Emotional": 0.35,
    "Default Mode Network": 0.05,
    "Attention Network": 0.08,
    "Language Areas": 0.08,
    "Supplementary Motor": 0.02,
}

# Aggregated "systems" for reporting (mean of listed regions)
ROI_AGGREGATES: dict[str, list[str]] = {
    "Visual system (V1/V2 + Occipital)": [
        "Visual Cortex (V1/V2)",
        "Occipital Lobe",
    ],
    "Auditory system": ["Auditory Cortex"],
    "Language system (Language + Temporal)": [
        "Language Areas",
        "Temporal Lobe",
    ],
}


# ---------------------------------------------------------------------------
# Stimulus resolution — build TRIBE kwargs and temp artefacts
# ---------------------------------------------------------------------------


@dataclass
class StimulusSpec:
    """User intent for one variant before TRIBE preprocessing."""

    mode: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None
    image_path: Optional[str] = None


def effective_stimulus_spec(
    mode: str,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
) -> StimulusSpec:
    """
    Gradio keeps state for hidden components; zero out paths not applicable to `mode`
    so leftover uploads do not change Video- or Audio-only runs.
    """
    m = (mode or ModalityMode.VIDEO.value).strip().lower()
    if m == ModalityMode.VIDEO.value:
        return StimulusSpec(m, video_path=video_path, audio_path=None, text=None, image_path=None)
    if m == ModalityMode.AUDIO.value:
        return StimulusSpec(m, video_path=None, audio_path=audio_path, text=None, image_path=None)
    if m == ModalityMode.TEXT.value:
        return StimulusSpec(m, video_path=None, audio_path=None, text=text, image_path=None)
    if m == ModalityMode.IMAGE.value:
        return StimulusSpec(m, video_path=None, audio_path=None, text=None, image_path=image_path)
    # Multimodal — pass every stream the user supplied
    return StimulusSpec(
        m,
        video_path=video_path,
        audio_path=audio_path,
        text=text,
        image_path=image_path,
    )


@dataclass
class ResolvedStimulus:
    """Exact paths passed to `get_events_dataframe` after preprocessing."""

    video_path: Optional[str]
    audio_path: Optional[str]
    text_path: Optional[str]
    modalities_used: tuple[str, ...]
    warnings: tuple[str, ...]
    temp_files: list[str] = field(default_factory=list)
    image_derived_video: bool = False


def _write_temp_text_file(text: str) -> str:
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tf.write(text)
    tf.close()
    return tf.name


def image_to_static_video_mp4(
    image_path: str,
    duration_sec: float = IMAGE_VIDEO_DURATION_SEC,
    fps: int = IMAGE_VIDEO_FPS,
) -> str:
    """
    TRIBE has no standalone image encoder; we treat a static image as a short constant
    video so the visual stream matches the model's temporal convolutions.
    Prefer moviepy; fall back to imageio + repeated RGB frames (ffmpeg in PATH).
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    try:
        from moviepy.editor import ImageClip  # type: ignore

        clip = ImageClip(image_path).set_duration(duration_sec)
        clip = clip.set_fps(fps)
        clip.write_videofile(
            out_path,
            codec="libx264",
            audio=False,
            fps=fps,
            preset="ultrafast",
            logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        clip.close()
        return out_path
    except Exception as moviepy_exc:
        logger.warning("moviepy image→video failed (%s), trying imageio", moviepy_exc)
        try:
            from PIL import Image
            import imageio.v2 as imageio  # type: ignore

            img = np.asarray(Image.open(image_path).convert("RGB"))
            n_frames = max(2, int(duration_sec * fps))
            frames = [img] * n_frames
            imageio.mimwrite(
                out_path,
                frames,
                fps=fps,
                codec="libx264",
                quality=8,
            )
            return out_path
        except Exception as e2:
            if os.path.exists(out_path):
                os.unlink(out_path)
            raise RuntimeError(
                "Could not convert image to video (need moviepy or imageio+ffmpeg). "
                f"Original errors: {moviepy_exc}; {e2}"
            ) from e2


def resolve_stimulus_for_tribe(spec: StimulusSpec, cache_dir: Path) -> ResolvedStimulus:
    """
    Map UI modality mode + uploads into `video_path` / `audio_path` / `text_path`
    for `TribeModel.get_events_dataframe`, creating temps as needed.

    Multimodal: pass every non-None stream; at least two of {video,audio,text} recommended.
    Image + other streams: image becomes `video_path` (static clip); text/audio additionally set.
    """
    mode = (spec.mode or "video").strip().lower()
    warnings_list: list[str] = []
    temp_files: list[str] = []

    video_path = spec.video_path if spec.video_path else None
    audio_path = spec.audio_path if spec.audio_path else None
    raw_text = (spec.text or "").strip()
    text_path: Optional[str] = None
    image_derived = False

    if raw_text:
        text_path = _write_temp_text_file(raw_text)
        temp_files.append(text_path)

    modalities_used: list[str] = []

    if mode == ModalityMode.VIDEO.value:
        if not video_path:
            raise ValueError("Video mode requires a video upload.")
        modalities_used.append("video")

    elif mode == ModalityMode.AUDIO.value:
        if not audio_path:
            raise ValueError("Audio mode requires an audio upload.")
        modalities_used.append("audio")

    elif mode == ModalityMode.TEXT.value:
        if not raw_text:
            raise ValueError("Text mode requires non-empty text.")
        modalities_used.append("text")
        warnings_list.append(
            "Text-only: TRIBE internally synthesizes timing (e.g. speech pipeline); "
            "predictions may be shorter or smoother than audiovisual stimuli."
        )

    elif mode == ModalityMode.IMAGE.value:
        if not spec.image_path:
            raise ValueError("Image mode requires an image upload.")
        iv_path = image_to_static_video_mp4(spec.image_path)
        temp_files.append(iv_path)
        video_path = iv_path
        image_derived = True
        modalities_used.append("image→video")
        warnings_list.append(
            f"Image was converted to a {IMAGE_VIDEO_DURATION_SEC:.0f}s static video at "
            f"{IMAGE_VIDEO_FPS} fps for TRIBE (visual cortex sees sustained fixation)."
        )

    elif mode == ModalityMode.MULTIMODAL.value:
        count = sum(
            bool(x)
            for x in (video_path, audio_path, raw_text, spec.image_path)
        )
        if count < 1:
            raise ValueError("Multimodal mode needs at least video, audio, text, or image.")

        if spec.image_path and not video_path:
            iv_path = image_to_static_video_mp4(spec.image_path)
            temp_files.append(iv_path)
            video_path = iv_path
            image_derived = True
            modalities_used.append("image→video")
            warnings_list.append(
                "Static image encoded as short video clip for the visual channel."
            )

        if video_path and not image_derived:
            modalities_used.append("video")
        if audio_path:
            modalities_used.append("audio")
        if raw_text:
            modalities_used.append("text")
        if len([m for m in modalities_used if not m.startswith("image")]) < 2:
            warnings_list.append(
                "Multimodal mode with a single primary channel — comparisons work, but "
                "richer stimuli combine 2+ of video / audio / text."
            )
    else:
        raise ValueError(f"Unknown modality mode: {spec.mode}")

    return ResolvedStimulus(
        video_path=video_path,
        audio_path=audio_path,
        text_path=text_path,
        modalities_used=tuple(modalities_used),
        warnings=tuple(warnings_list),
        temp_files=temp_files,
        image_derived_video=image_derived,
    )


def modality_interpretation_notes(
    modalities_a: tuple[str, ...], modalities_b: tuple[str, ...]
) -> str:
    """Short heuristic bullets for the results pane (not peer-reviewed claims)."""
    lines: list[str] = []
    a_set = set(modalities_a)
    b_set = set(modalities_b)
    if "text" in a_set and "video" in b_set:
        lines.append(
            "- **A vs B heuristic:** Text-heavy **A** often loads **language / temporal** "
            "cortex more uniformly; **B** video may drive **visual + motion** streams more strongly."
        )
    if "video" in a_set and "text" in b_set:
        lines.append(
            "- **A vs B heuristic:** Video **A** recruits **visual + audiovisual integration**; "
            "text **B** emphasizes **language network** activation (after TTS alignment)."
        )
    if any("image" in m for m in modalities_a) or any("image" in m for m in modalities_b):
        lines.append(
            "- **Static image stimuli** lack motion; differences vs dynamic video are "
            "expected in **early visual vs higher-order** recruitment."
        )
    if "audio" in a_set and "text" in b_set and "video" not in a_set.union(b_set):
        lines.append(
            "- **Audio vs transcript:** Listening recruits **auditory cortex + prosody**; "
            "reading text (or TTS-pipeline text) shifts emphasis toward **language areas**."
        )
    if not lines:
        lines.append(
            "- Compare **Visual / Auditory / Language** ROI bars below for modality-specific lift."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Stores TRIBE v2 output for one stimulus variant."""

    label: str
    preds: np.ndarray  # (T, V)
    segments: pd.DataFrame | None
    events_df: pd.DataFrame | None
    modalities: tuple[str, ...] = ()
    stimulus_warnings: tuple[str, ...] = ()
    tr_sec: float = DEFAULT_TR_SEC
    mean_activation: float = field(init=False)
    activation_over_time: np.ndarray = field(init=False)
    duration_sec: float = field(init=False)
    n_timesteps: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_timesteps = int(self.preds.shape[0])
        self.mean_activation = float(self.preds.mean())
        self.activation_over_time = self.preds.mean(axis=1)
        self.duration_sec = self.n_timesteps * self.tr_sec

    @property
    def time_axis_sec(self) -> np.ndarray:
        return np.arange(self.n_timesteps, dtype=np.float64) * self.tr_sec


@dataclass
class ComparisonResult:
    """Statistical + narrative comparison of variant A vs B."""

    result_a: PredictionResult
    result_b: PredictionResult
    diff_over_time: np.ndarray
    time_axis_sec: np.ndarray
    overlay_time_sec: np.ndarray
    overlay_a: np.ndarray
    overlay_b: np.ndarray
    p_value: float
    t_statistic: float
    effect_size: float
    bootstrap_ci: tuple[float, float]
    winner: str
    confidence_label: str
    region_scores_a: dict[str, float]
    region_scores_b: dict[str, float]
    roi_aggregate_a: dict[str, float]
    roi_aggregate_b: dict[str, float]
    engagement_a: float
    engagement_b: float
    attention_a: float
    attention_b: float
    emotion_a: float
    emotion_b: float
    alignment_note: str
    stats_note: str
    modality_notes: str


# ---------------------------------------------------------------------------
# Alignment — compare stimuli of different lengths fairly
# ---------------------------------------------------------------------------


def align_activation_curves(
    result_a: PredictionResult,
    result_b: PredictionResult,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Interpolate both mean activation traces onto a common time base in seconds,
    from t=0 through min(duration_a, duration_b) so overlays are apple-to-apple.
    """
    ta, tb = result_a.time_axis_sec, result_b.time_axis_sec
    ya, yb = result_a.activation_over_time, result_b.activation_over_time
    t_end = min(ta.max() if len(ta) else 0, tb.max() if len(tb) else 0)
    if t_end <= 0:
        t_common = np.array([0.0])
        return t_common, ya[:1], yb[:1], yb[:1] - ya[:1], "Both stimuli have near-zero duration."

    t_common = np.linspace(0.0, t_end, num=n_points)
    ya_i = np.interp(t_common, ta, ya)
    yb_i = np.interp(t_common, tb, yb)
    diff = yb_i - ya_i
    note = (
        f"Time series overlaid on 0–{t_end:.2f}s (shorter of the two durations); "
        f"Δ uses interpolated values ({n_points} points)."
    )
    return t_common, ya_i, yb_i, diff, note


def _equal_length_crop_for_stats(preds_a: np.ndarray, preds_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both along time to the same number of frames (start-aligned)."""
    t = min(preds_a.shape[0], preds_b.shape[0])
    return preds_a[:t].ravel(), preds_b[:t].ravel()


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class TribeInferenceEngine:
    """TribeModel wrapper + disk cache + synthetic demo mode."""

    _model = None
    _demo_mode: bool = False

    def __init__(self, cache_dir: str = "./cache", device: str = "auto") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(str(self.cache_dir / "predictions"))
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        if TribeInferenceEngine._model is not None:
            return
        try:
            from tribev2 import TribeModel  # type: ignore

            logger.info("Loading TRIBE v2 model …")
            TribeInferenceEngine._model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder=str(self.cache_dir / "model"),
            )
            logger.info("TRIBE v2 loaded on %s", self.device)
        except (ImportError, Exception) as exc:
            logger.warning(
                "Could not load TribeModel (%s). Running in DEMO MODE with synthetic predictions.",
                exc,
            )
            TribeInferenceEngine._demo_mode = True

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
        return h.hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _synthetic_predictions(
        seed: int,
        n_timesteps: int = 120,
        n_vertices: int = 20484,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        preds = np.zeros((n_timesteps, n_vertices), dtype=np.float32)
        for region, (v_start, v_end) in BRAIN_REGIONS.items():
            amplitude = rng.uniform(0.1, 0.9)
            t = np.linspace(0, 2 * np.pi, n_timesteps)
            phase = rng.uniform(0, np.pi)
            envelope = 0.5 * (1 + np.sin(t + phase)) * amplitude
            spatial_noise = rng.normal(0, 0.05, (n_timesteps, v_end - v_start))
            preds[:, v_start:v_end] = envelope[:, None] + spatial_noise

        preds = np.clip(preds, 0, 1)
        segments = pd.DataFrame(
            {
                "start": np.arange(n_timesteps) * DEFAULT_TR_SEC,
                "end": np.arange(n_timesteps) * DEFAULT_TR_SEC + DEFAULT_TR_SEC,
                "label": [f"seg_{i}" for i in range(n_timesteps)],
            }
        )
        return preds, segments

    def _cache_key(self, resolved: ResolvedStimulus) -> str:
        parts = ["|".join(resolved.modalities_used)]
        if resolved.video_path:
            parts.append("v:" + self._hash_file(resolved.video_path))
        if resolved.audio_path:
            parts.append("a:" + self._hash_file(resolved.audio_path))
        if resolved.text_path:
            parts.append("t:" + self._hash_file(resolved.text_path))
        return "||".join(parts)

    def predict(
        self,
        label: str,
        spec: StimulusSpec,
        tr_sec: float = DEFAULT_TR_SEC,
    ) -> PredictionResult:
        """
        Full multimodal entrypoint: resolve files → call TRIBE (or demo) → PredictionResult.
        """
        self._load_model()
        resolved = resolve_stimulus_for_tribe(spec, self.cache_dir)
        cache_key = self._cache_key(resolved)

        if cache_key in self.disk_cache:
            logger.info("Cache hit for variant %s", label)
            cached = self.disk_cache[cache_key]
            return PredictionResult(
                label=label,
                preds=cached["preds"],
                segments=cached["segments"],
                events_df=cached.get("events_df"),
                modalities=cached.get("modalities", resolved.modalities_used),
                stimulus_warnings=cached.get("warnings", resolved.warnings),
                tr_sec=tr_sec,
            )

        try:
            if TribeInferenceEngine._demo_mode:
                seed = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
                preds, segments = self._synthetic_predictions(seed=seed)
                events_df = None
            else:
                model = TribeInferenceEngine._model
                events_df = model.get_events_dataframe(
                    video_path=resolved.video_path,
                    audio_path=resolved.audio_path,
                    text_path=resolved.text_path,
                )
                preds, segments = model.predict(events=events_df)

            payload = {
                "preds": preds,
                "segments": segments,
                "events_df": events_df,
                "modalities": resolved.modalities_used,
                "warnings": resolved.warnings,
            }
            self.disk_cache[cache_key] = payload

            return PredictionResult(
                label=label,
                preds=preds,
                segments=segments,
                events_df=events_df,
                modalities=resolved.modalities_used,
                stimulus_warnings=resolved.warnings,
                tr_sec=tr_sec,
            )
        finally:
            for path in resolved.temp_files:
                try:
                    if path and os.path.isfile(path):
                        os.unlink(path)
                except OSError:
                    pass

    @property
    def is_demo_mode(self) -> bool:
        return TribeInferenceEngine._demo_mode


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


class EngagementScorer:
    """Proxy scores + per-system aggregates."""

    @staticmethod
    def _region_mean(preds: np.ndarray, region: str) -> float:
        v_start, v_end = BRAIN_REGIONS[region]
        return float(preds[:, v_start:v_end].mean())

    @classmethod
    def compute_region_scores(cls, preds: np.ndarray) -> dict[str, float]:
        return {region: cls._region_mean(preds, region) for region in BRAIN_REGIONS}

    @classmethod
    def compute_roi_aggregates(cls, preds: np.ndarray) -> dict[str, float]:
        region = cls.compute_region_scores(preds)
        out: dict[str, float] = {}
        for sys_name, memb in ROI_AGGREGATES.items():
            out[sys_name] = float(np.mean([region[m] for m in memb]))
        return out

    @classmethod
    def _weighted_score(
        cls, region_scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        raw = sum(region_scores[r] * w for r, w in weights.items())
        normalized = (raw + 0.15) / 1.15
        return float(np.clip(normalized * 100, 0, 100))

    @classmethod
    def score(cls, preds: np.ndarray) -> dict[str, float]:
        region_scores = cls.compute_region_scores(preds)
        return {
            "engagement": cls._weighted_score(region_scores, ENGAGEMENT_WEIGHTS),
            "attention": cls._weighted_score(region_scores, ATTENTION_WEIGHTS),
            "emotion": cls._weighted_score(region_scores, EMOTION_WEIGHTS),
        }


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class ABComparator:
    """Welch t-test + duration-aware notes + aligned curves for plotting."""

    N_BOOTSTRAP = 1000

    @classmethod
    def compare(cls, result_a: PredictionResult, result_b: PredictionResult) -> ComparisonResult:
        flat_a = result_a.preds.ravel()
        flat_b = result_b.preds.ravel()

        # If differing numbers of timepoints, also report time-matched crop statistics
        cropped_a, cropped_b = _equal_length_crop_for_stats(result_a.preds, result_b.preds)
        t_stat, p_val = stats.ttest_ind(cropped_a, cropped_b, equal_var=False)

        pooled_std = np.sqrt((cropped_a.std() ** 2 + cropped_b.std() ** 2) / 2)
        effect_size = (cropped_b.mean() - cropped_a.mean()) / (pooled_std + 1e-9)

        rng = np.random.default_rng(42)
        diffs = []
        n_boot = min(len(cropped_a), len(cropped_b))
        step = max(1, n_boot // 5000)
        sub_a = cropped_a[::step][:5000]
        sub_b = cropped_b[::step][:5000]
        for _ in range(cls.N_BOOTSTRAP):
            sample_a = rng.choice(sub_a, size=min(len(sub_a), 4000), replace=True)
            sample_b = rng.choice(sub_b, size=min(len(sub_b), 4000), replace=True)
            diffs.append(sample_b.mean() - sample_a.mean())
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

        winner, confidence_label = cls._declare_winner(p_val, cropped_b.mean() - cropped_a.mean(), effect_size)

        scorer = EngagementScorer()
        scores_a = scorer.compute_region_scores(result_a.preds)
        scores_b = scorer.compute_region_scores(result_b.preds)
        roi_a = scorer.compute_roi_aggregates(result_a.preds)
        roi_b = scorer.compute_roi_aggregates(result_b.preds)
        proxy_a = EngagementScorer.score(result_a.preds)
        proxy_b = EngagementScorer.score(result_b.preds)

        overlay_t, ya, yb, diff_aligned, align_note = align_activation_curves(result_a, result_b)
        modality_notes = modality_interpretation_notes(result_a.modalities, result_b.modalities)

        dur_a, dur_b = result_a.duration_sec, result_b.duration_sec
        if abs(dur_a - dur_b) > 1e-3:
            stats_note = (
                f"Welch t-test used **time-aligned crops** (first {min(result_a.n_timesteps, result_b.n_timesteps)} "
                f"frames) so short vs long clips don't skew sample counts. "
                f"Durations: **A** {dur_a:.2f}s · **B** {dur_b:.2f}s."
            )
        else:
            stats_note = (
                "Durations match; t-test compares flattened vertex × time activations "
                "with equal time depth."
            )

        # Legacy diff_over_time used bar chart with one bar per timestep A — use aligned diff
        diff_over_time = diff_aligned

        return ComparisonResult(
            result_a=result_a,
            result_b=result_b,
            diff_over_time=diff_over_time,
            time_axis_sec=result_a.time_axis_sec,
            overlay_time_sec=overlay_t,
            overlay_a=ya,
            overlay_b=yb,
            p_value=float(p_val),
            t_statistic=float(t_stat),
            effect_size=float(effect_size),
            bootstrap_ci=(float(ci_low), float(ci_high)),
            winner=winner,
            confidence_label=confidence_label,
            region_scores_a=scores_a,
            region_scores_b=scores_b,
            roi_aggregate_a=roi_a,
            roi_aggregate_b=roi_b,
            engagement_a=proxy_a["engagement"],
            engagement_b=proxy_b["engagement"],
            attention_a=proxy_a["attention"],
            attention_b=proxy_b["attention"],
            emotion_a=proxy_a["emotion"],
            emotion_b=proxy_b["emotion"],
            alignment_note=align_note,
            stats_note=stats_note,
            modality_notes=modality_notes,
        )

    @staticmethod
    def _declare_winner(p_value: float, mean_diff: float, effect_size: float) -> tuple[str, str]:
        if p_value >= 0.05 or abs(effect_size) < 0.1:
            return "Tie", "Low"
        winner = "B" if mean_diff > 0 else "A"
        if p_value < 0.05 and abs(effect_size) > 0.5:
            return winner, "High"
        if p_value < 0.05:
            return winner, "Medium"
        return "Tie", "Low"


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class BrainVisualizer:
    """Charts for Gradio; activation plot uses aligned overlays from ComparisonResult."""

    @staticmethod
    def activation_over_time(comparison: ComparisonResult) -> go.Figure:
        ca, cb = comparison.result_a, comparison.result_b
        t_overlay = comparison.overlay_time_sec

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.65, 0.35],
            shared_xaxes=True,
            subplot_titles=[
                "Mean activation (aligned to common time base, s)",
                "Difference (B − A), interpolated",
            ],
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=t_overlay,
                y=comparison.overlay_a,
                mode="lines",
                name=f"Variant A ({', '.join(ca.modalities) or '—'})",
                line=dict(color="#60a5fa", width=2),
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.1)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_overlay,
                y=comparison.overlay_b,
                mode="lines",
                name=f"Variant B ({', '.join(cb.modalities) or '—'})",
                line=dict(color="#f97316", width=2),
                fill="tozeroy",
                fillcolor="rgba(249,115,22,0.1)",
            ),
            row=1,
            col=1,
        )

        colors = ["#f97316" if d > 0 else "#60a5fa" for d in comparison.diff_over_time]
        fig.add_trace(
            go.Bar(
                x=t_overlay,
                y=comparison.diff_over_time,
                name="B − A",
                marker_color=colors,
                opacity=0.75,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="rgba(100,160,210,0.5)", opacity=0.8, row=2, col=1
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#0a2a40", family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=40),
            height=520,
            annotations=[
                dict(
                    text=f"A: {ca.duration_sec:.2f}s &nbsp;|&nbsp; B: {cb.duration_sec:.2f}s",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.08,
                    showarrow=False,
                    font=dict(size=11, color="#4a7fa0"),
                )
            ],
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines","sendDataToCloud","editInChartStudio"],
        )
        fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                         gridcolor="rgba(100,160,210,0.15)", zerolinecolor="rgba(100,160,210,0.3)")
        fig.update_yaxes(title_text="Mean activation", row=1, col=1,
                         gridcolor="rgba(100,160,210,0.15)")
        fig.update_yaxes(title_text="Δ activation", row=2, col=1,
                         gridcolor="rgba(100,160,210,0.15)")
        return fig

    @staticmethod
    def region_comparison_bar(comparison: ComparisonResult) -> go.Figure:
        regions = list(BRAIN_REGIONS.keys())
        vals_a = [comparison.region_scores_a[r] for r in regions]
        vals_b = [comparison.region_scores_b[r] for r in regions]

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Variant A",
                    x=regions,
                    y=vals_a,
                    marker_color="#60a5fa",
                    opacity=0.85,
                ),
                go.Bar(
                    name="Variant B",
                    x=regions,
                    y=vals_b,
                    marker_color="#f97316",
                    opacity=0.85,
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            title="Mean activation by brain region",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#0a2a40", family="Inter, sans-serif"),
            xaxis_tickangle=-35,
            xaxis_gridcolor="rgba(100,160,210,0.15)",
            yaxis_gridcolor="rgba(100,160,210,0.15)",
            margin=dict(l=50, r=20, t=60, b=130),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines","sendDataToCloud","editInChartStudio"],
        )
        return fig

    @staticmethod
    def roi_system_bar(comparison: ComparisonResult) -> go.Figure:
        keys = list(ROI_AGGREGATES.keys())
        va = [comparison.roi_aggregate_a[k] for k in keys]
        vb = [comparison.roi_aggregate_b[k] for k in keys]
        fig = go.Figure(
            data=[
                go.Bar(name="Variant A", x=keys, y=va, marker_color="#60a5fa"),
                go.Bar(name="Variant B", x=keys, y=vb, marker_color="#f97316"),
            ]
        )
        fig.update_layout(
            barmode="group",
            title="ROI systems (visual / auditory / language aggregates)",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#0a2a40", family="Inter, sans-serif"),
            xaxis_tickangle=-20,
            xaxis_gridcolor="rgba(100,160,210,0.15)",
            yaxis_gridcolor="rgba(100,160,210,0.15)",
            height=400,
            margin=dict(l=50, r=20, t=60, b=100),
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines","sendDataToCloud","editInChartStudio"],
        )
        return fig

    @staticmethod
    def proxy_score_radar(comparison: ComparisonResult) -> go.Figure:
        categories = ["Engagement", "Attention", "Emotion"]
        vals_a = [comparison.engagement_a, comparison.attention_a, comparison.emotion_a]
        vals_b = [comparison.engagement_b, comparison.attention_b, comparison.emotion_b]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=vals_a + [vals_a[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Variant A",
                line_color="#60a5fa",
                fillcolor="rgba(96,165,250,0.25)",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=vals_b + [vals_b[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Variant B",
                line_color="#f97316",
                fillcolor="rgba(249,115,22,0.25)",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#4a7fa0", gridcolor="rgba(100,160,210,0.25)"),
                bgcolor="#f0f8ff",
                angularaxis=dict(color="#0a2a40"),
            ),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            font=dict(color="#0a2a40", family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            title="Neural proxy scores (0–100)",
            margin=dict(l=60, r=60, t=60, b=60),
            height=420,
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines",
                            "hoverClosestPolar","sendDataToCloud","editInChartStudio"],
        )
        return fig

    @staticmethod
    def single_activation_timeline(result: PredictionResult) -> go.Figure:
        t = result.time_axis_sec
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=result.activation_over_time,
                mode="lines",
                name=f"Variant {result.label}",
                line=dict(color="#a78bfa", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.15)",
            )
        )
        fig.update_layout(
            title=f"Activation over time — {result.label} ({', '.join(result.modalities) or 'stimulus'})",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
            font=dict(color="#e5e7eb", family="Inter, sans-serif"),
            xaxis_title="Time (s)",
            yaxis_title="Mean activation",
            margin=dict(l=50, r=20, t=60, b=40),
            height=350,
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines","sendDataToCloud","editInChartStudio"],
        )
        return fig

    @staticmethod
    def single_region_bar(result: PredictionResult) -> go.Figure:
        scorer = EngagementScorer()
        region_scores = scorer.compute_region_scores(result.preds)
        regions = list(region_scores.keys())
        vals = list(region_scores.values())

        norm = plt.Normalize(vmin=min(vals), vmax=max(vals))
        cmap = cm.get_cmap("plasma")
        bar_colors = [
            f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.85)"
            for r, g, b, _ in [cmap(norm(v)) for v in vals]
        ]

        fig = go.Figure(go.Bar(x=regions, y=vals, marker_color=bar_colors))
        fig.update_layout(
            title=f"Region activation — {result.label}",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
            font=dict(color="#e5e7eb", family="Inter, sans-serif"),
            xaxis_tickangle=-35,
            margin=dict(l=50, r=20, t=60, b=130),
            height=430,
            modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                            "autoScale2d","hoverClosestCartesian",
                            "hoverCompareCartesian","toggleSpikelines","sendDataToCloud","editInChartStudio"],
        )
        return fig

    @staticmethod
    def brain_surface_plot(
        preds: np.ndarray,
        title: str = "Max activation surface map",
    ) -> bytes:
        max_activation = preds.max(axis=0)

        try:
            import nilearn.plotting as nlplt
            from nilearn import datasets

            fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"}, facecolor="#111827"
            )
            plt.suptitle(title, color="white", fontsize=14, y=1.01)
            n_vertices_hemi = max_activation.shape[0] // 2

            for ax, hemi, surf in zip(
                axes,
                ["left", "right"],
                [fsaverage.infl_left, fsaverage.infl_right],
            ):
                hemi_data = (
                    max_activation[:n_vertices_hemi]
                    if hemi == "left"
                    else max_activation[n_vertices_hemi:]
                )
                nlplt.plot_surf_stat_map(
                    surf_mesh=surf,
                    stat_map=hemi_data,
                    hemi=hemi,
                    view="lateral",
                    colorbar=True,
                    cmap="hot",
                    bg_map=(
                        fsaverage.sulc_left
                        if hemi == "left"
                        else fsaverage.sulc_right
                    ),
                    title=f"{hemi.capitalize()} hemisphere",
                    axes=ax,
                    figure=fig,
                )

            plt.tight_layout()
        except Exception as exc:
            logger.warning("nilearn surface plot failed (%s), using fallback", exc)
            fig = BrainVisualizer._fallback_activation_map(max_activation, title)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#111827")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _fallback_activation_map(max_activation: np.ndarray, title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#ffffff")
        ax.set_facecolor("#f8fafc")
        side = int(np.sqrt(len(max_activation)))
        data_2d = max_activation[: side * side].reshape(side, side)
        im = ax.imshow(data_2d, cmap="YlOrRd", aspect="auto", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="Max activation", shrink=0.8)
        ax.set_title(title, color="#0a2a40", fontsize=13)
        ax.set_xlabel("Vertex (spatial proxy)", color="#4a7fa0")
        ax.set_ylabel("Vertex (spatial proxy)", color="#4a7fa0")
        ax.tick_params(colors="#4a7fa0")
        for spine in ax.spines.values():
            spine.set_edgecolor((100/255, 160/255, 210/255, 0.3))
        return fig


# ---------------------------------------------------------------------------
# Export + markdown
# ---------------------------------------------------------------------------


def build_summary_dataframe(comparison: ComparisonResult) -> pd.DataFrame:
    regions = list(BRAIN_REGIONS.keys())
    rows = []
    for region in regions:
        rows.append(
            {
                "brain_region": region,
                "activation_A": comparison.region_scores_a[region],
                "activation_B": comparison.region_scores_b[region],
                "delta_B_minus_A": comparison.region_scores_b[region]
                - comparison.region_scores_a[region],
            }
        )
    df = pd.DataFrame(rows)

    sys_keys = list(ROI_AGGREGATES.keys())
    sys_df = pd.DataFrame(
        {
            "brain_region": [f"ROI::{k}" for k in sys_keys],
            "activation_A": [comparison.roi_aggregate_a[k] for k in sys_keys],
            "activation_B": [comparison.roi_aggregate_b[k] for k in sys_keys],
            "delta_B_minus_A": [
                comparison.roi_aggregate_b[k] - comparison.roi_aggregate_a[k]
                for k in sys_keys
            ],
        }
    )

    meta = pd.DataFrame(
        [
            {
                "brain_region": "SUMMARY",
                "activation_A": comparison.result_a.mean_activation,
                "activation_B": comparison.result_b.mean_activation,
                "delta_B_minus_A": comparison.result_b.mean_activation
                - comparison.result_a.mean_activation,
            },
            {
                "brain_region": "duration_sec_A",
                "activation_A": comparison.result_a.duration_sec,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "duration_sec_B",
                "activation_A": comparison.result_b.duration_sec,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "modalities_A",
                "activation_A": ",".join(comparison.result_a.modalities),
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "modalities_B",
                "activation_A": ",".join(comparison.result_b.modalities),
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "p-value",
                "activation_A": comparison.p_value,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "Cohen's d",
                "activation_A": comparison.effect_size,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "Winner",
                "activation_A": comparison.winner,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
            {
                "brain_region": "Confidence",
                "activation_A": comparison.confidence_label,
                "activation_B": "",
                "delta_B_minus_A": "",
            },
        ]
    )

    return pd.concat([df, sys_df, meta], ignore_index=True)


def format_winner_html(comparison: ComparisonResult) -> str:
    """Build the results panel as raw HTML.
    Uses a <style> block with #id selectors — DOMPurify allows <style> tags and
    #id beats any Svelte-scoped class selector, so colors are guaranteed.
    """
    winner = comparison.winner
    conf = comparison.confidence_label
    icon = {"Tie": "⚖️"}.get(winner, "")
    conf_cls = {"High": "conf-hi", "Medium": "conf-med", "Low": "conf-lo"}.get(conf, "conf-lo")

    mean_a = comparison.result_a.mean_activation
    mean_b = comparison.result_b.mean_activation
    delta_pct = ((mean_b - mean_a) / (mean_a + 1e-9)) * 100
    ma = ", ".join(comparison.result_a.modalities) or "—"
    mb = ", ".join(comparison.result_b.modalities) or "—"

    def cv(text: str) -> str:
        """Wrap in a code pill span."""
        return f'<span class="cv">{text}</span>'

    def trend_badge(val_a: float, val_b: float) -> str:
        if val_a > val_b:
            return '<span class="tr-a">A ▲</span>'
        if val_b > val_a:
            return '<span class="tr-b">B ▲</span>'
        return '<span class="tr-tie">Tie</span>'

    def row(*cells: str) -> str:
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def row_r(*cells: str) -> str:
        """Right-align all cells."""
        return "<tr>" + "".join(f'<td class="ra">{c}</td>' for c in cells) + "</tr>"

    def th_row(*headers: str) -> str:
        return "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

    # ── CSS (scoped to #tr-results — !important beats Gradio's Svelte !important rules) ──
    CSS = """<style>
#tr-results{font-family:Inter,sans-serif!important;font-size:.88rem!important;line-height:1.6!important;color:#0a2a40!important;background:#fff!important;opacity:1!important}
#tr-results *{color:#0a2a40!important;opacity:1!important}
#tr-results h2{font-size:1.1rem!important;font-weight:700!important;margin:0 0 3px!important;color:#0a2a40!important}
#tr-results h3{font-size:.7rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.1em!important;margin:16px 0 6px!important;color:rgba(30,80,140,.85)!important}
#tr-results p{font-size:.78rem!important;margin:3px 0!important;color:#0a2a40!important}
#tr-results table{border-collapse:collapse!important;width:100%!important;margin:4px 0 10px!important;font-size:.84rem!important}
#tr-results th{background:rgba(200,230,250,.5)!important;color:#0a2a40!important;font-weight:600!important;padding:6px 10px!important;border-bottom:1px solid rgba(100,160,210,.3)!important;font-size:.73rem!important;letter-spacing:.04em!important;text-align:left!important}
#tr-results td{background:#fff!important;color:#0a2a40!important;padding:6px 10px!important;border-bottom:.5px solid rgba(100,160,210,.18)!important;text-align:left!important}
#tr-results td.ra{text-align:right!important}
#tr-results .cv{background:rgba(200,230,250,.5)!important;color:#0a2a40!important;border-radius:4px!important;padding:1px 6px!important;font-family:'Space Mono',monospace!important;font-size:.78em!important}
#tr-results .conf-hi{color:#16a34a!important;font-weight:600!important}
#tr-results .conf-med{color:#b45309!important;font-weight:600!important}
#tr-results .conf-lo{color:#dc2626!important;font-weight:600!important}
#tr-results .tr-a{color:#1d4ed8!important;font-weight:600!important}
#tr-results .tr-b{color:#c2410c!important;font-weight:600!important}
#tr-results .tr-tie{color:#4b5563!important}
#tr-results .warn{border-left:3px solid rgba(180,83,9,.45)!important;padding:4px 10px!important;margin:4px 0!important;background:rgba(255,237,213,.4)!important;border-radius:0 6px 6px 0!important;font-size:.8rem!important;color:#0a2a40!important}
</style>"""

    parts = [CSS, '<div id="tr-results">']

    # ── Header ──
    parts.append(f'<h2>{icon} Winner: Variant {winner}</h2>')
    parts.append(f'<p>Confidence: <span class="{conf_cls}">{conf}</span></p>')

    # ── Stimuli ──
    parts.append("<h3>Stimuli</h3><table>")
    parts.append(f"<thead>{th_row('', 'Variant A', 'Variant B')}</thead><tbody>")
    parts.append(row("Modalities", cv(ma), cv(mb)))
    dur_a = f"{comparison.result_a.duration_sec:.2f} s"
    dur_b = f"{comparison.result_b.duration_sec:.2f} s"
    parts.append(row("Duration", cv(dur_a), cv(dur_b)))
    parts.append("</tbody></table>")

    # ── Comparison metrics ──
    parts.append("<h3>Comparison metrics</h3><table>")
    parts.append(f"<thead>{th_row('Metric', 'Variant A', 'Variant B')}</thead><tbody>")
    parts.append(row("Mean activation (all vertices × time)", cv(f"{mean_a:.4f}"), cv(f"{mean_b:.4f}")))
    delta_sign = "+" if mean_b > mean_a else ""
    parts.append(row("Delta B−A", "", cv(f"{delta_sign}{mean_b - mean_a:.4f}") + f" ({delta_pct:+.1f}%)"))
    parts.append(row("p-value (time-aligned)", "", cv(f"{comparison.p_value:.4f}")))
    cohens_d = "Cohen's d"
    parts.append(row(cohens_d, "", cv(f"{comparison.effect_size:.3f}")))
    ci_lo, ci_hi = comparison.bootstrap_ci
    parts.append(row("95% CI (B−A)", "", cv(f"[{ci_lo:.4f}, {ci_hi:.4f}]")))
    parts.append("</tbody></table>")
    parts.append(f"<p>{comparison.stats_note}</p>")
    parts.append(f"<p>{comparison.alignment_note}</p>")

    # ── ROI systems ──
    parts.append("<h3>ROI systems (mean activation)</h3><table>")
    parts.append(f"<thead>{th_row('System', 'A', 'B', 'Delta')}</thead><tbody>")
    for k in ROI_AGGREGATES:
        ra, rb = comparison.roi_aggregate_a[k], comparison.roi_aggregate_b[k]
        d_sign = "+" if rb > ra else ""
        parts.append(row_r(k, cv(f"{ra:.4f}"), cv(f"{rb:.4f}"), cv(f"{d_sign}{rb - ra:.4f}")))
    parts.append("</tbody></table>")
    modality_note = comparison.modality_notes.replace("- ", "").replace("**", "")
    parts.append(f"<p>{modality_note}</p>")

    # ── Proxy scores ──
    parts.append("<h3>Proxy scores</h3><table>")
    parts.append(f"<thead>{th_row('Proxy', 'A', 'B', 'Trend')}</thead><tbody>")
    for pname, va, vb in [
        ("Engagement", comparison.engagement_a, comparison.engagement_b),
        ("Attention",  comparison.attention_a,  comparison.attention_b),
        ("Emotion",    comparison.emotion_a,    comparison.emotion_b),
    ]:
        parts.append(row_r(pname, cv(f"{va:.1f}"), cv(f"{vb:.1f}"), trend_badge(va, vb)))
    parts.append("</tbody></table>")

    # ── Warnings ──
    warn_lines = list(dict.fromkeys(
        list(comparison.result_a.stimulus_warnings) + list(comparison.result_b.stimulus_warnings)
    ))
    if warn_lines:
        parts.append("<h3>Warnings</h3>")
        for w in warn_lines:
            parts.append(f'<div class="warn">⚠️ {w}</div>')

    parts.append("</div>")
    return "\n".join(parts)


# Keep old name as alias so existing callers don't break
def format_winner_markdown(comparison: ComparisonResult) -> str:
    return format_winner_html(comparison)


def _proxy_row(name: str, val_a: float, val_b: float) -> str:
    trend = "A" if val_a > val_b else ("B" if val_b > val_a else "Tie")
    return f"| {name} | `{val_a:.1f}` | `{val_b:.1f}` | {trend} |"


def validate_stimulus_spec(spec: StimulusSpec) -> Optional[str]:
    """Return user-facing error message or None. Does not create temp files."""
    mode = (spec.mode or "video").strip().lower()
    raw = (spec.text or "").strip()
    v, a, img = spec.video_path, spec.audio_path, spec.image_path

    try:
        ModalityMode(mode)
    except ValueError:
        return f"Unknown modality mode: {spec.mode}"

    if mode == ModalityMode.VIDEO.value and not v:
        return "Video mode requires a video upload (.mp4, .mov, …)."
    if mode == ModalityMode.AUDIO.value and not a:
        return "Audio mode requires an audio upload (.wav, .mp3, …)."
    if mode == ModalityMode.TEXT.value and not raw:
        return "Text mode requires non-empty text."
    if mode == ModalityMode.IMAGE.value and not img:
        return "Image mode requires an image upload (.jpg, .png, .webp)."
    if mode == ModalityMode.MULTIMODAL.value:
        n = sum(bool(x) for x in (v, a, raw, img))
        if n < 1:
            return (
                "Multimodal mode requires at least one of: video, audio, text, or image."
            )
    return None


def format_single_summary_md(
    label: str,
    result: PredictionResult,
    proxy: dict[str, float],
) -> str:
    modalities = ", ".join(result.modalities) or "—"
    warns = "\n".join(f"> ⚠️ {w}" for w in result.stimulus_warnings)
    roi = EngagementScorer.compute_roi_aggregates(result.preds)
    roi_md = "\n".join(
        f"| {k} | `{v:.4f}` |" for k, v in roi.items()
    )
    return "\n".join(
        [
            f"## 🧠 {label}",
            f"**Modalities:** `{modalities}`  ·  **Duration:** `{result.duration_sec:.2f}` s",
            f"**Mean activation:** `{result.mean_activation:.4f}`",
            "",
            "| ROI system | Mean activation |",
            "|------------|-----------------|",
            roi_md,
            "",
            "| Proxy | Score |",
            "|-------|------:|",
            f"| Engagement | `{proxy['engagement']:.1f}` |",
            f"| Attention  | `{proxy['attention']:.1f}` |",
            f"| Emotion    | `{proxy['emotion']:.1f}` |",
            "",
            warns,
        ]
    )