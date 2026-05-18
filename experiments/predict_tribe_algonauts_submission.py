"""Generate Algonauts-style prediction files from a TRIBE checkpoint.

This script is a submission-format harness, not an evaluation shortcut. Friends
season 7 and OOD fMRI labels are withheld; scoring still requires Codabench.
The output is a nested dict saved with np.save:

    {"sub-01": {"s07e01a": float32[N, 1000], ...}, ...}

TRIBE predicts fsaverage5 vertices at stimulus-driven segment times. We map
those predictions back to each subject's projected Schaefer-1000 parcels and
resample to the official target sample counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from neuralset.events.transforms import AddContextToWords, AddSentenceToWords
from neuralset.events.transforms import ChunkEvents, ExtractAudioFromVideo, RemoveMissing
from neuralset.events.utils import standardize_events

from experiments.eval_pretrained_tribe_algonauts import (
    DEFAULT_CACHE,
    DEFAULT_DATA,
    atlas_surface_labels,
    surface_to_parcels,
)
from tribev2.demo_utils import TribeModel
from tribev2.utils_fmri import TribeSurfaceProjector


SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
OOD_MOVIES = ("chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _competitors_root(data_path: Path) -> Path:
    return data_path / "download" / "algonauts_2025.competitors"


def _target_sample_path(data_path: Path, subject: str, phase: str) -> Path:
    suffix = "friends-s7" if phase == "friends-s7" else "ood"
    return (
        _competitors_root(data_path)
        / "fmri"
        / subject
        / "target_sample_number"
        / f"{subject}_{suffix}_fmri_samples.npy"
    )


def load_target_sample_counts(data_path: Path, subject: str, phase: str) -> dict[str, int]:
    path = _target_sample_path(data_path, subject, phase)
    return {str(k): int(v) for k, v in np.load(path, allow_pickle=True).item().items()}


def discover_items(
    data_path: Path,
    subject: str,
    phase: str,
    requested_items: list[str] | None = None,
    max_items: int | None = None,
) -> dict[str, int]:
    counts = load_target_sample_counts(data_path, subject, phase)
    if requested_items:
        missing = [item for item in requested_items if item not in counts]
        if missing:
            raise ValueError(f"Unknown {phase} items for {subject}: {missing}")
        counts = {item: counts[item] for item in requested_items}
    if max_items is not None:
        counts = dict(list(counts.items())[:max_items])
    return counts


def _friends_paths(data_path: Path, item: str) -> tuple[Path, Path | None]:
    season = item[1:3]
    movie = _competitors_root(data_path) / "stimuli" / "movies" / "friends" / f"s{int(season)}"
    transcript = (
        _competitors_root(data_path)
        / "stimuli"
        / "transcripts"
        / "friends"
        / f"s{int(season)}"
    )
    return movie / f"friends_{item}.mkv", transcript / f"friends_{item}.tsv"


def _ood_paths(data_path: Path, item: str) -> tuple[Path, Path | None]:
    movie = next((name for name in OOD_MOVIES if item.startswith(name)), None)
    if movie is None:
        raise ValueError(f"Cannot infer OOD movie name from item {item!r}")
    video_path = (
        _competitors_root(data_path)
        / "stimuli"
        / "movies"
        / "ood"
        / movie
        / f"task-{item}_video.mkv"
    )
    if movie == "chaplin":
        return video_path, None
    transcript_path = (
        _competitors_root(data_path)
        / "stimuli"
        / "transcripts"
        / "ood"
        / movie
        / f"ood_{item}.tsv"
    )
    return video_path, transcript_path


def stimulus_paths(data_path: Path, phase: str, item: str) -> tuple[Path, Path | None]:
    if phase == "friends-s7":
        return _friends_paths(data_path, item)
    if phase == "ood":
        return _ood_paths(data_path, item)
    raise ValueError(f"Unknown phase: {phase}")


def _read_word_events(transcript_path: Path) -> list[dict[str, Any]]:
    transcript_df = pd.read_csv(transcript_path, sep="\t")
    word_events = []
    for _, row in transcript_df.iterrows():
        words = ast.literal_eval(row["words_per_tr"])
        starts = ast.literal_eval(row["onsets_per_tr"])
        durations = ast.literal_eval(row["durations_per_tr"])
        for word, start, duration in zip(words, starts, durations):
            word_events.append(
                {
                    "type": "Word",
                    "text": word,
                    "start": float(start),
                    "duration": float(duration),
                    "stop": float(start) + float(duration),
                    "language": "english",
                    "modality": "heard",
                }
            )
    return word_events


def load_submission_events(
    data_path: Path,
    subject: str,
    phase: str,
    item: str,
    features: list[str],
) -> pd.DataFrame:
    video_path, transcript_path = stimulus_paths(data_path, phase, item)
    need_video_file = "audio" in features or "video" in features
    if need_video_file and not video_path.exists():
        raise FileNotFoundError(
            f"Need movie file for features {features}, but it is not materialized: "
            f"{video_path}"
        )
    if "text" in features and transcript_path is not None and not transcript_path.exists():
        raise FileNotFoundError(f"Transcript is not materialized: {transcript_path}")

    all_events: list[dict[str, Any]] = []
    if need_video_file:
        all_events.append({"type": "Video", "filepath": str(video_path), "start": 0.0})

    if "text" in features and transcript_path is not None:
        word_events = _read_word_events(transcript_path)
        if word_events:
            word_df = pd.DataFrame(word_events)
            all_events.append(
                {
                    "type": "Text",
                    "text": " ".join(word_df["text"].tolist()),
                    "start": float(word_df["start"].min()),
                    "duration": float(word_df["stop"].max() - word_df["start"].min()),
                    "stop": float(word_df["stop"].max()),
                    "language": "english",
                    "modality": "heard",
                }
            )
            all_events.extend(word_events)

    if not all_events:
        raise ValueError(f"No usable stimulus events for {phase}/{item} with {features}")

    events = pd.DataFrame(all_events)
    events["timeline"] = f"Algonauts2025/{subject}_{phase}_{item}"
    events["study"] = "Algonauts2025"
    events["subject"] = f"Algonauts2025/{subject}"
    events["task"] = phase
    events["movie"] = item
    events["chunk"] = item
    events["split"] = "test"
    events = standardize_events(events)

    transforms = []
    if "audio" in features:
        transforms.extend(
            [
                ExtractAudioFromVideo(),
                ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
            ]
        )
    if "text" in features and "Word" in set(events.type):
        transforms.extend(
            [
                AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
            ]
        )
    if "video" in features:
        transforms.append(
            ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30)
        )
    transforms.append(RemoveMissing())

    for transform in transforms:
        events = transform(events)
    if "audio" not in features and "Audio" in set(events.type):
        events = events[events.type != "Audio"].copy()
    if "video" not in features and "Video" in set(events.type):
        events = events[events.type != "Video"].copy()
    if "text" not in features and "Word" in set(events.type):
        events = events[~events.type.isin(["Word", "Sentence", "Text"])].copy()
    return standardize_events(events)


def subject_surface_labels(data_path: Path, subject: str) -> np.ndarray:
    atlas_path = (
        _competitors_root(data_path)
        / "fmri"
        / subject
        / "atlas"
        / f"{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    )
    atlas_img = nib.load(atlas_path)
    projector = TribeSurfaceProjector(
        mesh="fsaverage5",
        radius=3.0,
        kind="ball",
        interpolation="linear",
        center_depth=0.5,
        extract_fsaverage_from_mni=False,
    )
    label_projector = projector.model_copy(
        update={"interpolation": "nearest_most_frequent"}
    )
    return atlas_surface_labels(atlas_img, label_projector)


def resample_predictions_to_target_samples(
    pred_parcels: np.ndarray,
    sample_times: np.ndarray,
    n_samples: int,
    native_frequency: float = 1 / 1.49,
) -> np.ndarray:
    from scipy.interpolate import interp1d

    if pred_parcels.ndim != 2:
        raise ValueError(f"Expected 2-D parcel predictions, got {pred_parcels.shape}")
    if pred_parcels.shape[0] != sample_times.shape[0]:
        raise ValueError(
            "Prediction/sample-time mismatch: "
            f"{pred_parcels.shape[0]} vs {sample_times.shape[0]}"
        )
    if pred_parcels.shape[0] == 0:
        raise ValueError("Cannot resample empty predictions")
    order = np.argsort(sample_times)
    x = sample_times[order].astype(np.float64)
    y = pred_parcels[order].astype(np.float32)
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    x = x[keep]
    y = y[keep]
    if len(x) == 1:
        return np.repeat(y, n_samples, axis=0).astype(np.float32)
    target_times = np.arange(n_samples, dtype=np.float32) / native_frequency
    interp = interp1d(
        x,
        y,
        axis=0,
        bounds_error=False,
        fill_value=(y[0], y[-1]),
        assume_sorted=True,
    )
    return interp(target_times).astype(np.float32)


def fill_nonfinite_predictions(predictions: np.ndarray, method: str) -> tuple[np.ndarray, int]:
    out = predictions.astype(np.float32, copy=True)
    bad = ~np.isfinite(out)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return out, 0
    if method == "zero":
        out[bad] = 0.0
    elif method == "row-mean":
        finite = np.isfinite(out)
        counts = finite.sum(axis=1)
        sums = np.where(finite, out, 0.0).sum(axis=1)
        row_means = np.divide(
            sums,
            counts,
            out=np.zeros(out.shape[0], dtype=np.float32),
            where=counts > 0,
        ).astype(np.float32)
        rows, _ = np.where(bad)
        out[bad] = row_means[rows]
    else:
        raise ValueError(f"Unknown nonfinite fill method: {method}")
    return out, n_bad


def _prediction_cache_path(
    cache_dir: Path | None,
    checkpoint: str,
    phase: str,
    subject: str,
    item: str,
    features: list[str],
    offset_seconds: float,
) -> Path | None:
    if cache_dir is None:
        return None
    checkpoint_key = checkpoint.replace("/", "_")
    features_key = "-".join(features)
    offset_key = f"{offset_seconds:g}s".replace(".", "p")
    return (
        cache_dir
        / checkpoint_key
        / phase
        / subject
        / f"{item}_{features_key}_offset-{offset_key}_parcel1000.npy"
    )


def _surface_prediction_cache_path(
    cache_dir: Path | None,
    checkpoint: str,
    phase: str,
    item: str,
    features: list[str],
    offset_seconds: float,
) -> Path | None:
    if cache_dir is None:
        return None
    checkpoint_key = checkpoint.replace("/", "_")
    features_key = "-".join(features)
    offset_key = f"{offset_seconds:g}s".replace(".", "p")
    return (
        cache_dir
        / checkpoint_key
        / phase
        / "_stimulus_surface"
        / f"{item}_{features_key}_offset-{offset_key}_fsaverage5.npz"
    )


def get_surface_prediction_for_item(
    *,
    model: Any,
    data_path: Path,
    checkpoint: str,
    phase: str,
    subject: str,
    item: str,
    features: list[str],
    target_offset_seconds: float,
    prediction_cache_dir: Path | None,
    stimulus_cache: dict[tuple[str, str, tuple[str, ...], float], dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, int, str, float]:
    """Load or compute subject-independent TRIBE surface predictions for an item."""
    key = (phase, item, tuple(features), float(target_offset_seconds))
    if key in stimulus_cache:
        cached = stimulus_cache[key]
        return (
            cached["preds_surface"],
            cached["sample_times"],
            int(cached["n_segments"]),
            "memory",
            0.0,
        )

    surface_cache_path = _surface_prediction_cache_path(
        prediction_cache_dir,
        checkpoint,
        phase,
        item,
        features,
        target_offset_seconds,
    )
    if surface_cache_path is not None and surface_cache_path.exists():
        loaded = np.load(surface_cache_path)
        payload = {
            "preds_surface": loaded["preds_surface"].astype(np.float32),
            "sample_times": loaded["sample_times"].astype(np.float32),
            "n_segments": np.asarray(loaded["n_segments"]),
        }
        stimulus_cache[key] = payload
        return (
            payload["preds_surface"],
            payload["sample_times"],
            int(payload["n_segments"]),
            "disk",
            0.0,
        )

    events = load_submission_events(data_path, subject, phase, item, features)
    pred_start = time.perf_counter()
    preds_surface, segments = model.predict(events, verbose=False)
    predict_seconds = time.perf_counter() - pred_start
    sample_times = np.array(
        [float(getattr(segment, "start", idx)) for idx, segment in enumerate(segments)],
        dtype=np.float32,
    )
    sample_times = sample_times + float(target_offset_seconds)
    payload = {
        "preds_surface": preds_surface.astype(np.float32),
        "sample_times": sample_times.astype(np.float32),
        "n_segments": np.asarray(int(preds_surface.shape[0]), dtype=np.int64),
    }
    stimulus_cache[key] = payload
    if surface_cache_path is not None:
        surface_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(surface_cache_path, **payload)
    return (
        payload["preds_surface"],
        payload["sample_times"],
        int(payload["n_segments"]),
        "computed",
        predict_seconds,
    )


def _zip_file(source: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source, arcname=source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--phase", choices=["friends-s7", "ood"], default="friends-s7")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--items", nargs="+")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["text", "audio", "video"],
        default=["text", "audio", "video"],
    )
    parser.add_argument("--target-offset-seconds", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-items-per-subject", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prediction-cache-dir", type=Path)
    parser.add_argument(
        "--nonfinite-fill",
        choices=["row-mean", "zero"],
        default="row-mean",
        help=(
            "How to fill projection-missing parcels before writing a submission. "
            "The approximate fsaverage5-to-Schaefer projection leaves some parcels "
            "empty; Codabench-shaped files cannot contain NaNs."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/submissions/tribe_algonauts_predictions.npy"),
    )
    parser.add_argument("--zip-out", type=Path)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("cache/submissions/tribe_algonauts_predictions_manifest.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    requested = {
        subject: discover_items(
            args.data_path,
            subject,
            args.phase,
            requested_items=args.items,
            max_items=args.max_items_per_subject,
        )
        for subject in args.subjects
    }

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "phase": args.phase,
        "features": args.features,
        "subjects": args.subjects,
        "target_offset_seconds": args.target_offset_seconds,
        "nonfinite_fill": args.nonfinite_fill,
        "dry_run": args.dry_run,
        "items": {},
        "caveat": (
            "Submission-shaped predictions only; Friends S7/OOD labels are withheld "
            "and official scoring requires Codabench."
        ),
    }

    if args.dry_run:
        for subject, counts in requested.items():
            manifest["items"][subject] = {}
            for item, n_samples in counts.items():
                video_path, transcript_path = stimulus_paths(args.data_path, args.phase, item)
                manifest["items"][subject][item] = {
                    "n_target_samples": n_samples,
                    "video_exists": video_path.exists(),
                    "transcript_exists": None if transcript_path is None else transcript_path.exists(),
                    "video_path": str(video_path),
                    "transcript_path": None if transcript_path is None else str(transcript_path),
                }
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
        )
        print(args.manifest_out)
        print(json.dumps(_jsonable(manifest), indent=2))
        return

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    model_start = time.perf_counter()
    model = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=args.cache_folder,
        cluster=None,
        device=args.device,
        config_update={
            "data.features_to_use": list(args.features),
            "data.num_workers": 0,
            "data.batch_size": 1,
            "enable_progress_bar": False,
        },
    )
    manifest["model_load_seconds"] = time.perf_counter() - model_start

    predictions: dict[str, dict[str, np.ndarray]] = {}
    labels_by_subject = {
        subject: subject_surface_labels(args.data_path, subject)
        for subject in args.subjects
    }
    stimulus_cache: dict[tuple[str, str, tuple[str, ...], float], dict[str, np.ndarray]] = {}

    for subject, counts in requested.items():
        predictions[subject] = {}
        manifest["items"][subject] = {}
        for item, n_samples in counts.items():
            item_start = time.perf_counter()
            cache_path = _prediction_cache_path(
                args.prediction_cache_dir,
                args.checkpoint,
                args.phase,
                subject,
                item,
                list(args.features),
                args.target_offset_seconds,
            )
            if cache_path is not None and cache_path.exists():
                pred_target = np.load(cache_path).astype(np.float32)
                used_cache = True
                n_segments = None
                predict_seconds = 0.0
                surface_prediction_source = None
            else:
                (
                    preds_surface,
                    sample_times,
                    n_segments,
                    surface_prediction_source,
                    predict_seconds,
                ) = get_surface_prediction_for_item(
                    model=model,
                    data_path=args.data_path,
                    checkpoint=args.checkpoint,
                    phase=args.phase,
                    subject=subject,
                    item=item,
                    features=list(args.features),
                    target_offset_seconds=float(args.target_offset_seconds),
                    prediction_cache_dir=args.prediction_cache_dir,
                    stimulus_cache=stimulus_cache,
                )
                pred_parcels = surface_to_parcels(
                    preds_surface.astype(np.float32), labels_by_subject[subject]
                )
                pred_target = resample_predictions_to_target_samples(
                    pred_parcels=pred_parcels,
                    sample_times=sample_times,
                    n_samples=n_samples,
                )
                used_cache = False

            pred_target, n_filled = fill_nonfinite_predictions(
                pred_target, args.nonfinite_fill
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, pred_target)

            if pred_target.shape != (n_samples, 1000):
                raise ValueError(
                    f"{subject}/{item} prediction shape {pred_target.shape} "
                    f"does not match expected {(n_samples, 1000)}"
                )
            if not np.isfinite(pred_target).all():
                raise ValueError(f"{subject}/{item} prediction contains NaN/inf")

            predictions[subject][item] = pred_target.astype(np.float32)
            manifest["items"][subject][item] = {
                "shape": list(pred_target.shape),
                "n_target_samples": n_samples,
                "n_prediction_segments": n_segments,
                "n_nonfinite_values_filled": n_filled,
                "used_prediction_cache": used_cache,
                "surface_prediction_source": surface_prediction_source,
                "predict_seconds": predict_seconds,
                "total_item_seconds": time.perf_counter() - item_start,
            }

    manifest["runtime_seconds"] = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, predictions)
    manifest["out"] = str(args.out)
    if args.zip_out is not None:
        _zip_file(args.out, args.zip_out)
        manifest["zip_out"] = str(args.zip_out)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    print(args.out)
    print(args.manifest_out)
    print(json.dumps(_jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
