"""Evaluate pretrained TRIBE on local Algonauts parcellated labels.

This is a compatibility/evaluation harness, not a leaderboard reproduction.
The public TRIBE checkpoint predicts fsaverage5 surface vertices, while the
local Algonauts files are Schaefer-1000 parcels. This script can either project
the local parcels to fsaverage5, or approximately summarize the model surface
prediction back into Schaefer parcels. Treat both paths as approximate until
validated against the official competition preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from neuralset.events.transforms import AddContextToWords, AddSentenceToWords
from neuralset.events.transforms import ChunkEvents, ExtractAudioFromVideo, RemoveMissing
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import TribeModel
from tribev2.studies.algonauts2025 import Algonauts2025
from tribev2.utils_fmri import TribeSurfaceProjector


DEFAULT_DATA = Path("/home/ripper/data/tribe_benchmarks")
DEFAULT_CACHE = Path("/home/ripper/data/tribe_runs/cache/pretrained_tribe_eval")


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


def _chunk_to_episode(chunk: str) -> str:
    return chunk[:3] if chunk.startswith("e") else chunk


def _timeline_for(subject: str, movie: str, chunk: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "task": "friends",
        "movie": movie,
        "chunk": chunk,
        "run": 0,
    }


def load_stimulus_events(
    data_path: Path,
    subject: str,
    movie: str,
    chunks: list[str],
    features: list[str],
) -> pd.DataFrame:
    study = Algonauts2025(path=data_path)
    frames = []
    for chunk in chunks:
        timeline = _timeline_for(subject, movie, chunk)
        events = study._load_timeline_events(timeline)
        events = events[events.type != "Fmri"].copy()
        name = f"Algonauts2025/{subject}_{movie}{chunk}"
        events["timeline"] = name
        events["study"] = "Algonauts2025"
        events["subject"] = f"Algonauts2025/{subject}"
        events["task"] = "friends"
        events["movie"] = movie
        events["chunk"] = chunk
        events["episode"] = f"{movie}{_chunk_to_episode(chunk)}"
        frames.append(events)
    events = standardize_events(pd.concat(frames, ignore_index=True))
    events = prune_unrequested_events(events, features, before_transforms=True)

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
    events = prune_unrequested_events(events, features, before_transforms=False)
    return standardize_events(events)


def prune_unrequested_events(
    events: pd.DataFrame,
    features: list[str],
    *,
    before_transforms: bool,
) -> pd.DataFrame:
    """Drop event types that are not needed for the requested modality set."""
    keep_video = "video" in features or (before_transforms and "audio" in features)
    mask = pd.Series(True, index=events.index)
    if not keep_video and "Video" in set(events.type):
        mask &= events.type != "Video"
    if "audio" not in features and "Audio" in set(events.type):
        mask &= events.type != "Audio"
    if "text" not in features:
        mask &= ~events.type.isin(["Word", "Sentence", "Text"])
    return events.loc[mask].copy()


def load_parcel_data(
    data_path: Path, subject: str, movie: str, chunk: str
) -> np.ndarray:
    study = Algonauts2025(path=data_path)
    timeline = _timeline_for(subject, movie, chunk)
    h5_path = study._get_fmri_filepath(timeline)
    key_fragment = f"{movie[1:]}{chunk}"
    with h5py.File(h5_path, "r") as h5:
        keys = [key for key in h5.keys() if key_fragment in key]
        if len(keys) != 1:
            raise ValueError(f"Expected one key for {key_fragment}, got {keys}")
        return h5[keys[0]][:].astype(np.float32)


def parcels_to_surface(
    parcel_data: np.ndarray,
    atlas_img: nib.Nifti1Image,
    projector: TribeSurfaceProjector,
    batch_size: int,
) -> np.ndarray:
    labels = atlas_img.get_fdata().astype(np.int32)
    mask = labels > 0
    parcel_index = labels[mask] - 1
    if parcel_data.shape[1] <= int(parcel_index.max()):
        raise ValueError(
            f"Parcel data has {parcel_data.shape[1]} columns but atlas needs "
            f"parcel index {int(parcel_index.max())}."
        )

    projected = []
    for start in range(0, parcel_data.shape[0], batch_size):
        stop = min(start + batch_size, parcel_data.shape[0])
        values = parcel_data[start:stop]
        vol = np.zeros(labels.shape + (stop - start,), dtype=np.float32)
        vol[mask, :] = values[:, parcel_index].T
        img = nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header)
        projected.append(projector.apply(img).astype(np.float32))
    return np.concatenate(projected, axis=1)


def atlas_surface_labels(
    atlas_img: nib.Nifti1Image,
    projector: TribeSurfaceProjector,
) -> np.ndarray:
    labels = atlas_img.get_fdata().astype(np.float32)
    img = nib.Nifti1Image(labels[..., None], atlas_img.affine, atlas_img.header)
    projected = projector.apply(img)[:, 0]
    rounded = np.rint(projected).astype(np.int32)
    rounded[~np.isfinite(projected)] = 0
    rounded[(rounded < 0) | (rounded > 1000)] = 0
    return rounded


def surface_to_parcels(
    surface: np.ndarray,
    surface_labels: np.ndarray,
    n_parcels: int = 1000,
) -> np.ndarray:
    if surface.ndim != 2:
        raise ValueError(f"Expected surface predictions to be 2-D, got {surface.shape}")
    if surface.shape[1] != surface_labels.shape[0]:
        raise ValueError(
            "Prediction/label vertex mismatch: "
            f"{surface.shape[1]} vs {surface_labels.shape[0]}"
        )
    out = np.full((surface.shape[0], n_parcels), np.nan, dtype=np.float32)
    for parcel in range(1, n_parcels + 1):
        mask = surface_labels == parcel
        if mask.any():
            out[:, parcel - 1] = np.nanmean(surface[:, mask], axis=1)
    return out


def resample_parcels_to_segments(
    parcel_data: np.ndarray,
    native_frequency: float,
    sample_times: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import interp1d

    native_times = np.arange(parcel_data.shape[0], dtype=np.float32) / native_frequency
    interp = interp1d(
        native_times,
        parcel_data,
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return interp(sample_times).astype(np.float32)


def resample_surface_to_segments(
    surface: np.ndarray,
    native_frequency: float,
    sample_times: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import interp1d

    native_times = np.arange(surface.shape[1], dtype=np.float32) / native_frequency
    interp = interp1d(
        native_times,
        surface,
        axis=1,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return interp(sample_times).T.astype(np.float32)


def vertexwise_pearson(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    mask = np.isfinite(pred).all(axis=0) & np.isfinite(target).all(axis=0)
    pred = pred[:, mask].astype(np.float64)
    target = target[:, mask].astype(np.float64)
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(pred, axis=0) * np.linalg.norm(target, axis=0)
    valid = denom > 0
    corr = np.full(mask.sum(), np.nan, dtype=np.float64)
    corr[valid] = (pred[:, valid] * target[:, valid]).sum(axis=0) / denom[valid]
    finite = np.isfinite(corr)
    return {
        "mean": float(np.nanmean(corr)),
        "median": float(np.nanmedian(corr)),
        "n_vertices_total": int(pred.shape[1]),
        "n_vertices_finite": int(finite.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--movie", default="s01")
    parser.add_argument("--chunks", nargs="+", default=["e01a"])
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["text", "audio", "video"],
        default=["text"],
    )
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--target-offset-seconds", type=float, default=5.0)
    parser.add_argument(
        "--metric-space",
        choices=["surface", "parcel"],
        default="surface",
        help=(
            "surface: project Schaefer parcel targets to fsaverage5; "
            "parcel: summarize fsaverage5 predictions back to Schaefer parcels."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("cache/ablation_results/pretrained_tribe_eval.json"))
    parser.add_argument(
        "--arrays-out",
        type=Path,
        help="Optional NPZ path for metric-space predictions, targets, sample times, and finite rows.",
    )
    args = parser.parse_args()

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
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
    load_seconds = time.perf_counter() - start

    events = load_stimulus_events(
        args.data_path, args.subject, args.movie, args.chunks, list(args.features)
    )
    pred_start = time.perf_counter()
    preds, segments = model.predict(events, verbose=False)
    predict_seconds = time.perf_counter() - pred_start
    sample_times = np.array(
        [float(getattr(segment, "start", i)) for i, segment in enumerate(segments)],
        dtype=np.float32,
    )
    sample_times = sample_times + float(args.target_offset_seconds)

    study = Algonauts2025(path=args.data_path)
    atlas_path = (
        args.data_path
        / "download/algonauts_2025.competitors/fmri"
        / args.subject
        / "atlas"
        / f"{args.subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
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

    target_parts = []
    target_start = time.perf_counter()
    if args.metric_space == "parcel":
        label_projector = projector.model_copy(
            update={"interpolation": "nearest_most_frequent"}
        )
        surface_labels = atlas_surface_labels(atlas_img, label_projector)
        preds_metric = surface_to_parcels(preds, surface_labels)
    else:
        surface_labels = None
        preds_metric = preds

    time_cursor = 0.0
    for chunk in args.chunks:
        parcels = load_parcel_data(args.data_path, args.subject, args.movie, chunk)
        chunk_duration = parcels.shape[0] / study._FREQUENCY
        in_chunk = (sample_times >= time_cursor) & (
            sample_times < time_cursor + chunk_duration
        )
        if in_chunk.any():
            local_times = sample_times[in_chunk] - time_cursor
            if args.metric_space == "parcel":
                values = resample_parcels_to_segments(
                    parcels, study._FREQUENCY, local_times
                )
            else:
                surface = parcels_to_surface(
                    parcels,
                    atlas_img=atlas_img,
                    projector=projector,
                    batch_size=args.target_batch_size,
                )
                values = resample_surface_to_segments(
                    surface, study._FREQUENCY, local_times
                )
            target_parts.append((in_chunk, values))
        time_cursor += chunk_duration
    target = np.full_like(preds_metric, np.nan, dtype=np.float32)
    for in_chunk, values in target_parts:
        target[in_chunk] = values
    target_seconds = time.perf_counter() - target_start

    finite_rows = np.isfinite(target).any(axis=1)
    metrics = vertexwise_pearson(preds_metric[finite_rows], target[finite_rows])
    if args.metric_space == "parcel" and surface_labels is not None:
        label_counts = np.bincount(surface_labels, minlength=1001)[1:]
        n_units = int(np.count_nonzero(label_counts))
    else:
        label_counts = None
        n_units = None
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "features": args.features,
        "metric_space": args.metric_space,
        "subject": args.subject,
        "movie": args.movie,
        "chunks": args.chunks,
        "target_projection": (
            "Schaefer1000 parcels -> MNI label volume -> standard fsaverage5 vol_to_surf"
            if args.metric_space == "surface"
            else "standard fsaverage5 prediction -> nearest projected Schaefer1000 parcel averages"
        ),
        "target_projection_caveat": (
            "Approximate local evaluation; not the official Algonauts surface "
            "preprocessing and not directly comparable to the public leaderboard."
        ),
        "target_offset_seconds": args.target_offset_seconds,
        "n_prediction_samples": int(preds.shape[0]),
        "prediction_shape": list(preds.shape),
        "metric_prediction_shape": list(preds_metric.shape),
        "target_shape": list(target.shape),
        "n_finite_target_rows": int(finite_rows.sum()),
        "n_nonempty_projected_parcels": n_units,
        "metrics": metrics,
        "load_seconds": load_seconds,
        "predict_seconds": predict_seconds,
        "target_projection_seconds": target_seconds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    if args.arrays_out is not None:
        args.arrays_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.arrays_out,
            preds=preds_metric.astype(np.float32),
            target=target.astype(np.float32),
            sample_times=sample_times.astype(np.float32),
            finite_rows=finite_rows,
        )
        result["arrays_path"] = str(args.arrays_out)
        args.out.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
