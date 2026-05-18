"""Ablate direct text events against the legacy TTS/ASR text path.

This script has two modes:

1. Event-only, the default: compares text-to-events latency and schema.
2. Prediction mode, with --predict: also runs TRIBE and compares prediction
   shapes/timings and direct-vs-legacy prediction correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tribev2.demo_utils import DirectTextToEvents, TextToEvents, TribeModel


DEFAULT_TEXTS = {
    "short_story_hook": (
        "She walked into the empty room and froze. The phone on the desk was "
        "ringing, but the cord had been cut."
    ),
    "legal_identity": (
        "You have won 847 cases in a row, but you are about to lose the "
        "verdict that actually matters."
    ),
}


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _event_summary(events) -> dict[str, Any]:
    words = events[events.type == "Word"]
    return {
        "rows": int(len(events)),
        "types": {k: int(v) for k, v in events.type.value_counts().items()},
        "duration_s": float(events.stop.max() - events.start.min()),
        "word_count": int(len(words)),
        "has_audio": bool((events.type == "Audio").any()),
        "has_context_for_all_words": bool(words.context.astype(bool).all())
        if len(words)
        else False,
    }


def _time_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _predict_summary(model: TribeModel, events) -> dict[str, Any]:
    (preds, segments), seconds = _time_call(lambda: model.predict(events, verbose=False))
    return {
        "seconds": seconds,
        "preds": preds,
        "shape": tuple(preds.shape),
        "segments": len(segments),
    }


def run_text(
    name: str,
    text: str,
    cache_folder: Path,
    include_legacy: bool,
    predict: bool,
    model: TribeModel | None,
) -> dict[str, Any]:
    direct_events, direct_seconds = _time_call(
        lambda: DirectTextToEvents(text=text).get_events()
    )
    result: dict[str, Any] = {
        "name": name,
        "chars": len(text),
        "direct": {
            "event_seconds": direct_seconds,
            "events": _event_summary(direct_events),
        },
    }

    legacy_events = None
    if include_legacy:
        legacy_events, legacy_seconds = _time_call(
            lambda: TextToEvents(
                text=text,
                infra={"folder": str(cache_folder), "mode": "retry"},
            ).get_events()
        )
        result["legacy"] = {
            "event_seconds": legacy_seconds,
            "events": _event_summary(legacy_events),
        }
        result["event_speedup_x"] = legacy_seconds / max(direct_seconds, 1e-9)

    if predict:
        if model is None:
            raise RuntimeError("Prediction requested without a loaded model")
        direct_pred = _predict_summary(model, direct_events)
        result["direct"]["predict_seconds"] = direct_pred["seconds"]
        result["direct"]["prediction_shape"] = direct_pred["shape"]
        result["direct"]["prediction_segments"] = direct_pred["segments"]
        if include_legacy and legacy_events is not None:
            legacy_pred = _predict_summary(model, legacy_events)
            result["legacy"]["predict_seconds"] = legacy_pred["seconds"]
            result["legacy"]["prediction_shape"] = legacy_pred["shape"]
            result["legacy"]["prediction_segments"] = legacy_pred["segments"]
            n = min(direct_pred["preds"].shape[0], legacy_pred["preds"].shape[0])
            if n:
                result["prediction_flat_pearson_common_timesteps"] = _pearson_flat(
                    direct_pred["preds"][:n],
                    legacy_pred["preds"][:n],
                )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("cache/ablation_results"))
    parser.add_argument("--cache-folder", type=Path, default=Path("cache"))
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--text-name", default="custom")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--model", default="facebook/tribev2")
    parser.add_argument("--max-texts", type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.text_file:
        texts = {args.text_name: args.text_file.read_text(encoding="utf-8")}
    else:
        texts = DEFAULT_TEXTS
    if args.max_texts is not None:
        texts = dict(list(texts.items())[: args.max_texts])

    model = None
    load_seconds = None
    if args.predict:
        model, load_seconds = _time_call(
            lambda: TribeModel.from_pretrained(
                args.model,
                cache_folder=args.cache_folder,
                config_update={
                    "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
                    "data.num_workers": 0,
                },
            )
        )

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "prediction" if args.predict else "events",
        "include_legacy": args.legacy,
        "model_load_seconds": load_seconds,
        "texts": [
            run_text(
                name=name,
                text=text,
                cache_folder=args.cache_folder,
                include_legacy=args.legacy,
                predict=args.predict,
                model=model,
            )
            for name, text in texts.items()
        ],
    }
    out_path = args.out / f"text_path_ablation_{_now_slug()}.json"
    out_path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
