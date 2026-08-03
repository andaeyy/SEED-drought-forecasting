#!/usr/bin/env python3
"""validates one deployed ET-SM pair against locked 2020 archives"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from app.adapt.config import BASE_DIR, TIMESCALES
from app.adapt.inference import _predict_single
from app.adapt.models import load_models, resolve_timescale_model_dir


BACKEND_DIR = Path(__file__).resolve().parent
APP_ROOT = BACKEND_DIR.parent
REPOSITORY_ROOT = APP_ROOT.parents[1]
HOLDOUT_ROOT = REPOSITORY_ROOT / "independent_holdout_2020"
DATASET_PATH = HOLDOUT_ROOT / "data/aligned_2019_context_2020_test.npz"
MASK_PATH = HOLDOUT_ROOT / "data/raw_nldas_nonfinite_mask_2019_2020.npz"
FIXED_DATES = ("2020-01-15", "2020-06-15", "2020-12-15")
TIMESCALES_BY_INDEX = ("Weekly", "Monthly", "Seasonal")


def unpack_mask(packed: np.ndarray, flat_size: int) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        axis=-1,
        count=flat_size,
        bitorder="little",
    ).reshape((*packed.shape[:-1], 142, 95, 7)).astype(bool, copy=False)


def normalized_like_app(
    filled: np.ndarray,
    missing: np.ndarray,
    input_mu: np.ndarray,
    input_sd: np.ndarray,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(filled, dtype=np.float32).copy()
    raw[missing] = np.nan
    mu = np.asarray(input_mu, dtype=np.float32)
    sd = np.asarray(input_sd, dtype=np.float32)
    production = np.nan_to_num((raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    training = np.nan_to_num(
        np.where(missing, 0.0, (filled - mu) / sd),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    max_difference = float(np.max(np.abs(production - training)))
    if max_difference != 0.0:
        raise AssertionError(f"App and training missing-value policies differ by {max_difference}")
    return production, max_difference


def archive_path(target: str, horizon: int, model_id: str) -> Path:
    return HOLDOUT_ROOT / "predictions" / f"{target.lower()}_{horizon}d_{model_id}" / "endpoint_predictions_2020.npz"


def validate_target(
    *,
    target: str,
    model: tf.keras.Model,
    norms: dict[str, Any],
    model_id: str,
    horizon: int,
    input_days: int,
    inputs: np.ndarray,
    days: np.ndarray,
    packed_mask: np.ndarray,
    flat_size: int,
) -> dict[str, Any]:
    reference_path = archive_path(target, horizon, model_id)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_dates = np.asarray(archive["dates"])
        reference_predictions = np.asarray(archive["predicted"], dtype=np.float32)
        if str(archive["model_id"]) != model_id:
            raise AssertionError(f"Reference archive model ID differs for {target}/{horizon}")

    date_results: list[dict[str, Any]] = []
    for date_string in FIXED_DATES:
        target_hits = np.where(days == np.datetime64(date_string))[0]
        reference_hits = np.where(reference_dates == np.datetime64(date_string))[0]
        if target_hits.size != 1 or reference_hits.size != 1:
            raise AssertionError(f"Fixed date {date_string} is not unique")
        target_index = int(target_hits[0])
        start = target_index - (input_days + horizon - 1)
        indices = np.arange(start, start + input_days, dtype=np.int32)
        if start < 0 or int(indices[-1]) != target_index - horizon:
            raise AssertionError(f"Window alignment failed for {target}/{date_string}")

        window = np.asarray(inputs[indices], dtype=np.float32)
        missing = unpack_mask(packed_mask[indices], flat_size)
        normalized, normalization_difference = normalized_like_app(
            window,
            missing,
            norms["input_mu"],
            norms["input_sd"],
        )
        predicted_norm = _predict_single(model, normalized[None, ...])
        predicted = predicted_norm * float(np.asarray(norms["target_sd"]).reshape(-1)[0]) + float(
            np.asarray(norms["target_mu"]).reshape(-1)[0]
        )
        reference = reference_predictions[int(reference_hits[0])]
        difference = np.abs(np.asarray(predicted, dtype=np.float32) - reference)
        max_abs = float(np.max(difference))
        mean_abs = float(np.mean(difference))
        # bounds cuDNN batch-size drift in physical units
        tolerance = 5e-5
        if not np.isfinite(predicted).all() or max_abs > tolerance:
            raise AssertionError(
                f"{target}/{horizon}/{date_string} parity failed: max_abs={max_abs}, tolerance={tolerance}"
            )
        date_results.append(
            {
                "date": date_string,
                "window_start": str(days[start]),
                "window_end": str(days[int(indices[-1])]),
                "normalization_max_abs_difference": normalization_difference,
                "prediction_max_abs_difference": max_abs,
                "prediction_mean_abs_difference": mean_abs,
                "prediction_range": [float(np.min(predicted)), float(np.max(predicted))],
            }
        )

    return {
        "target": target,
        "model_id": model_id,
        "reference_archive": str(reference_path),
        "dates": date_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True, choices=range(3))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    timescale = TIMESCALES_BY_INDEX[args.task_index]
    spec = TIMESCALES[timescale]
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow cannot see the Slurm GPU allocation")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model_dir = resolve_timescale_model_dir(BASE_DIR, spec.parent_dirs, spec.best_arch_folder)
    loaded = load_models(model_dir)
    manifest = loaded.manifest

    with np.load(DATASET_PATH, allow_pickle=False) as archive:
        inputs = np.asarray(archive["grid_inputs"], dtype=np.float32)
        days = np.asarray(archive["days"])
        variables = [str(value) for value in archive["input_variables"].tolist()]
    with np.load(MASK_PATH, allow_pickle=False) as archive:
        packed_mask = np.asarray(archive["nonfinite_mask_packed"], dtype=np.uint8)
        flat_size = int(np.asarray(archive["flat_size"]).reshape(-1)[0])

    if inputs.shape != (730, 142, 95, 7):
        raise AssertionError(f"Unexpected aligned input shape: {inputs.shape}")
    if variables != manifest["input_variables"]:
        raise AssertionError("Input variable order differs from the deployed manifest")

    results = []
    for target, model, norms in (
        ("ET", loaded.et_model, loaded.et_norms),
        ("SM", loaded.sm_model, loaded.sm_norms),
    ):
        item = manifest["targets"][target]
        results.append(
            validate_target(
                target=target,
                model=model,
                norms=norms,
                model_id=item["model_id"],
                horizon=int(manifest["horizon_days"]),
                input_days=int(manifest["input_days"]),
                inputs=inputs,
                days=days,
                packed_mask=packed_mask,
                flat_size=flat_size,
            )
        )

    memory = {}
    try:
        memory = tf.config.experimental.get_memory_info("GPU:0")
    except Exception as exc:  # pragma: no cover - device-dependent
        memory = {"unavailable": f"{type(exc).__name__}: {exc}"}

    payload = {
        "schema": "seed_webapp.selected_model_gpu_validation/v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "timescale": timescale,
        "input_days": int(manifest["input_days"]),
        "horizon_days": int(manifest["horizon_days"]),
        "model_version": manifest["version"],
        "prediction_semantics": manifest["prediction_semantics"],
        "tensorflow_gpus": [gpu.name for gpu in gpus],
        "gpu_memory_bytes": memory,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{timescale.lower()}_validation.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
