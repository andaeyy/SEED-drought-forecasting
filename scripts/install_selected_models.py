#!/usr/bin/env python3
"""installs locked target-specific models into versioned bundles"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SELECTION_SHA256 = "7aa520837e4ae0fddf9036f424a6df5ed1249cf6835d03e5c79e6112f5af5662"
DEFAULT_VERSION = "selected_2019_v20260731"
TIMESCALE_BY_HORIZON = {7: "Weekly", 30: "Monthly", 90: "Seasonal"}
CANONICAL_FILENAMES = {
    ("ET", "checkpoint"): "keras_convlstm_et_best.keras",
    ("ET", "normalizer"): "keras_convlstm_et_norms.npz",
    ("SM", "checkpoint"): "keras_convlstm_sm_best.keras",
    ("SM", "normalizer"): "keras_convlstm_sm_norms.npz",
}
TRAINING_INPUT_VARIABLES = ["PRECTmms", "TBOT", "WIND", "QBOT", "PSRF", "FSDS", "FLDS"]
REQUIRED_INPUT_VARIABLES = {"PRECTmms", "TBOT", "QBOT", "PSRF", "WIND", "FSDS", "FLDS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def verify_source(path: Path, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected-model source: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")


def copy_verified(source: Path, destination: Path, expected_hash: str, dry_run: bool) -> str:
    verify_source(source, expected_hash)
    if destination.exists():
        destination_hash = sha256_file(destination)
        if destination_hash != expected_hash:
            raise FileExistsError(
                f"Refusing to overwrite {destination}: expected {expected_hash}, got {destination_hash}"
            )
        return "verified_existing"

    if dry_run:
        return "would_copy"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        shutil.copy2(source, temporary_path)
        copied_hash = sha256_file(temporary_path)
        if copied_hash != expected_hash:
            raise ValueError(
                f"Copied artifact failed verification for {destination}: expected {expected_hash}, got {copied_hash}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "copied"


def write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == serialized:
        return
    path.write_text(serialized, encoding="utf-8")


def install(selection_path: Path, artifact_root: Path, version: str, dry_run: bool) -> dict[str, Any]:
    selection_hash = sha256_file(selection_path)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(
            f"Locked selection hash changed: expected {EXPECTED_SELECTION_SHA256}, got {selection_hash}"
        )

    selection = load_json(selection_path)
    records = selection.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Locked selection must contain exactly six target-by-horizon records")

    by_horizon: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each selected-model record must be an object")
        target = str(record.get("target"))
        horizon = int(record.get("horizon"))
        input_days = int(record.get("input_days"))
        if target not in {"ET", "SM"} or horizon not in TIMESCALE_BY_HORIZON:
            raise ValueError(f"Unexpected target/horizon record: {target}/{horizon}")
        expected_input_days = {7: 10, 30: 45, 90: 135}[horizon]
        if input_days != expected_input_days:
            raise ValueError(
                f"Unexpected input window for {target}/{horizon}: expected {expected_input_days}, got {input_days}"
            )
        config_meta = record.get("config")
        if not isinstance(config_meta, dict):
            raise ValueError(f"Missing locked config metadata for {target}/{horizon}")
        config_path = Path(str(config_meta["path"]))
        verify_source(config_path, str(config_meta["sha256"]))
        config = load_json(config_path)
        config_inputs = config.get("input_variables")
        if config_inputs != TRAINING_INPUT_VARIABLES or set(config_inputs or []) != REQUIRED_INPUT_VARIABLES:
            raise ValueError(
                f"Unexpected training-time input channel order for {target}/{horizon}: {config_inputs}"
            )
        by_horizon.setdefault(horizon, {})[target] = record

    if any(set(targets) != {"ET", "SM"} for targets in by_horizon.values()) or set(by_horizon) != {7, 30, 90}:
        raise ValueError("Selection must contain ET and SM records for horizons 7, 30, and 90")

    version_root = artifact_root / version
    root_records: list[dict[str, Any]] = []
    actions: list[dict[str, str]] = []

    for horizon in (7, 30, 90):
        timescale = TIMESCALE_BY_HORIZON[horizon]
        bundle_dir = version_root / timescale / "target_specific"
        target_manifest: dict[str, Any] = {}

        for target in ("ET", "SM"):
            record = by_horizon[horizon][target]
            target_files: dict[str, Any] = {}
            for artifact_kind in ("checkpoint", "normalizer"):
                source_meta = record.get(artifact_kind)
                if not isinstance(source_meta, dict):
                    raise ValueError(f"Missing {artifact_kind} metadata for {target}/{horizon}")
                source = Path(str(source_meta["path"])).resolve()
                expected_hash = str(source_meta["sha256"])
                destination = bundle_dir / CANONICAL_FILENAMES[(target, artifact_kind)]
                action = copy_verified(source, destination, expected_hash, dry_run)
                actions.append({"path": str(destination), "action": action})
                target_files[artifact_kind] = {
                    "filename": destination.name,
                    "source_path": str(source),
                    "bytes": int(source_meta["bytes"]),
                    "sha256": expected_hash,
                }

            target_manifest[target] = {
                "model_id": record["model_id"],
                "family": record["family"],
                "architecture": record["architecture"],
                "trial": record["trial"],
                "input_channels": 7,
                "input_days": int(record["input_days"]),
                "horizon_days": horizon,
                "prediction_semantics": record["prediction_semantics"],
                "selection_period": "2019 validation",
                "independent_test_period": "2020",
                "validation_2019": record["validation_2019"],
                **target_files,
            }

        bundle_manifest = {
            "schema": "seed_target_specific_model_bundle/v1",
            "version": version,
            "timescale": timescale,
            "input_days": {7: 10, 30: 45, 90: 135}[horizon],
            "horizon_days": horizon,
            "prediction_semantics": "one endpoint map at lead day K",
            "input_variables": TRAINING_INPUT_VARIABLES,
            "selection": {
                "period": "2019 validation",
                "source_path": str(selection_path.resolve()),
                "source_sha256": selection_hash,
            },
            "independent_test_period": "2020",
            "targets": target_manifest,
        }
        write_json(bundle_dir / "model_manifest.json", bundle_manifest, dry_run)
        root_records.append(
            {
                "timescale": timescale,
                "horizon_days": horizon,
                "bundle_dir": str(bundle_dir),
                "manifest": str(bundle_dir / "model_manifest.json"),
                "et_model_id": target_manifest["ET"]["model_id"],
                "sm_model_id": target_manifest["SM"]["model_id"],
            }
        )

    root_manifest = {
        "schema": "seed_model_deployment/v1",
        "version": version,
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": str(selection_path.resolve()),
        "selection_sha256": selection_hash,
        "selection_period": "2019 validation",
        "independent_test_period": "2020",
        "records": root_records,
    }
    write_json(version_root / "selection_manifest.json", root_manifest, dry_run)
    return {"status": "dry_run" if dry_run else "installed", "version_root": str(version_root), "actions": actions}


def parse_args() -> argparse.Namespace:
    app_root = Path(__file__).resolve().parents[1]
    repository_root = app_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=repository_root / "independent_holdout_2020/data/selected_models_2019_locked_base_only.json",
    )
    parser.add_argument("--artifact-root", type=Path, default=app_root / "model_artifacts")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = install(args.selection.resolve(), args.artifact_root.resolve(), args.version, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
