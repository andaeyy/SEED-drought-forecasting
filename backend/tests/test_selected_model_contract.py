from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from app.core.forecast_jobs import list_model_metadata


BACKEND_DIR = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_DIR.parent
MODEL_ROOT = APP_ROOT / "model_artifacts" / "selected_2019_v20260731"
EXPECTED_INPUTS = ["PRECTmms", "TBOT", "WIND", "QBOT", "PSRF", "FSDS", "FLDS"]
EXPECTED_MODELS = {
    ("Weekly", "ET"): "001_base_7var_weekly_ARconvlstm_7d_low_lr_clip",
    ("Weekly", "SM"): "007_base_7var_weekly_DEconvlstm_7d_low_lr_wd_hidden48",
    ("Monthly", "ET"): "029_base_7var_monthly_Seq2seqconvlstm_30d_compact_1layer",
    ("Monthly", "SM"): "024_base_7var_monthly_DEconvlstm_30d_compact_1layer",
    ("Seasonal", "ET"): "034_base_7var_seasonal_ARconvlstm_90d_compact_1layer",
    ("Seasonal", "SM"): "037_base_7var_seasonal_DEconvlstm_90d_low_lr_wd_hidden48",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SelectedModelContractTest(unittest.TestCase):
    def test_retrieval_windows_match_model_input_contract(self) -> None:
        from app.adapt import config

        expected = {"Weekly": 10, "Monthly": 45, "Seasonal": 135}
        self.assertEqual(
            {name: spec.default_history_days for name, spec in config.TIMESCALES.items()},
            expected,
        )

    def test_versioned_manifests_and_artifact_hashes(self) -> None:
        seen: set[tuple[str, str]] = set()
        for timescale, input_days, horizon_days in (
            ("Weekly", 10, 7),
            ("Monthly", 45, 30),
            ("Seasonal", 135, 90),
        ):
            bundle_dir = MODEL_ROOT / timescale / "target_specific"
            manifest = json.loads((bundle_dir / "model_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "seed_target_specific_model_bundle/v1")
            self.assertEqual(manifest["version"], "selected_2019_v20260731")
            self.assertEqual(manifest["input_variables"], EXPECTED_INPUTS)
            self.assertEqual(manifest["input_days"], input_days)
            self.assertEqual(manifest["horizon_days"], horizon_days)
            self.assertEqual(manifest["prediction_semantics"], "one endpoint map at lead day K")
            self.assertEqual(manifest["selection"]["period"], "2019 validation")

            for target in ("ET", "SM"):
                item = manifest["targets"][target]
                key = (timescale, target)
                seen.add(key)
                self.assertEqual(item["model_id"], EXPECTED_MODELS[key])
                self.assertEqual(item["input_channels"], 7)
                self.assertEqual(item["input_days"], input_days)
                self.assertEqual(item["horizon_days"], horizon_days)
                for kind in ("checkpoint", "normalizer"):
                    artifact = item[kind]
                    artifact_path = bundle_dir / artifact["filename"]
                    self.assertTrue(artifact_path.is_file())
                    self.assertEqual(sha256(artifact_path), artifact["sha256"])

        self.assertEqual(seen, set(EXPECTED_MODELS))

    def test_api_metadata_has_three_target_specific_pairs(self) -> None:
        records = list_model_metadata()
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record["input_variables"], EXPECTED_INPUTS)
            self.assertEqual(record["version"], "selected_2019_v20260731")
            self.assertEqual(record["selection_period"], "2019 validation")
            self.assertEqual(record["prediction_semantics"], "one endpoint map at lead day K")
            for target_key, target in (("et", "ET"), ("sm", "SM")):
                self.assertEqual(
                    record[target_key]["model_id"],
                    EXPECTED_MODELS[(record["timescale"], target)],
                )
                self.assertEqual(record[target_key]["input_channels"], 7)


if __name__ == "__main__":
    unittest.main()
