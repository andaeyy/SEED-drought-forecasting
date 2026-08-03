from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from app.core.map_features import forecast_result_scalar_to_geojson


class ForecastScalarGeoJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = SimpleNamespace(
            lat2d=np.array([[30.0, 30.0], [30.125, 30.125]], dtype=np.float32),
            lon2d=np.array([[-100.0, -99.875], [-100.0, -99.875]], dtype=np.float32),
            et_pred=np.array([[0.5, 1.0], [1.5, np.nan]], dtype=np.float32),
            sm_pred=np.array([[0.1, 0.2], [0.3, np.nan]], dtype=np.float32),
            target_day=np.datetime64("2020-06-15"),
            horizon_days=30,
        )

    def test_et_layer_preserves_values_units_and_endpoint_metadata(self) -> None:
        collection = forecast_result_scalar_to_geojson(self.result, layer="et")

        self.assertEqual(collection["properties"]["layer"], "et")
        self.assertEqual(collection["properties"]["units"], "mm/day")
        self.assertEqual(collection["properties"]["target_day"], "2020-06-15")
        self.assertEqual(collection["properties"]["horizon_days"], 30)
        self.assertEqual(len(collection["features"]), 3)
        self.assertEqual(
            [feature["properties"]["value"] for feature in collection["features"]],
            [0.5, 1.0, 1.5],
        )

    def test_sm_layer_rejects_nonfinite_cells_and_unknown_layers(self) -> None:
        collection = forecast_result_scalar_to_geojson(self.result, layer="sm")
        self.assertEqual(collection["properties"]["units"], "m3/m3")
        self.assertEqual(len(collection["features"]), 3)

        with self.assertRaisesRegex(ValueError, "Unsupported scalar forecast layer"):
            forecast_result_scalar_to_geojson(self.result, layer="temperature")


if __name__ == "__main__":
    unittest.main()
