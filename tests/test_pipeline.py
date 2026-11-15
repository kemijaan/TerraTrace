import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.api import get_available_regions, load_annual_series
from pipeline.process import PipelineError, process_file
from pipeline.run import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region = json.loads((ROOT / "config" / "regions.json").read_text())["malaysia_sarawak"]

    def test_temperature_values_and_yearly_mean(self):
        path = ROOT / "data/raw/malaysia_sarawak/temperature.csv"
        result = process_file(path, "malaysia_sarawak", self.region, "temperature")
        self.assertEqual(len(result.observations), 9497)
        self.assertEqual(result.metadata["missing_values"], 0)
        first = result.observations.iloc[0]
        self.assertAlmostEqual(first["value"], 24.581354, places=5)
        year_2025 = result.annual.set_index("year").loc[2025]
        self.assertAlmostEqual(year_2025["annual_value"], 26.260726, places=5)
        self.assertTrue(result.annual["complete_year"].all())

    def test_rainfall_values_and_yearly_total(self):
        path = ROOT / "data/raw/malaysia_sarawak/rainfall.csv"
        result = process_file(path, "malaysia_sarawak", self.region, "rainfall")
        self.assertEqual(len(result.observations), 312)
        first = result.observations.iloc[0]
        self.assertAlmostEqual(first["value"], 10.5248089 * 31, places=5)
        year_2025 = result.annual.set_index("year").loc[2025]
        self.assertAlmostEqual(year_2025["annual_value"], 3959.683584, places=4)
        self.assertTrue(result.annual["complete_year"].all())

    def test_wrong_region_bbox_is_rejected(self):
        bad_region = dict(self.region)
        bad_region["bbox"] = [100.0, 0.5, 115.5, 5.0]
        with self.assertRaisesRegex(PipelineError, "region mismatch"):
            process_file(ROOT / "data/raw/malaysia_sarawak/rainfall.csv",
                         "malaysia_sarawak", bad_region, "rainfall")

    def test_wrong_dataset_column_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "expected .* found"):
            process_file(ROOT / "data/raw/malaysia_sarawak/temperature.csv",
                         "malaysia_sarawak", self.region, "rainfall")

    def test_missing_fill_value_requires_explicit_override(self):
        original = (ROOT / "data/raw/malaysia_sarawak/rainfall.csv").read_text()
        modified = original.replace("10.5248089", "-99999", 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rainfall.csv"
            path.write_text(modified)
            with self.assertRaisesRegex(PipelineError, "missing/fill values"):
                process_file(path, "malaysia_sarawak", self.region, "rainfall")
            result = process_file(path, "malaysia_sarawak", self.region, "rainfall", allow_gaps=True)
            self.assertEqual(result.metadata["missing_values"], 1)
            self.assertFalse(bool(result.annual.set_index("year").loc[2000, "complete_year"]))

    def test_duplicate_timestamp_is_rejected(self):
        original = (ROOT / "data/raw/malaysia_sarawak/rainfall.csv").read_text()
        modified = original.replace("2000-02-01 00:00:00", "2000-01-01 00:00:00", 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rainfall.csv"
            path.write_text(modified)
            with self.assertRaisesRegex(PipelineError, "duplicate timestamps"):
                process_file(path, "malaysia_sarawak", self.region, "rainfall")

    def test_full_batch_and_app_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            test_root = Path(d)
            shutil.copytree(ROOT / "data/raw", test_root / "data/raw")
            shutil.copytree(ROOT / "config", test_root / "config")
            report = run_pipeline(test_root)
            self.assertEqual(report["dataset_count"], 2)
            self.assertEqual(report["observation_count"], 9809)
            self.assertEqual(get_available_regions(test_root), ["malaysia_sarawak"])
            annual = load_annual_series("malaysia_sarawak", "rainfall", test_root)
            self.assertEqual(len(annual), 26)
            self.assertEqual(annual["complete_year"].sum(), 26)
            catalogue = json.loads((test_root / "data/processed/catalog.json").read_text())
            self.assertEqual(catalogue["region_count"], 1)


if __name__ == "__main__":
    unittest.main()
