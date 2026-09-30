"""CPU-safe M21 packaging and tamper tests using the prepared small evidence."""
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluation.experiments import write_json
from src.evaluation.final_package import (build_package, comparison, create_manifest, digest,
    export_zip, point_rows, resolve, strict_json, check_protected)
from src.evaluation.final_validation import compare_packages, validate_package, validate_metrics, read_points

ROOT = Path(__file__).resolve().parents[1]
PREPARED = (ROOT / "results/milestone_20_experiments/experiments.json").is_file()


class FinalFormatTests(unittest.TestCase):
    def test_strict_json_rejects_nonfinite_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for value in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    strict_json(path)

    def test_writer_sanitizes_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.json"
            write_json(path, {"x": np.inf, "y": np.array([np.nan])})
            self.assertEqual(strict_json(path), {"x": None, "y": [None]})

    def test_match_export_preserves_float32_confidence(self):
        confidence = np.array([0.123456789], np.float32)
        rows, fields = point_rows([[1, 2]], [[3, 4]], confidence, [True])
        self.assertEqual(rows[0]["confidence"], float(confidence[0]))
        self.assertEqual(fields[-1], "inlier")

    def test_missing_membership_not_fabricated(self):
        rows, fields = point_rows([[1, 2]], [[3, 4]])
        self.assertNotIn("inlier", fields)
        self.assertNotIn("confidence", rows[0])

    def test_invalid_points_rejected(self):
        for source, reference in (([[np.nan, 0]], [[1, 2]]), ([[1, 2]], [])):
            with self.assertRaises(ValueError):
                point_rows(source, reference)

    def test_invalid_inlier_mask_rejected(self):
        with self.assertRaises(ValueError):
            point_rows([[1, 2]], [[3, 4]], inliers=[2])

    def test_path_escape_and_empty_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty").touch()
            for relative in ("../escape", "empty", "missing"):
                with self.assertRaises(ValueError):
                    resolve(root, relative)

    def test_protected_change_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source"
            path.write_bytes(b"original")
            s = path.stat()
            evidence = {"source": {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "sha256": digest(path)}}
            check_protected(root, evidence)
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                check_protected(root, evidence)


@unittest.skipUnless(PREPARED, "Prepared scientific evidence unavailable")
class FinalPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="m21_test_", dir=ROOT / "results")
        cls.package = Path(cls.temporary.name) / "package"
        # Unit fixture protects one source; production inventories all originals/history.
        path = ROOT / "configs/milestone_20.json"
        s = path.stat()
        snapshot = {"configs/milestone_20.json": {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "sha256": digest(path)}}
        with patch("src.evaluation.final_package.protected_snapshot", return_value=snapshot):
            cls.summary = build_package(ROOT, cls.package)
        cls.records = strict_json(cls.package / "experiments.json")["experiments"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_configured_matrix_imported(self):
        names = {c["name"] for c in strict_json(ROOT / "configs/milestone_20.json")["experiments"]}
        self.assertEqual({r["name"] for r in self.records}, names)
        self.assertTrue(all(r["evidence_mode"] == "imported_historical_evidence" for r in self.records))

    def test_clean_reproduction_matches_authoritative_science(self):
        report = compare_packages(ROOT / "results/final", self.package)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["experiment_count"], len(self.records))
        self.assertEqual(report["artifacts_byte_identical"], sum(len(r["final_artifacts"]) for r in self.records))

    def test_prepared_inputs_preserve_exact_regions_and_bands(self):
        inputs = strict_json(self.package / "prepared_inputs.json")
        pairs = {p["sensor"]: p for p in inputs["pairs"]}
        self.assertEqual(pairs["TMC"]["source_window"]["row_start"], 115734)
        self.assertEqual(pairs["OHRC"]["source_window"]["row_start"], 40978)
        self.assertEqual(pairs["IIRS"]["source_window"]["rows"], [12571, 13083])
        self.assertEqual([r["band_indices"] for r in pairs["IIRS"]["spectral_reductions"]], [[17], list(range(12, 24))])
        for pair in pairs.values():
            for item in pair["inputs"]:
                self.assertTrue((ROOT / item["path"]).is_file())
                if item["sha256"]:
                    self.assertEqual(digest(ROOT / item["path"]), item["sha256"])

    def test_prepared_input_tampering_rejected_before_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            shutil.copytree(self.package, target)
            inputs = strict_json(target / "prepared_inputs.json")
            inputs["pairs"][0]["inputs"][0]["size"] += 1
            (target / "prepared_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Protected file changed"):
                validate_package(target, root=ROOT, sealed=False)

    def test_standard_records_and_strict_json(self):
        for r in self.records:
            self.assertTrue(r["execution_id"])
            self.assertTrue(r["source_product"])
            self.assertTrue(r["reference_product"])
            self.assertEqual(strict_json(self.package / r["record_path"]), r)

    def test_summary_and_comparison(self):
        self.assertEqual(self.summary["experiment_count"], len(self.records))
        self.assertEqual(self.summary["reliable_experiment_count"], 1)
        self.assertEqual(self.summary["withheld_experiment_count"], 8)
        self.assertEqual(len(comparison(self.records)), len(self.records))
        self.assertIn("RAW DN", (self.package / "summary.txt").read_text())

    def test_tmc_required_deliverables_and_real_raster(self):
        report = validate_package(self.package, root=ROOT)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(v == "PASS" for v in report["sih_deliverables"].values()))

    def test_all_match_point_counts_and_optional_columns(self):
        for r in self.records:
            rows, fields = read_points(self.package / r["final_artifacts"]["match_points"]["path"])
            self.assertEqual(len(rows), r["metrics"]["candidate_matches"])
            self.assertEqual("inlier" in fields, r["name"] == "TMC saved LoFTR")

    def test_ohrc_iirs_registration_withheld(self):
        for r in self.records:
            if r["parameters"]["config"]["source_sensor"] in {"OHRC", "IIRS"}:
                self.assertEqual(r["registration_status"], "WITHHELD")
                self.assertNotIn("registered_product", r["final_artifacts"])
                self.assertTrue(r["reliability"]["reasons"])

    def test_metric_tampering_rejected(self):
        original = next(r for r in self.records if r["status"] == "RELIABLE")
        for key, value in (("verified_inliers", 100000), ("inlier_ratio", 0.1), ("rmse", -1), ("spatial_occupied_cells", 0)):
            r = copy.deepcopy(original)
            r["metrics"][key] = value
            with self.assertRaises(ValueError):
                validate_metrics(r)

    def test_manifest_covers_payload(self):
        self.assertEqual(strict_json(self.package / "manifest.json"), create_manifest(self.package, self.records))

    def test_zip_deterministic_for_same_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)
            for path in self.package.rglob("*"):
                if path.is_file() and path.suffix != ".zip":
                    target = other / path.relative_to(self.package)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            export_zip(other)
            self.assertEqual(digest(other / "ASTRION_Final_Deliverable.zip"), digest(self.package / "ASTRION_Final_Deliverable.zip"))

    def test_existing_package_never_overwritten(self):
        with self.assertRaises(ValueError):
            build_package(ROOT, self.package)

    def test_tampered_package_rejected(self):
        for filename in ("comparison.csv", "summary.json", "manifest.json", "tmc_wac/tmc_saved_loftr/registered_product.tif"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "package"
                shutil.copytree(self.package, target)
                (target / filename).write_text("corrupt")
                with self.assertRaises(Exception):
                    validate_package(target)

    def test_negative_forged_product_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            shutil.copytree(self.package, target)
            report = strict_json(target / "experiments.json")
            record = report["experiments"][0]
            record["registration_status"] = "AVAILABLE"
            (target / record["record_path"]).write_text(json.dumps(record))
            (target / "experiments.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Rejected result"):
                validate_package(target, sealed=False)


if __name__ == "__main__":
    unittest.main()
