"""CPU-only presentation contract and prepared-result integration checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ui.service import (DEMOS, REPORT, artifacts, build_stages, discover_demos,
                            downloads, format_metric, load_results, metric_cards,
                            registration_allowed, reliability, safe_path)

ROOT = Path(__file__).resolve().parents[1]


def record(name="TMC saved LoFTR", status="RELIABLE"):
    return {"name": name, "status": status, "metrics": {"transform_available": True,
            "candidate_matches": 20, "verified_inliers": 15, "rmse": 1.25,
            "spatial_occupied_cells": 6, "spatial_distribution": {"total_cells": 16}},
            "parameters": {}, "metadata": {}, "reliability": {"status": status, "reasons": []},
            "timing": {"historical_seconds": .2}}


class UIServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / REPORT
        self.report.parent.mkdir(parents=True)
        self.write([record()])

    def write(self, records):
        self.report.write_text(json.dumps({"schema_version": 1, "experiments": records}), encoding="utf-8")

    def test_discovery_order_independent_of_record_order(self):
        self.write([record(d.experiment) for d in reversed(DEMOS)])
        self.assertEqual(discover_demos(self.root), DEMOS)

    def test_missing_case_omitted(self):
        self.assertEqual(discover_demos(self.root), (DEMOS[0],))

    def test_load_independent_cached_snapshots(self):
        first = load_results(self.root)
        first[DEMOS[0].experiment]["status"] = "FAILED"
        self.assertEqual(load_results(self.root)[DEMOS[0].experiment]["status"], "RELIABLE")

    def test_missing_report(self):
        self.report.unlink()
        with self.assertRaises(FileNotFoundError):
            load_results(self.root)

    def test_invalid_json(self):
        self.report.write_text("not JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_results(self.root)

    def test_unsupported_schema(self):
        self.report.write_text('{"schema_version": 9, "experiments": []}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_results(self.root)

    def test_duplicate_results(self):
        self.write([record(), record()])
        with self.assertRaises(ValueError):
            load_results(self.root)

    def test_reliability_conflict(self):
        r = record()
        r["metrics"]["reliability"] = {"status": "FAILED"}
        self.write([r])
        with self.assertRaises(ValueError):
            load_results(self.root)

    def test_unknown_status_rejected(self):
        self.write([record(status="MAGIC")])
        with self.assertRaises(ValueError):
            load_results(self.root)

    def test_nonfinite_and_unavailable(self):
        for value in (None, float("nan"), float("inf"), float("-inf"), "NaN", True):
            with self.subTest(value=value):
                self.assertEqual(format_metric(value), "N/A")

    def test_numeric_formatting(self):
        self.assertEqual(format_metric(.84047, percent=True), "84.05%")
        self.assertEqual(format_metric(677, digits=0), "677")
        self.assertEqual(format_metric(1.1642065, digits=3, suffix=" px"), "1.164 px")
        self.assertEqual(format_metric(.000015, digits=3, suffix=" px"), "1.50e-05 px")

    def test_metric_cards(self):
        cards = metric_cards(record())
        self.assertEqual(cards["Spatial cells"], "6 / 16")
        self.assertEqual(cards["Inlier ratio"], "N/A")
        self.assertEqual(cards["Recorded runtime"], "0.2000 s")

    def test_status_presentation(self):
        for status, color in (("RELIABLE", "good"), ("INSUFFICIENT_SUPPORT", "warning"),
                              ("NO_MODEL", "warning"), ("FAILED", "failed")):
            self.assertEqual(reliability(record(status=status))[1], color)

    def test_insufficient_fit_never_authorizes_registration(self):
        for status in ("INSUFFICIENT_SUPPORT", "NO_MODEL", "FAILED"):
            self.assertFalse(registration_allowed(record(status=status)))
        r = record()
        r["metrics"]["transform_available"] = False
        self.assertFalse(registration_allowed(r))

    def test_missing_artifacts(self):
        self.assertEqual(artifacts(self.root, DEMOS[0], record()), ())

    def test_mismatched_demo_rejected(self):
        with self.assertRaises(ValueError):
            artifacts(self.root, DEMOS[1], record())

    def test_safe_path(self):
        for value in ("../secrets", "results/../../secrets", "data/raw/input.img", str(self.root / "outside")):
            with self.subTest(path=value), self.assertRaises(ValueError):
                safe_path(self.root, value)
        self.assertEqual(safe_path(self.root, REPORT), self.report.resolve())

    def test_symlink_escape(self):
        # Simulate resolution on Windows without requiring symlink privileges.
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            if path.name == "escape.png":
                return self.root.parent / "private.png"
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve), self.assertRaises(ValueError):
            safe_path(self.root, "results/escape.png")

    def test_stage_evidence_absence_is_explicit(self):
        stages = {s.title: s for s in build_stages(record(), ())}
        self.assertEqual(stages["Ingestion"].status, "SKIPPED")
        self.assertEqual(stages["Metadata & geolocation"].status, "NOT RECORDED")
        self.assertEqual(stages["Registration"].status, "SKIPPED")
        self.assertTrue(all(s.status != "RUNNING" for s in stages.values()))

    def test_negative_stages(self):
        stages = {s.title: s for s in build_stages(record(status="INSUFFICIENT_SUPPORT"), ())}
        self.assertEqual(stages["Registration"].description, "Registration withheld")
        self.assertEqual(stages["Evaluation"].status, "WARNING")

    def test_download_existing_reports_only(self):
        result = downloads(self.root)
        self.assertEqual(list(result), ["Experiment JSON"])
        self.assertEqual(result["Experiment JSON"][1], self.report.read_bytes())


@unittest.skipUnless((ROOT / REPORT).exists(), "Prepared local demo artifacts unavailable")
class PreparedDemoTests(unittest.TestCase):
    def test_three_real_cases(self):
        records = load_results(ROOT)
        self.assertEqual(discover_demos(ROOT), DEMOS)
        self.assertEqual([records[d.experiment]["status"] for d in DEMOS],
                         ["RELIABLE", "INSUFFICIENT_SUPPORT", "INSUFFICIENT_SUPPORT"])

    def test_tmc_registration_and_real_figures(self):
        r = load_results(ROOT)[DEMOS[0].experiment]
        figures = artifacts(ROOT, DEMOS[0], r)
        names = {a.title for a in figures}
        self.assertTrue({"Verified inliers", "Before / after", "Overlay", "Checkerboard"} <= names)
        self.assertEqual(next(s for s in build_stages(r, figures) if s.title == "Registration").status, "COMPLETE")
        negative = copy.deepcopy(r)
        negative["status"] = "INSUFFICIENT_SUPPORT"
        self.assertFalse({"Before / after", "Overlay", "Checkerboard"} & {a.title for a in artifacts(ROOT, DEMOS[0], negative)})

    def test_startup_reads_only_small_prepared_files(self):
        original = Path.open
        opened = []
        def checked(path, *args, **kwargs):
            resolved = path.resolve()
            self.assertTrue(resolved.is_relative_to(ROOT / "results"))
            self.assertLess(resolved.stat().st_size, 8_000_000)
            opened.append(path)
            return original(path, *args, **kwargs)
        from src.ui.service import _json_snapshot
        _json_snapshot.cache_clear()
        with patch.object(Path, "open", checked):
            records = load_results(ROOT)
            for demo in discover_demos(ROOT):
                artifacts(ROOT, demo, records[demo.experiment])
            downloads(ROOT)
        self.assertTrue(opened)


if __name__ == "__main__":
    unittest.main()
