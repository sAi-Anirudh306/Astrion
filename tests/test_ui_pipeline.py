"""Prepared walkthrough integration: real outputs, provenance and scientific gates."""
import copy
import hashlib
from pathlib import Path
import unittest

from src.ui.pipeline import TITLES, advance_pipeline, build_pipeline_run
from src.ui.service import DEMOS, REPORT, load_results

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / REPORT).exists(), "Prepared evidence not available")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_results(ROOT)
        cls.runs = []
        for demo in DEMOS:
            run = build_pipeline_run(ROOT, demo, cls.records[demo.experiment])
            advance_pipeline(run, 11)
            cls.runs.append(run)

    def test_stage_order_and_output_descriptors(self):
        for run in self.runs:
            self.assertEqual(tuple(o.title for o in run.outputs), TITLES)
            self.assertTrue(all(o.method and o.origin and o.explanation for o in run.outputs))
            self.assertTrue(all(o.images for i, o in enumerate(run.outputs) if i != 9))

    def test_sensor_control(self):
        for run, expected in zip(self.runs, ("TMC-2", "OHRC", "IIRS")):
            self.assertIn(expected, run.outputs[0].facts["Source"])
            self.assertEqual(run.outputs[0].facts["Reference"], "LRO WAC")
        bad = copy.deepcopy(self.records[DEMOS[0].experiment])
        bad["parameters"]["config"]["source_sensor"] = "IIRS"
        with self.assertRaises(ValueError):
            build_pipeline_run(ROOT, DEMOS[0], bad)

    def test_generated_paths_safe_and_receipts_exist(self):
        for run in self.runs:
            self.assertTrue(run.directory.is_relative_to(ROOT / "results/ui_pipeline"))
            for index, output in enumerate(run.outputs):
                self.assertTrue((run.directory / f"{index:02d}_receipt.json").is_file())
                for image in output.images:
                    self.assertTrue(image.path.resolve().is_relative_to(ROOT / "results"))

    def test_no_falsely_new_historical_support(self):
        for run in self.runs:
            for index in (5, 6, 7):
                self.assertIn("Recorded evidence", run.outputs[index].origin)
        self.assertIn("Computed now", self.runs[0].outputs[3].origin)
        self.assertIn("not its original intermediate", self.runs[0].outputs[3].facts["Evidence boundary"])

    def test_reliable_registered_product(self):
        self.assertTrue((self.runs[0].directory / "registered_product.npz").is_file())
        self.assertTrue(self.runs[0].outputs[9].images)
        self.assertTrue(self.runs[0].outputs[10].images)

    def test_negative_registration_suppressed(self):
        for run in self.runs[1:]:
            self.assertFalse((run.directory / "registered_product.npz").exists())
            self.assertEqual(run.outputs[9].status, "SKIPPED")
            self.assertFalse(run.outputs[9].images)
            self.assertIn("WITHHELD", run.outputs[9].explanation)

    def test_final_evidence_and_metrics(self):
        for run in self.runs:
            final = run.outputs[-1]
            self.assertTrue(final.images)
            self.assertEqual(set(final.facts), {"Candidate matches", "Verified inliers", "Inlier ratio", "Fitted residual RMSE", "Spatial cells", "Recorded runtime"})

    def test_m20_events_and_no_duplicate_work(self):
        for run in self.runs:
            self.assertEqual([e["phase"] for e in run.events], ["started", "importing", "completed"])
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.directory.iterdir()}
            advance_pipeline(run, 11)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.directory.iterdir()})

    def test_bounded_scientific_inputs(self):
        paths = self.runs[0].provenance["inputs"]
        self.assertTrue(paths)
        self.assertTrue(all(p.startswith("data/processed/") for p in paths))
        self.assertTrue(all((ROOT / p).stat().st_size < 8_000_000 for p in paths))


if __name__ == "__main__":
    unittest.main()
