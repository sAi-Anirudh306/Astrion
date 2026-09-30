"""Optional Streamlit interaction smoke tests; no screenshot/pixel assertions."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("streamlit") is not None and (ROOT / "results/final/experiments.json").exists()


@unittest.skipUnless(AVAILABLE, "Optional UI dependency or prepared results unavailable")
class UIAppTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_landing_hero_and_tmc_quick_result(self):
        app = self.app()
        self.assertFalse(app.exception)
        markup = "\n".join(m.value for m in app.markdown)
        self.assertIn("data:image/webp;base64,", markup)
        self.assertIn("RELIABLE", markup)
        self.assertIn("569", markup)
        self.assertTrue(app.get("image"))
        self.assertIn("Checkerboard", app.selectbox[0].options)

    def test_final_product_exports_respect_reliability(self):
        app = self.app()
        for case in ("tmc", "ohrc", "iirs"):
            app.button_group(key="demo").set_value(case).run()
            self.assertFalse(app.exception)
            labels = [button.label for button in app.get("download_button")]
            self.assertIn("Download corresponding match points (CSV)", labels)
            self.assertEqual("Download registered GeoTIFF" in labels, case == "tmc")
            self.assertEqual("Download verified inliers (CSV)" in labels, case == "tmc")

    def test_all_cases_both_modes(self):
        app = self.app()
        for case in ("tmc", "ohrc", "iirs"):
            app.button_group(key="demo").set_value(case).run()
            for mode in ("Quick Result", "Watch Pipeline"):
                app.button_group(key="mode").set_value(mode).run()
                self.assertFalse(app.exception, (case, mode))
                markup = "\n".join(m.value for m in app.markdown)
                if case != "tmc" and mode == "Quick Result":
                    self.assertTrue("INSUFFICIENT SUPPORT" in markup)
                    self.assertTrue("REGISTRATION WITHHELD" in markup)
                if mode == "Watch Pipeline":
                    begin = [b for b in app.button if b.label == "Begin walkthrough"]
                    if begin:
                        begin[0].click().run()
                    self.assertEqual(len(app.select_slider[0].options), 12)
                    app.select_slider[0].set_value(6).run()
                    self.assertFalse(app.exception)

    def test_missing_optional_artifacts(self):
        with patch("src.ui.service.artifacts", return_value=()):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("No prepared images" in c.value for c in app.caption))
            app.button_group(key="mode").set_value("Watch Pipeline").run()
            self.assertFalse(app.exception)

    def test_watch_has_outputs_without_bottom_accordions(self):
        app = self.app()
        app.button_group(key="mode").set_value("Watch Pipeline").run()
        next(b for b in app.button if b.label == "Begin walkthrough").click().run()
        next(b for b in app.button if b.label == "Show final output").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.expander)
        self.assertGreater(len(app.get("image")), 2)
        markup = "\n".join(m.value for m in app.markdown)
        self.assertIn("Final ASTRION Output", markup)
        self.assertIn("RELIABLE", markup)
        app.button_group(key="mode").set_value("Quick Result").run()
        self.assertIn("Technical details & reproducibility", [e.label for e in app.expander])

    def test_missing_result_has_friendly_error(self):
        with patch("src.ui.service.discover_demos", side_effect=FileNotFoundError("Missing report")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertIn("Prepared experiment records are unavailable", app.error[0].value)

    def test_corrupt_image_has_friendly_fallback(self):
        with patch("PIL.Image.open", side_effect=OSError("Corrupt prepared image")):
            from app import display_image
            display_image.clear()
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("unavailable" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
