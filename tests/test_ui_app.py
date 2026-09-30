"""Optional Streamlit interaction smoke tests; no screenshot/pixel assertions."""
import importlib.util
from pathlib import Path
import sys
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
        from streamlit.testing.v1 import AppTest

        def exports(case):
            from pathlib import Path
            from app import final_products
            from src.ui.service import DEMOS, load_results
            demo = next(d for d in DEMOS if d.key == case)
            final_products(load_results(Path.cwd())[demo.experiment])

        for case in ("tmc", "ohrc", "iirs"):
            app = AppTest.from_function(exports, args=(case,), default_timeout=30).run()
            self.assertFalse(app.exception)
            labels = [button.label for button in app.get("download_button")]
            self.assertIn("Download corresponding match points (CSV)", labels)
            self.assertEqual("Download registered GeoTIFF" in labels, case == "tmc")
            self.assertEqual("Download verified inliers (CSV)" in labels, case == "tmc")

    def test_pitch_uses_tmc_in_both_modes(self):
        app = self.app()
        app.session_state["demo"] = "iirs"  # Ignore selection left by the previous UI.
        for mode in ("Quick Result", "Show Pipeline"):
            app.button_group(key="mode").set_value(mode).run()
            self.assertFalse(app.exception, mode)
            self.assertNotIn("demo", [group.key for group in app.button_group])
            markup = "\n".join(m.value for m in app.markdown)
            self.assertIn("Chandrayaan-2 TMC-2", markup)
            self.assertNotIn("REGISTRATION WITHHELD", markup)

    def test_missing_optional_artifacts(self):
        # Load the controller before patching so its module-level import is not
        # permanently bound to this test's empty-artifact mock.
        import src.ui.pipeline

        with patch("src.ui.service.artifacts", return_value=()):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("No prepared images" in c.value for c in app.caption))
            app.button_group(key="mode").set_value("Show Pipeline").run()
            self.assertFalse(app.exception)

    def test_pitch_pipeline_completes_and_returns_to_quick_result(self):
        app = self.app()
        app.button_group(key="mode").set_value("Show Pipeline").run()
        next(b for b in app.button if b.label == "Run ASTRION").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["pitch_done_tmc"], [e.value for e in app.error])
        self.assertGreater(len(app.get("image")), 2)
        markup = "\n".join(m.value for m in app.markdown)
        self.assertIn("ASTRION evidence ready", markup)
        self.assertIn("RELIABLE", markup)
        app.button_group(key="mode").set_value("Quick Result").run()
        self.assertIn("Technical details & reproducibility", [e.label for e in app.expander])

    def test_slider_is_inside_overlay_tab_with_static_artifact(self):
        if importlib.util.find_spec("streamlit_image_comparison") is None:
            self.skipTest("Optional slider package unavailable")
        app = self.app()
        comparison = next(e for e in app.expander if e.label == "Registration comparison")
        self.assertEqual([tab.label for tab in comparison.tabs],
                         ["Before / after", "Overlay", "Checkerboard"])
        before, overlay, checkerboard = comparison.tabs
        self.assertFalse(before.get("iframe"))
        self.assertFalse(checkerboard.get("iframe"))
        slider_html = overlay.get("iframe")[0].proto.srcdoc
        self.assertLess(slider_html.index("Before Registration"),
                        slider_html.index("After Registration"))
        self.assertEqual(len(overlay.get("image")), 1)
        self.assertIn("**Final Registration Overlay**", [m.value for m in overlay.markdown])

    def test_missing_slider_package_shows_side_by_side_views(self):
        with patch.dict(sys.modules, {"streamlit_image_comparison": None}):
            app = self.app()
        self.assertFalse(app.exception)
        overlay = next(tab for tab in app.tabs if tab.label == "Overlay")
        self.assertFalse(overlay.get("iframe"))
        captions = [image.caption for item in overlay.get("image") for image in item.proto.imgs]
        self.assertEqual(captions[:2], ["Before Registration", "After Registration"])
        self.assertEqual(len(captions), 3)  # Includes the original overlay.

    def test_missing_tmc_does_not_select_another_sensor(self):
        from src.ui.service import DEMOS
        with patch("src.ui.service.discover_demos", return_value=DEMOS[1:]):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertIn("validated TMC-2", app.info[0].value)
        self.assertFalse(app.get("download_button"))

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
