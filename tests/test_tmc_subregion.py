"""Focused geographic row selection and bounded raw-read tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.ingestion.tmc_loader import latitude_row_range, load_tmc_latitude_region, load_tmc_image


class SubregionTests(unittest.TestCase):
    def metadata(self) -> dict:
        return {"image_height": 101, "upper_left_latitude": -50., "upper_right_latitude": -50.,
                "lower_left_latitude": -40., "lower_right_latitude": -40.}

    def test_row_calculation_both_directions(self) -> None:
        metadata = self.metadata()
        self.assertEqual(latitude_row_range(metadata, -48, -46), (20, 41))
        metadata.update(upper_left_latitude=-40., upper_right_latitude=-40.,
                        lower_left_latitude=-50., lower_right_latitude=-50.)
        self.assertEqual(latitude_row_range(metadata, -48, -46), (60, 81))

    def test_outward_envelope_and_full_extent(self) -> None:
        metadata = self.metadata()
        self.assertEqual(latitude_row_range(metadata, -50, -40), (0, 101))
        metadata["upper_right_latitude"] = -49.
        metadata["lower_right_latitude"] = -39.
        self.assertEqual(latitude_row_range(metadata, -48.05, -46.05), (9, 41))

    def test_invalid_bounds_and_footprints(self) -> None:
        for bounds in ((-46, -48), (-51, -46), (np.nan, -46), (-48, -48)):
            with self.assertRaises(ValueError):
                latitude_row_range(self.metadata(), *bounds)
        for update in ({"upper_left_latitude": None}, {"image_height": 1},
                       {"lower_left_latitude": -50.},
                       {"upper_right_latitude": -40., "lower_right_latitude": -50.}):
            metadata = self.metadata() | update
            with self.assertRaises(ValueError):
                latitude_row_range(metadata, -48, -46)

    def test_extract_real_label_synthetic_pixels_bounded_read(self) -> None:
        # Reuse a real PDS4 label structure, altering only a temporary copy.
        import xml.etree.ElementTree as ET
        from src.ingestion.metadata import NS
        root = Path(__file__).resolve().parents[1]
        fixture = root / "data/test/borrow_k/data/raw/20260629/ch2_tmc_nrf_20260629T2059373111_d_img_d18.xml"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.img"
            pixels = (np.arange(101 * 4, dtype=np.uint16) * 100).reshape(101, 4)
            path.write_bytes(b"head" + pixels.astype("<u2").tobytes())
            tree = ET.parse(fixture)
            changes = {"pds:file_name": path.name, "pds:file_size": str(path.stat().st_size), "pds:offset": "4",
                       "isda:upper_left_latitude": "-50", "isda:upper_right_latitude": "-50",
                       "isda:lower_left_latitude": "-40", "isda:lower_right_latitude": "-40"}
            for tag, value in changes.items():
                tree.find(".//" + tag, NS).text = value
            axes = tree.findall(".//pds:Axis_Array/pds:elements", NS)
            axes[0].text, axes[1].text = "101", "4"
            label = path.with_suffix(".xml")
            tree.write(label)
            before = path.read_bytes(), label.read_bytes()
            with patch("src.ingestion.tmc_loader.np.fromfile", wraps=np.fromfile) as reader:
                result = load_tmc_latitude_region(path, -48, -46)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.kwargs["count"], 21 * 4)
            self.assertEqual(reader.call_args.kwargs["offset"], 4 + 20 * 4 * 2)
            np.testing.assert_array_equal(result.data, pixels[20:41])
            self.assertEqual(result.data.dtype.str, "<u2")
            self.assertEqual(before, (path.read_bytes(), label.read_bytes()))
            for rows in ((-1, 10), (10, 102), (3, 3), (1.5, 4)):
                with self.assertRaises(ValueError):
                    load_tmc_image(path, row_range=rows)
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "File-size mismatch"):
                load_tmc_latitude_region(path, -48, -46)


if __name__ == "__main__":
    unittest.main()
