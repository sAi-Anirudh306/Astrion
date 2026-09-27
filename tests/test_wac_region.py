"""Footprint interpolation and windowed lunar GeoTIFF extraction tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from src.ingestion.tmc_loader import row_subregion_footprint
from src.ingestion.wac_loader import signed_longitude, extract_wac_region


def metadata() -> dict:
    return {"image_height": 101, "upper_left_latitude": -1., "upper_left_longitude": 359.,
            "upper_right_latitude": -.9, "upper_right_longitude": 1.,
            "lower_left_latitude": 1., "lower_left_longitude": 359.2,
            "lower_right_latitude": 1.2, "lower_right_longitude": 1.4}


class WACRegionTests(unittest.TestCase):
    def test_independent_edges(self) -> None:
        corners = row_subregion_footprint(metadata(), 25, 76)
        self.assertAlmostEqual(corners["upper_left"]["latitude"], -.5)
        self.assertAlmostEqual(corners["upper_right"]["latitude"], -.375)
        self.assertAlmostEqual(corners["lower_left"]["longitude"], 359.15)
        self.assertAlmostEqual(corners["lower_right"]["longitude"], 1.3)

    def test_boundaries(self) -> None:
        full = row_subregion_footprint(metadata(), 0, 101)
        self.assertEqual(full["upper_left"]["latitude"], -1)
        self.assertAlmostEqual(full["lower_right"]["latitude"], 1.2)
        one = row_subregion_footprint(metadata(), 50, 51)
        self.assertEqual(one["upper_left"], one["lower_left"])
        for rows in ((-1, 50), (0, 102), (50, 50), (1.5, 50), (True, 50)):
            with self.assertRaises(ValueError):
                row_subregion_footprint(metadata(), *rows)

    def test_longitude_seam(self) -> None:
        self.assertAlmostEqual(signed_longitude(348.5), -11.5)
        self.assertEqual(signed_longitude(360), 0)
        self.assertEqual(signed_longitude(180), -180)
        with self.assertRaises(ValueError):
            signed_longitude(np.nan)
        values = metadata() | {"lower_left_longitude": 1.}
        self.assertEqual(row_subregion_footprint(values, 50, 51)["upper_left"]["longitude"], 0)

    def test_window_and_metadata_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            source_path = directory / "source.tif"
            pixels = np.arange(10000, dtype=np.uint16).reshape(100, 100)
            pixels[48:52, 48:52] = 65535
            crs = "+proj=ortho +lat_0=0 +lon_0=0 +R=1737400 +units=m +no_defs"
            with rasterio.open(source_path, "w", driver="GTiff", width=100, height=100,
                               count=1, dtype="uint16", crs=crs,
                               transform=from_origin(-50000, 50000, 1000, 1000), nodata=65535) as target:
                target.write(pixels, 1)
            before = source_path.read_bytes()
            with patch("src.ingestion.wac_loader.read_metadata", return_value=metadata()):
                report = extract_wac_region(directory / "label.xml", source_path, 25, 76, directory / "output")
            w = report["window"]
            window = Window(w["column_start"], w["row_start"],
                            w["column_stop"]-w["column_start"], w["row_stop"]-w["row_start"])
            self.assertLess(window.width * window.height, 10000)
            with rasterio.open(source_path) as source, rasterio.open(report["output_tif"]) as saved:
                np.testing.assert_array_equal(saved.read(1), source.read(1, window=window))
                self.assertEqual(saved.crs, source.crs)
                self.assertEqual(saved.res, source.res)
                self.assertEqual(saved.nodata, source.nodata)
                self.assertEqual(saved.transform, source.window_transform(window))
                self.assertEqual(report["valid_pixel_count"], int((saved.read(1) != 65535).sum()))
                for x, y in report["projected_corners"].values():
                    self.assertTrue(saved.bounds.left <= x <= saved.bounds.right)
                    self.assertTrue(saved.bounds.bottom <= y <= saved.bounds.top)
            self.assertEqual(before, source_path.read_bytes())
            with patch("src.ingestion.wac_loader.read_metadata", return_value=metadata()):
                with self.assertRaises(FileExistsError):
                    extract_wac_region(directory / "label.xml", source_path, 25, 76, directory / "output")


if __name__ == "__main__":
    unittest.main()
