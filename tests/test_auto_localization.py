"""Synthetic localization geometry, provenance, scale and gate tests; no inference."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform

from src.geometry.georeferencing import FootprintMapping, project_coordinates, lunar_geographic_crs
from src.ingestion.wac_loader import footprint_window, extract_geolocated_wac, localize_tmc_reference
from src.evaluation.metrics import localization_overlap
from src.matching.cross_modal import support_metrics
from scripts.auto_localization import new_output, provenance, prepare_pair, register_if_reliable


class LocalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.crs = CRS.from_string('+proj=ortho +lat_0=0 +lon_0=0 +R=1737400 +units=m')
        self.grid = from_origin(-50000, 50000, 1000, 1000)
        self.path = self.root/'wac.tif'
        self.pixels = (np.arange(10000).reshape(100, 100) % 255 + 1).astype(np.uint16)
        self.mask = np.ones((100, 100), np.uint8)*255
        self.mask[45:47, 45:47] = 0
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(self.path, 'w', driver='GTiff', count=1, dtype='uint16',
                               width=100, height=100, transform=self.grid, crs=self.crs) as target:
                target.write(self.pixels, 1)
                target.write_mask(self.mask)
        self.metadata = dict(image_height=40, image_width=30, target='Moon', logical_identifier='test:TMC',
            pixel_resolution_m_per_pixel=100., upper_left_longitude=-.4, upper_left_latitude=.5,
            upper_right_longitude=.4, upper_right_latitude=.5,
            lower_left_longitude=-.4, lower_left_latitude=-.5,
            lower_right_longitude=.4, lower_right_latitude=-.5)

    def geographic_pixels(self, pixels) -> np.ndarray:
        world = np.array([self.grid @ tuple(p) for p in pixels])
        lon, lat = transform(self.crs, lunar_geographic_crs(self.crs), *world.T.tolist())
        return np.column_stack((lon, lat))

    def test_lunar_coordinate_transform_matches_analytic_sphere(self) -> None:
        p = project_coordinates(np.array([[0., 0.], [30., 0.], [0., 30.]]), self.crs)
        np.testing.assert_allclose(p, [[0, 0], [868700, 0], [0, 868700]], atol=1e-8)

    def test_densified_boundary_closed_and_longitude_seam(self) -> None:
        corners = dict(upper_left=dict(longitude=359, latitude=1), upper_right=dict(longitude=1, latitude=1),
                       lower_left=dict(longitude=359, latitude=-1), lower_right=dict(longitude=1, latitude=-1))
        mapping = FootprintMapping(corners, 20, 30)
        boundary = mapping.boundary(17)
        np.testing.assert_allclose(boundary[0], boundary[-1])
        self.assertLessEqual(np.abs(boundary[:, 0]).max(), 1)
        self.assertTrue(np.any(boundary[:, 0] == 0))
        for value in (1, True, 2.5):
            with self.assertRaises(ValueError):
                mapping.boundary(value)

    def test_outward_bounds_and_margin(self) -> None:
        points = self.geographic_pixels([[20.2, 30.2], [40.4, 60.4]])
        window, result = footprint_window(points, self.crs, self.grid, (100, 100), margin_pixels=2.5)
        self.assertEqual(result['automatic_pixel_bounds'], [17, 27, 44, 64])
        self.assertEqual((window.width, window.height), (27, 37))
        np.testing.assert_allclose(result['predicted_pixel_bounds'], [20.2, 30.2, 40.4, 60.4])

    def test_margin_clips_but_never_footprint(self) -> None:
        points = self.geographic_pixels([[1.2, 2.2], [20.4, 25.4]])
        _, result = footprint_window(points, self.crs, self.grid, (100, 100), margin_pixels=5, clip_margin=True)
        self.assertEqual(result['automatic_pixel_bounds'], [0, 0, 26, 31])
        self.assertTrue(result['margin_clipped'])
        with self.assertRaisesRegex(ValueError, 'margin not covered'):
            footprint_window(points, self.crs, self.grid, (100, 100), margin_pixels=5)
        points = self.geographic_pixels([[-.2, 2], [20, 25]])
        with self.assertRaisesRegex(ValueError, 'footprint not covered'):
            footprint_window(points, self.crs, self.grid, (100, 100), clip_margin=True)

    def test_hidden_hemisphere_and_wrong_crs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            footprint_window(np.array([[180., 0.]]), self.crs, self.grid, (100, 100))
        for crs in (None, CRS.from_epsg(3857)):
            with self.assertRaises(ValueError):
                footprint_window(np.array([[0., 0.]]), crs, self.grid, (100, 100))

    def test_invalid_coordinates_and_margin(self) -> None:
        for points in ([], [[0, 91]], [[np.nan, 0]], [[0, 0, 0]]):
            with self.assertRaises(ValueError):
                footprint_window(np.asarray(points), self.crs, self.grid, (100, 100))
        for margin in (-1, np.nan, True):
            with self.assertRaises(ValueError):
                footprint_window(np.array([[0., 0.]]), self.crs, self.grid, (100, 100), margin_pixels=margin)

    def test_exact_window_values_mask_grid_and_no_overwrite(self) -> None:
        points = self.geographic_pixels([[40.2, 40.2], [50.4, 50.4]])
        destination = self.root/'roi.tif'
        before = self.path.read_bytes()
        result = extract_geolocated_wac(self.path, points, destination, margin_pixels=2)
        with rasterio.open(destination) as saved:
            np.testing.assert_array_equal(saved.read(1), self.pixels[38:53, 38:53])
            np.testing.assert_array_equal(saved.read_masks(1), self.mask[38:53, 38:53])
            self.assertEqual(saved.transform, self.grid @ rasterio.Affine.translation(38, 38))
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(result['shape'], [15, 15])
        with self.assertRaises(FileExistsError):
            extract_geolocated_wac(self.path, points, destination)

    def test_nodata_only_window_rejected(self) -> None:
        points = self.geographic_pixels([[45.2, 45.2], [46.8, 46.8]])
        with self.assertRaisesRegex(ValueError, 'no valid'):
            extract_geolocated_wac(self.path, points, self.root/'bad.tif', margin_pixels=0)

    def localize(self) -> dict:
        with patch('src.ingestion.wac_loader.read_metadata', return_value=self.metadata):
            return localize_tmc_reference(self.root/'label.xml', self.path, self.root/'automatic_roi.tif', margin_km=2.)

    def test_localization_uses_full_metadata_and_no_old_crop(self) -> None:
        result = self.localize()
        self.assertEqual(result['source_product_id'], 'test:TMC')
        self.assertFalse(result['manual_crop_coordinates_supplied'])
        self.assertFalse(result['old_reference_crop_used'])
        self.assertEqual(len(result['geographic_boundary']), 1028)
        self.assertEqual(result['localization']['margin_pixels_rounded'], 2)
        # Geographic source changes must move the automatic ROI, without crop hints.
        for key in self.metadata:
            if key.endswith('_longitude'):
                self.metadata[key] += .5
        with patch('src.ingestion.wac_loader.read_metadata', return_value=self.metadata):
            shifted = localize_tmc_reference(self.root/'label.xml', self.path, self.root/'shifted.tif', margin_km=2.)
        self.assertGreater(shifted['window']['column_start'], result['window']['column_start'])

    def test_preparation_scale_and_scientific_values(self) -> None:
        result = self.localize()
        raw = (np.arange(1200).reshape(40, 30)*20).astype(np.uint16)
        original = raw.copy()
        projected, reference, rm, a, b, common, scale = prepare_pair(raw, result, self.root)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(projected.data.dtype, np.float32)
        self.assertGreater(np.nanmax(projected.data), 255)
        self.assertEqual(a.shape, reference.shape)
        self.assertEqual(scale['native_nominal_normalization']['scaled_shape'], (4, 3))
        self.assertEqual(scale['matching_source_scale']['factors_xy'], (1., 1.))
        self.assertEqual(scale['native_nominal_factor'], .1)
        self.assertTrue(np.all(a[~common] == 0))

    def test_localization_overlap_and_unrelated_observation(self) -> None:
        result = localization_overlap([0, 0, 20, 20], [5, 5, 15, 15], 100., same_observation=True)
        self.assertEqual(result['iou'], .25)
        self.assertEqual(result['center_displacement_pixels'], 0.)
        self.assertTrue(result['contains_comparison'])
        result = localization_overlap([0, 0, 10, 10], [30, 40, 40, 50], 100., same_observation=False)
        self.assertEqual(result['iou'], 0.)
        self.assertEqual(result['center_displacement_pixels'], 50.)
        self.assertEqual(result['center_displacement_map_km'], 5.)
        self.assertIsNone(result['localization_error_pixels'])
        self.assertFalse(result['same_observation'])

    def test_invalid_comparison_bounds(self) -> None:
        with self.assertRaises(ValueError):
            localization_overlap([0, 0, 0, 1], [0, 0, 1, 1], 100., same_observation=True)
        with self.assertRaises(ValueError):
            localization_overlap([0, 0, 1, 1], [0, 0, 1, 1], -1., same_observation=True)

    def test_output_confinement_and_overwrite(self) -> None:
        with self.assertRaises(ValueError):
            new_output(self.root, self.root/'results/final')
        target = self.root/'results/v2_auto_localization'
        new_output(self.root, target)
        with self.assertRaises(FileExistsError):
            new_output(self.root, target)

    def test_provenance_records_code_and_actual_input_hashes(self) -> None:
        (self.root/'src').mkdir()
        code = self.root/'src/test.py'
        code.write_text('original')
        with patch('scripts.auto_localization.git', return_value='test_revision'):
            first = provenance(self.root, {'reference': self.path})
            code.write_text('changed')
            second = provenance(self.root, {'reference': self.path})
        self.assertNotEqual(first['code_sha256'], second['code_sha256'])
        self.assertEqual(first['inputs'], second['inputs'])
        self.assertEqual(len(first['inputs']['reference']['sha256']), 64)

    def test_registration_withheld_for_insufficient_support(self) -> None:
        points = np.empty((0, 2))
        metrics = support_metrics(points, points, None, None, (20, 20))
        self.assertIsNone(register_if_reliable(metrics, None, None, None, self.root))
        self.assertTrue((self.root/'registration_withheld.json').is_file())
        self.assertFalse((self.root/'registered.tif').exists())

    def test_registration_allowed_only_with_existing_redundant_gate(self) -> None:
        points = np.array([(x, y) for x in (2, 7, 12, 17) for y in (2, 7, 12, 17)], float)
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        metrics = support_metrics(points, points, matrix, np.ones(16, bool), (20, 20))
        data = np.arange(400, dtype=np.float32).reshape(20, 20)
        profile = dict(driver='GTiff', width=20, height=20, dtype='float32', count=1,
                       crs=self.crs, transform=self.grid, nodata=np.nan)
        projected = SimpleNamespace(data=data, valid=np.ones((20, 20), bool), profile=profile)
        result = register_if_reliable(metrics, projected, data, projected.valid, self.root)
        np.testing.assert_array_equal(result.image, data)
        self.assertTrue((self.root/'registered.tif').is_file())


if __name__ == '__main__':
    unittest.main()
