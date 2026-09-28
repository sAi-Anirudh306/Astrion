"""Synthetic footprint inversion and exact-grid scientific raster tests."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from src.geometry.georeferencing import FootprintMapping, project_to_reference, project_coordinates, lunar_geographic_crs


def corners() -> dict:
    return {"upper_left": {"longitude": -.02, "latitude": .02},
        "upper_right": {"longitude": .02, "latitude": .018},
        "lower_left": {"longitude": -.018, "latitude": -.02},
        "lower_right": {"longitude": .024, "latitude": -.019}}


class GeoreferencingTests(unittest.TestCase):
    def test_corners_center_interior(self):
        c = corners()
        mapping = FootprintMapping(c, 9, 11)
        expected = np.array([[c[k]['longitude'], c[k]['latitude']] for k in
            ('upper_left', 'upper_right', 'lower_left', 'lower_right')])
        np.testing.assert_allclose(mapping.forward(np.array([[0,0],[10,0],[0,8],[10,8]])), expected, atol=1e-12)
        np.testing.assert_allclose(mapping.forward([[5,4]])[0], expected.mean(axis=0), atol=1e-12)
        u,v = .2,.7
        expected_point = expected[0]*(1-u)*(1-v)+expected[1]*u*(1-v)+expected[2]*(1-u)*v+expected[3]*u*v
        np.testing.assert_allclose(mapping.forward([[2,5.6]])[0], expected_point, atol=1e-12)

    def test_roundtrip(self):
        mapping = FootprintMapping(corners(), 8525, 4000)
        xy = np.vstack(([[0,0],[3999,0],[0,8524],[3999,8524],[1999.5,4262]],
                        np.random.default_rng(0).random((100,2))*[3999,8524]))
        recovered, valid, residual = mapping.inverse(mapping.forward(xy))
        self.assertTrue(valid.all())
        self.assertLess(residual.max(), 1e-10)
        np.testing.assert_allclose(recovered, xy, atol=3e-5)

    def test_longitude_wrap(self):
        c = corners()
        for name, value in c.items():
            value['longitude'] = 359.8 if name.endswith('left') else .2
        mapping = FootprintMapping(c, 9, 11)
        self.assertAlmostEqual(mapping.forward([[5,4]])[0,0], 0)
        xy, valid, _ = mapping.inverse([[359.8,.02], [.2,.018]])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xy, [[0,0],[10,0]], atol=1e-8)

    def test_outside_and_invalid_models(self):
        mapping = FootprintMapping(corners(), 9, 11)
        xy, valid, _ = mapping.inverse([[10,10],[np.nan,0]])
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(xy).all())
        with self.assertRaises(ValueError): mapping.forward([[-1,0]])
        with self.assertRaises(ValueError): FootprintMapping(corners(), 1, 11)
        c = corners()
        c['lower_right'] = c['upper_left'].copy()
        with self.assertRaises(ValueError): FootprintMapping(c, 9, 11)
        with self.assertRaises(ValueError): lunar_geographic_crs(CRS.from_epsg(3857))

    def test_projected_coordinates(self):
        crs = CRS.from_string('+proj=ortho +lat_0=0 +lon_0=0 +R=1737400 +units=m')
        coordinates = project_coordinates(np.array([[0,0],[1,0]]), crs)
        np.testing.assert_allclose(coordinates[0], [0,0], atol=1e-8)
        self.assertAlmostEqual(coordinates[1,0], 1737400*np.sin(np.deg2rad(1)), places=6)

    def test_grid_mask_sampling_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, output = Path(folder)/'wac.tif', Path(folder)/'tmc.tif'
            crs = CRS.from_string('+proj=ortho +lat_0=0 +lon_0=0 +R=1737400 +units=m')
            affine = from_origin(-1000,1000,100,100)
            with rasterio.open(reference,'w',driver='GTiff',height=20,width=20,count=1,
                               dtype='uint8',crs=crs,transform=affine,nodata=0) as target:
                target.write(np.ones((20,20),np.uint8),1)
            source = np.arange(99,dtype=np.uint16).reshape(9,11)
            before = source.copy()
            result = project_to_reference(source,FootprintMapping(corners(),9,11),reference,output)
            self.assertTrue(result.valid.any())
            self.assertTrue((~result.valid).any())
            self.assertTrue(np.isfinite(result.data[result.valid]).all())
            self.assertTrue(np.isnan(result.data[~result.valid]).all())
            xy = result.source_pixels[result.valid.ravel()]
            np.testing.assert_allclose(result.data[result.valid], xy[:,0]+11*xy[:,1], atol=1e-5)
            np.testing.assert_array_equal(source,before)
            with rasterio.open(output) as saved:
                self.assertEqual(saved.crs,crs)
                self.assertEqual(saved.transform,affine)
                self.assertEqual(saved.shape,(20,20))
                self.assertTrue(np.isnan(saved.nodata))
                np.testing.assert_array_equal(saved.read_masks(1)>0,result.valid)
            with self.assertRaises(FileExistsError):
                project_to_reference(source,FootprintMapping(corners(),9,11),reference,output)
