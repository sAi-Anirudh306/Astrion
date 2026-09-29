"""Small fixtures matching inspected calibrated OHRC and geometry storage."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

from src.ingestion.metadata import PDS_NS, ISDA_NS
from src.ingestion.ohrc_loader import (read_ohrc_metadata, load_ohrc_image, load_ohrc_geometry,
    read_geometry_label, crop_to_full)
from src.ingestion.image_loader import load_image
from src.ingestion.wac_loader import extract_geolocated_wac
from src.geometry.ohrc_geolocation import OHRCGeolocation, held_out_validation
from src.preprocessing.preprocessing import preprocess_image
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.sift import extract_sift
from src.features.rootsift import transform_sift
from src.features.phase_congruency import phase_congruency


def image_label(path, shape=(6, 8), offset=0):
    h, w = shape
    path.with_suffix('.xml').write_text(f'''<Product_Observational xmlns="{PDS_NS}" xmlns:isda="{ISDA_NS}">
    <Observation_Area><Primary_Result_Summary><processing_level>Calibrated</processing_level></Primary_Result_Summary>
    <Observing_System><Observing_System_Component><type>Instrument</type><name>orbiter high resolution camera</name>
    </Observing_System_Component></Observing_System><Mission_Area><isda:Product_Parameters>
    <isda:pixel_resolution unit="m/pixel">0.26</isda:pixel_resolution></isda:Product_Parameters></Mission_Area></Observation_Area>
    <File_Area_Observational><File><file_name>{path.name}</file_name><file_size>{offset+h*w}</file_size></File>
    <Array_2D_Image><offset>{offset}</offset><axes>2</axes><axis_index_order>Last Index Fastest</axis_index_order>
    <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
    <Axis_Array><axis_name>Line</axis_name><elements>{h}</elements><sequence_number>1</sequence_number></Axis_Array>
    <Axis_Array><axis_name>Sample</axis_name><elements>{w}</elements><sequence_number>2</sequence_number></Axis_Array>
    </Array_2D_Image></File_Area_Observational></Product_Observational>''', encoding='utf-8')


def geometry_file(path, values, header_length=30):
    body = 'Longitude,Latitude,Pixel,Scan\n'+''.join(','.join(str(v) for v in row)+'\r\n' for row in values)
    path.write_bytes(body.encode('ascii'))
    fields = ''.join(f'<Field_Delimited><name>{name}</name><field_number>{i}</field_number><data_type>{kind}</data_type></Field_Delimited>'
        for i, (name, kind) in enumerate(zip(['Longitude','Latitude','Pixel','Scan'], ['ASCII_Real']*2+['ASCII_Integer']*2), 1))
    path.with_suffix('.xml').write_text(f'''<Product_Observational xmlns="{PDS_NS}"><File_Area_Observational>
    <File><file_name>{path.name}</file_name><file_size>{len(body)}</file_size></File>
    <Header><object_length>{header_length}</object_length></Header><Table_Delimited><offset>30</offset>
    <records>{len(values)}</records><record_delimiter>Carriage-Return Line-Feed</record_delimiter><field_delimiter>Comma</field_delimiter>
    <Record_Delimited>{fields}</Record_Delimited></Table_Delimited></File_Area_Observational></Product_Observational>''')


def controls():
    y, x = np.meshgrid(np.arange(0, 60, 10), np.arange(0, 50, 10), indexing='ij')
    return np.column_stack((25+x.ravel()*1e-4+y.ravel()*1e-6,
                            -13-y.ravel()*1e-4+x.ravel()*1e-6, x.ravel(), y.ravel()))


class OHRCTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root/'sample.img'
        self.values = np.arange(48, dtype=np.uint8).reshape(6, 8)
        self.values.tofile(self.path)
        image_label(self.path)

    def change(self, tag, text):
        path = self.path.with_suffix('.xml')
        tree = ET.parse(path)
        tree.find('.//{'+PDS_NS+'}'+tag).text = text
        tree.write(path)

    def test_metadata(self):
        m = read_ohrc_metadata(self.path.with_suffix('.xml'))
        self.assertEqual((m['image_height'], m['image_width']), (6, 8))
        self.assertEqual(m['file_size_bytes'], 48)
        self.assertEqual(m['normalized']['bands'], 1)
        self.assertEqual(m['normalized']['sample_bits'], 8)
        self.assertEqual(m['normalized']['dtype'], 'uint8')
        self.assertIn('not applicable', m['normalized']['byte_order'])
        self.assertIsNone(m['normalized']['sun_elevation_deg'])
        self.assertIn('Product_Observational', m['raw_xml'])

    def test_full_exact(self):
        result = load_ohrc_image(self.path)
        np.testing.assert_array_equal(result.data, self.values)
        self.assertEqual(result.sensor, 'OHRC')
        self.assertTrue(result.metadata['normalized']['calibrated'])

    def test_browse(self):
        path = self.root/'browse.png'
        self.assertTrue(cv2.imwrite(str(path), self.values))
        image_label(path)
        tree = ET.parse(path.with_suffix('.xml'))
        tree.find('.//{'+PDS_NS+'}file_size').text = str(path.stat().st_size)
        tree.write(path.with_suffix('.xml'))
        result = load_ohrc_image(path)
        self.assertEqual(result.product_kind, 'browse')
        np.testing.assert_array_equal(result.data, self.values)

    def test_window(self):
        result = load_ohrc_image(self.path, window=(1, 5, 2, 6))
        np.testing.assert_array_equal(result.data, self.values[1:5, 2:6])
        self.assertEqual(result.metadata['subset']['first_sample_byte_offset'], 10)

    def test_first_window(self):
        np.testing.assert_array_equal(load_ohrc_image(self.path, window=(0, 1, 0, 1)).data, [[0]])

    def test_last_window(self):
        np.testing.assert_array_equal(load_ohrc_image(self.path, window=(5, 6, 7, 8)).data, [[47]])

    def test_invalid_windows(self):
        for win in [(-1, 2, 0, 2), (0, 2, -1, 2), (0, 7, 0, 2), (0, 2, 0, 9),
                    (2, 2, 0, 1), (0, 1, 2, 2), (False, 1, 0, 1), (0., 1, 0, 1)]:
            with self.subTest(window=win), self.assertRaises(ValueError): load_ohrc_image(self.path, window=win)

    def test_truncated(self):
        self.path.write_bytes(b'0'*47)
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_oversized(self):
        self.path.write_bytes(b'0'*49)
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_expected_size(self):
        self.change('file_size', '49')
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_offset(self):
        self.path.write_bytes(b'xx'+self.values.tobytes())
        image_label(self.path, offset=2)
        np.testing.assert_array_equal(load_ohrc_image(self.path).data, self.values)

    def test_memory_guard(self):
        with self.assertRaises(ValueError): load_ohrc_image(self.path, max_bytes=47)

    def test_repeated_and_no_mutation(self):
        a = load_ohrc_image(self.path)
        a.data[:] = 0
        np.testing.assert_array_equal(load_ohrc_image(self.path).data, self.values)

    def test_wrong_sensor(self):
        self.change('name', 'terrain mapping camera')
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_wrong_calibration(self):
        self.change('processing_level', 'Raw')
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_wrong_dtype(self):
        self.change('data_type', 'UnsignedLSB2')
        with self.assertRaises(ValueError): load_ohrc_image(self.path)

    def test_dispatch(self):
        for sensor in ('OHRC', 'AUTO'):
            np.testing.assert_array_equal(load_image(self.path, sensor=sensor, window=(1, 2, 2, 4)).data, self.values[1:2, 2:4])

    def test_crop_mapping(self):
        image = load_ohrc_image(self.path, window=(1, 5, 2, 6))
        p = np.array([[0, 0], [3, 3]])
        np.testing.assert_array_equal(crop_to_full(p, image.metadata['subset']), [[2, 1], [5, 4]])
        with self.assertRaises(ValueError): crop_to_full(np.array([[-1, 0]]), image.metadata['subset'])

    def test_pipeline(self):
        processing = preprocess_image(load_ohrc_image(self.path)).data
        scale = normalize_resolution(processing, .26, .52)
        self.assertEqual(scale.data.shape, (3, 4))
        self.assertEqual(scale.effective_resolution_xy, (.52, .52))
        p = np.array([[1., 1.]])
        np.testing.assert_allclose(scale.scaled_to_original(scale.original_to_scaled(p)), p)
        features = transform_sift(extract_sift(scale.data))
        self.assertEqual(features.descriptors.shape, (0, 128))
        self.assertTrue(np.isfinite(phase_congruency(scale.data).data).all())

    def test_geometry(self):
        path = self.root/'geometry.csv'
        a = controls()
        geometry_file(path, a)
        result, summary = load_ohrc_geometry(path, (51, 41))
        np.testing.assert_array_equal(result, a)
        self.assertEqual(summary['valid_rows'], 30)
        self.assertTrue(summary['complete_rectilinear_grid'])
        self.assertTrue(summary['scan_major_pixel_minor'])
        self.assertEqual(summary['ranges']['pixel'], [0, 40])
        self.assertEqual(summary['ranges']['scan'], [0, 50])
        self.assertIsNone(read_geometry_label(path.with_suffix('.xml'))['fields'][0]['unit'])

    def test_geometry_header_exception(self):
        path = self.root/'geometry.csv'
        geometry_file(path, controls(), 31)
        with self.assertRaises(ValueError): load_ohrc_geometry(path, (51, 41))
        _, summary = load_ohrc_geometry(path, (51, 41), allow_header_length_discrepancy=True)
        self.assertEqual(len(summary['discrepancies']), 1)

    def test_unapproved_header_discrepancy(self):
        path = self.root/'geometry.csv'
        geometry_file(path, controls(), 32)
        with self.assertRaises(ValueError): load_ohrc_geometry(path, (51, 41), allow_header_length_discrepancy=True)

    def test_geometry_duplicate_count(self):
        path = self.root/'geometry.csv'
        a = np.vstack((controls(), controls()[0]))
        geometry_file(path, a)
        data, summary = load_ohrc_geometry(path, (51, 41))
        self.assertEqual(summary['duplicate_records'], 1)
        with self.assertRaises(ValueError): OHRCGeolocation(data)

    def test_geometry_reject_schema_change(self):
        path = self.root/'geometry.csv'
        geometry_file(path, controls())
        label = path.with_suffix('.xml')
        label.write_text(label.read_text().replace('Longitude', 'Elevation'))
        with self.assertRaises(ValueError): load_ohrc_geometry(path, (51, 41))

    def test_geometry_reject_record_count(self):
        path = self.root/'geometry.csv'
        geometry_file(path, controls())
        label = path.with_suffix('.xml')
        label.write_text(label.read_text().replace('<records>30</records>', '<records>31</records>'))
        with self.assertRaises(ValueError): load_ohrc_geometry(path, (51, 41))

    def test_geometry_missing_file(self):
        with self.assertRaises(FileNotFoundError): load_ohrc_geometry(self.root/'absent.csv', (51, 41))

    def test_geometry_invalid_rows(self):
        path = self.root/'geometry.csv'
        a = controls()
        a[0, 0] = np.nan
        a[1, 1] = 91
        a[2, 2] = -1
        a[3, 3] = 52
        geometry_file(path, a)
        data, summary = load_ohrc_geometry(path, (51, 41))
        self.assertEqual(summary['invalid_rows'], 4)
        with self.assertRaises(ValueError): OHRCGeolocation(data)

    def test_geolocation(self):
        model = OHRCGeolocation(controls())
        p = np.array([[5., 5.], [17.2, 22.3], [0, 0], [40, 50]])
        geo = model.forward(p)
        expected = np.column_stack((25+p[:, 0]*1e-4+p[:, 1]*1e-6, -13-p[:, 1]*1e-4+p[:, 0]*1e-6))
        np.testing.assert_allclose(geo, expected, atol=1e-12)
        inverse, valid = model.inverse(geo)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(inverse, p, atol=1e-6)

    def test_geolocation_outside(self):
        model = OHRCGeolocation(controls())
        with self.assertRaises(ValueError): model.forward(np.array([[-1, 0]]))
        _, valid = model.inverse(np.array([[80., 80.]]))
        self.assertFalse(valid.any())

    def test_nonfinite_geolocation(self):
        for value in (np.nan, np.inf):
            a = controls()
            a[0, 0] = value
            with self.assertRaises(ValueError): OHRCGeolocation(a)

    def test_empty_coordinates(self):
        model = OHRCGeolocation(controls())
        self.assertEqual(model.forward(np.empty((0, 2))).shape, (0, 2))
        self.assertEqual(model.inverse(np.empty((0, 2)))[0].shape, (0, 2))

    def test_duplicate_controls(self):
        a = controls()
        a[0] = a[1]
        with self.assertRaises(ValueError): OHRCGeolocation(a)

    def test_longitude_wrap(self):
        a = controls()
        a[:, 0] = (179.999+a[:, 2]*1e-4+180)%360-180
        model = OHRCGeolocation(a)
        self.assertLess(np.ptp(model.forward(np.array([[5, 5], [35, 5]]))[:, 0]), .01)

    def test_holdout(self):
        a, b = held_out_validation(controls()), held_out_validation(controls())
        self.assertEqual(a, b)
        self.assertGreater(a['sample_count'], 0)
        self.assertLess(a['longitude_absolute_degrees']['rmse'], 1e-10)

    def test_wac_crop(self):
        path, out = self.root/'wac.tif', self.root/'crop.tif'
        crs = '+proj=ortho +lat_0=0 +lon_0=0 +R=1737400 +units=m'
        with rasterio.open(path, 'w', driver='GTiff', height=100, width=100, count=1, dtype='uint8',
                           crs=crs, transform=from_origin(-5000, 5000, 100, 100), nodata=0) as f:
            f.write(np.ones((100, 100), np.uint8), 1)
        result = extract_geolocated_wac(path, np.array([[0, 0], [.01, .01]]), out)
        self.assertTrue(result['covered'])
        with rasterio.open(out) as f:
            self.assertEqual(f.res, (100, 100))
            self.assertTrue((f.read(1) == 1).all())


if __name__ == '__main__':
    unittest.main()
