"""Tiny IIRS fixtures; approved 3.28 GB evidence boundary is mocked explicitly.

The real runner checks the actual complete cube. Unit tests do not fake a large
scientific file: they separately test streaming evidence and every policy gate.
"""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from src.ingestion.iirs_loader import (IIRSCube, read_iirs_metadata, read_envi_header,
    discover_iirs, valid_samples, representation_bands, _compatibility, _scan_words,
    EXCEPTION_STEM, EXCEPTION_MD5, EXCEPTION_BYTES)
from src.ingestion.image_loader import load_image
from src.ingestion.metadata import PDS_NS, ISDA_NS
from src.preprocessing.preprocessing import preprocess_image
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.sift import extract_sift
from src.features.rootsift import transform_sift
from src.features.phase_congruency import phase_congruency
from src.geometry.georeferencing import FootprintMapping, project_coordinates
from rasterio.crs import CRS


def fixture(root):
    path = root/'ch2_iir_nri_20210628T1759527938_d_img_d32.qub'
    data = (np.arange(3*5*7,dtype=np.uint16).reshape(3,5,7)*500).astype('<u2')
    data.tofile(path)
    bins = ''.join(f'<Band_Bin><band_number>{i+1}</band_number><center_wavelength unit="nm">{900+i*100}</center_wavelength>'
                   '<band_width unit="nm">20</band_width></Band_Bin>' for i in range(3))
    corners = ''.join(f'<isda:{c}_longitude>309</isda:{c}_longitude><isda:{c}_latitude>10</isda:{c}_latitude>'
                      for c in ('upper_left','upper_right','lower_left','lower_right'))
    path.with_suffix('.xml').write_text(f'''<Product_Observational xmlns="{PDS_NS}" xmlns:isda="{ISDA_NS}">
    <Identification_Area><logical_identifier>urn:test:{path.stem.lower()}</logical_identifier></Identification_Area>
    <Observation_Area><Primary_Result_Summary><processing_level>Raw</processing_level></Primary_Result_Summary>
    <Observing_System><Observing_System_Component><type>Instrument</type><name>imaging infrared spectrometer</name></Observing_System_Component></Observing_System>
    <Mission_Area><isda:pixel_resolution>93.94</isda:pixel_resolution>{corners}</Mission_Area></Observation_Area>
    <File_Area_Observational><File><file_name>{path.name}</file_name><file_size>210</file_size></File>
    <Array_3D_Spectrum><offset>0</offset><axes>3</axes><axis_index_order>Last Index Fastest</axis_index_order>
    <Element_Array><data_type>UnsignedLSB2</data_type></Element_Array>
    <Axis_Array><axis_name>BAND</axis_name><elements>3</elements><sequence_number>1</sequence_number><Band_Bin_Set>{bins}</Band_Bin_Set></Axis_Array>
    <Axis_Array><axis_name>LINE</axis_name><elements>5</elements><sequence_number>2</sequence_number></Axis_Array>
    <Axis_Array><axis_name>SAMPLE</axis_name><elements>7</elements><sequence_number>3</sequence_number></Axis_Array>
    </Array_3D_Spectrum></File_Area_Observational></Product_Observational>''',encoding='utf-8')
    path.with_suffix('.hdr').write_text('ENVI\nsamples = 7\nlines = 5\nbands = 3\nheader offset = 0\nfile type = ENVI Standard\ndata type = 12\ninterleave = bsq\nbyte order = 0\n')
    return path,data


class IIRSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path,self.data = fixture(self.root)
        self.cube = IIRSCube(self.path)
        self.addCleanup(self.cube.close)

    def change(self, suffix, old, new):
        p = self.path.with_suffix(suffix)
        p.write_text(p.read_text().replace(old,new))

    def test_recognition(self):
        self.assertEqual(discover_iirs(self.root),(self.path,))

    def test_status(self):
        self.assertEqual(self.cube.metadata['processing_level'],'Raw')
        self.assertEqual(self.cube.metadata['stored_quantity'],'DN')

    def test_product_id(self):
        self.assertEqual(self.cube.metadata['product_id_codes'],dict(phase='n',type='r',instrument='i'))

    def test_status_conflict(self):
        self.change('.xml','>Raw<','>Calibrated<')
        with self.assertRaises(ValueError): IIRSCube(self.path)

    def test_instrument(self):
        self.change('.xml','imaging infrared spectrometer','OHRC')
        with self.assertRaises(ValueError): IIRSCube(self.path)

    def test_invalid_xml(self):
        self.path.with_suffix('.xml').write_text('<invalid')
        with self.assertRaises(ET.ParseError): IIRSCube(self.path)

    def test_header_parse(self):
        self.assertEqual(read_envi_header(self.path.with_suffix('.hdr'))['data type'],'12')

    def test_duplicate_header_field(self):
        with self.path.with_suffix('.hdr').open('a') as f: f.write('samples = 7\n')
        with self.assertRaises(ValueError): IIRSCube(self.path)

    def test_dimensions(self):
        m=self.cube.metadata
        self.assertEqual((m['bands'],m['height'],m['width']),(3,5,7))

    def test_layout(self):
        m=self.cube.metadata
        self.assertEqual((m['interleave'],m['byte_order'],m['dtype']),('BSQ','little-endian','uint16'))

    def test_size(self):
        self.assertEqual(self.cube.metadata['expected_total_bytes'],210)

    def test_full_cube(self):
        np.testing.assert_array_equal(self.cube.read().data,self.data)

    def test_first_band(self):
        np.testing.assert_array_equal(self.cube.read(0).data,self.data[0])

    def test_middle_band(self):
        np.testing.assert_array_equal(self.cube.read(1).data,self.data[1])

    def test_last_band_unsigned(self):
        result=self.cube.read(2)
        self.assertEqual(result.data.dtype,np.uint16)
        self.assertGreater(int(result.data.max()),32767)
        np.testing.assert_array_equal(result.data,self.data[-1])

    def test_negative_band(self):
        with self.assertRaises(ValueError): self.cube.read(-1)

    def test_overflow_band(self):
        with self.assertRaises(ValueError): self.cube.read(3)

    def test_invalid_band_types(self):
        for b in (True,1.5,[],[0,0]):
            with self.subTest(b=b),self.assertRaises(ValueError): self.cube.read(b)

    def test_selected_bands(self):
        np.testing.assert_array_equal(self.cube.read([2,0]).data,self.data[[2,0]])

    def test_window_cube(self):
        np.testing.assert_array_equal(self.cube.read(window=(1,4,2,6)).data,self.data[:,1:4,2:6])

    def test_band_window(self):
        np.testing.assert_array_equal(self.cube.read(1,window=(1,4,2,6)).data,self.data[1,1:4,2:6])

    def test_selected_window(self):
        np.testing.assert_array_equal(self.cube.read([2,0],window=(1,4,2,6)).data,self.data[[2,0],1:4,2:6])

    def test_edge_windows(self):
        for window in ((0,1,0,1),(4,5,6,7)):
            r0,r1,c0,c1=window
            np.testing.assert_array_equal(self.cube.read(0,window=window).data,self.data[0,r0:r1,c0:c1])

    def test_invalid_windows(self):
        for w in ((-1,2,0,2),(0,2,-1,2),(0,6,0,2),(0,2,0,8),(0,0,0,2),(0,2,0,0),(0.,2,0,2)):
            with self.subTest(window=w),self.assertRaises(ValueError): self.cube.read(0,window=w)

    def test_truncated(self):
        self.cube.close()
        self.path.write_bytes(self.path.read_bytes()[:-2])
        with self.assertRaises(ValueError): IIRSCube(self.path)

    def test_oversized(self):
        self.cube.close()
        with self.path.open('ab') as f:f.write(b'xx')
        with self.assertRaises(ValueError): IIRSCube(self.path)

    def test_independent_output(self):
        a=self.cube.read(0).data
        a[:]=0
        np.testing.assert_array_equal(self.cube.read(0).data,self.data[0])

    def test_memory_limit(self):
        with self.assertRaises(ValueError): self.cube.read(max_bytes=10)

    def test_wavelengths(self):
        self.assertEqual([self.cube.wavelength(i) for i in range(3)],[900,1000,1100])

    def test_nearest_wavelength(self):
        self.assertEqual(self.cube.nearest_band(1020),1)
        self.assertEqual(self.cube.nearest_band(950),0)

    def test_invalid_wavelength(self):
        for w in (np.nan,np.inf,0,-1,True):
            with self.subTest(w=w),self.assertRaises(ValueError): self.cube.nearest_band(w)

    def test_missing_optional(self):
        self.assertIsNone(self.cube.metadata['radiometric_units'])
        self.assertIsNone(self.cube.metadata['spectral_resolution'])

    def test_generic_dispatch(self):
        for sensor in ('IIRS','AUTO'):
            image=load_image(self.path,sensor=sensor,bands=1,window=(0,2,0,3))
            self.assertEqual(image.sensor,'IIRS')
            np.testing.assert_array_equal(image.data,self.data[1,:2,:3])

    def test_representation_selection(self):
        self.assertEqual(representation_bands(self.cube)['single_band'],1)
        self.assertEqual(representation_bands(self.cube)['mean_bands'],[0,1,2])

    def test_mean(self):
        mean,mask=self.cube.mean([0,1,2])
        np.testing.assert_allclose(mean,self.data.mean(axis=0))
        self.assertTrue(mask.all())

    def test_bad_band_exclusion(self):
        self.cube.metadata['invalid_bands']=[1]
        mean,_=self.cube.mean([0,1])
        np.testing.assert_array_equal(mean,self.data[0])
        self.assertEqual(self.cube.nearest_band(1000),0)

    def test_nodata_mean(self):
        self.cube.metadata['nodata']=0
        a,valid=self.cube.mean([0])
        self.assertFalse(valid[0,0]);self.assertTrue(np.isnan(a[0,0]))

    def test_mask(self):
        np.testing.assert_array_equal(valid_samples(np.array([0.,1.,np.nan,np.inf]),0),[False,True,False,False])

    def test_pipeline(self):
        image=np.tile(self.cube.read(1).data,(20,10))
        normalized=preprocess_image(image).data
        scaled=normalize_resolution(normalized,93.94,100)
        features=transform_sift(extract_sift(scaled.data))
        self.assertEqual(features.descriptors.shape[1],128)
        pc=phase_congruency(scaled.data).data
        self.assertTrue(np.isfinite(pc).all())
        self.assertEqual(pc.shape,scaled.data.shape)
        points=np.array([[0.,0.],[69.,99.]])
        np.testing.assert_allclose(scaled.scaled_to_original(scaled.original_to_scaled(points),pixel_footprint=True),points,atol=1e-12)

    def test_footprint(self):
        self.assertEqual(self.cube.metadata['footprint']['upper_left']['longitude'],309)

    def test_geolocation_and_wac(self):
        corners={c:dict(longitude=lon,latitude=lat) for c,lon,lat in
                 [('upper_left',309,11),('upper_right',308,11),('lower_left',309,10),('lower_right',308,10)]}
        mapping=FootprintMapping(corners,5,7)
        p=np.array([[0.,0.],[6.,4.],[3.,2.]])
        ll=mapping.forward(p)
        back,valid,_=mapping.inverse(ll)
        self.assertTrue(valid.all());np.testing.assert_allclose(back,p,atol=1e-10)
        self.assertTrue((ll[:,0]<0).all())
        self.assertTrue(np.isfinite(project_coordinates(ll,CRS.from_dict(proj='ortho',R=1737400,lat_0=0,lon_0=0))).all())

    def test_strict_signedness(self):
        self.change('.hdr','data type = 12','data type = 2')
        with self.assertRaisesRegex(ValueError,'strict'): IIRSCube(self.path)

    def test_unrelated_product_conflict(self):
        self.change('.hdr','data type = 12','data type = 2')
        with self.assertRaisesRegex(ValueError,'outside'): IIRSCube(self.path,allow_validated_signedness_exception=True)

    def test_other_metadata_conflicts(self):
        self.cube.close()
        for old,new in [('samples = 7','samples = 8'),('interleave = bsq','interleave = bil'),('byte order = 0','byte order = 1'),('header offset = 0','header offset = 4')]:
            fixture(self.root);self.change('.hdr',old,new)
            with self.subTest(field=old),self.assertRaises(ValueError): IIRSCube(self.path,allow_validated_signedness_exception=True)

    def test_streaming_evidence(self):
        e=_scan_words(self.path)
        self.assertEqual(e['md5'],hashlib.md5(self.path.read_bytes()).hexdigest())
        self.assertEqual(e['samples'],self.data.size)
        self.assertEqual(e['sign_bit_set'],int((self.data>=32768).sum()))

    def approved_metadata(self):
        return dict(self.cube.metadata,product_id=EXCEPTION_STEM,hdr_data_type=2,height=25655,width=250,bands=256,
                    expected_total_bytes=EXCEPTION_BYTES,md5_checksum=EXCEPTION_MD5)

    def evidence(self):
        return dict(md5=EXCEPTION_MD5,sha256='unit-test evidence boundary',samples=EXCEPTION_BYTES//2,sign_bit_set=0,minimum=0,maximum=14726)

    def test_exact_exception_policy(self):
        with patch('src.ingestion.iirs_loader._scan_words',return_value=self.evidence()) as scan:
            report=_compatibility(self.path,self.approved_metadata(),True,EXCEPTION_BYTES)
            scan.assert_called_once_with(self.path)
        self.assertTrue(report['exception_applied'])
        self.assertEqual((report['xml'],report['hdr_envi_data_type']),('UnsignedLSB2',2))

    def test_every_exception_gate(self):
        changes=dict(product_id='unrelated',xml_data_type='SignedLSB2',height=25654,width=251,bands=255,
                     interleave='BIL',byte_order='big-endian',header_bytes=2,expected_total_bytes=2,md5_checksum='bad')
        for k,v in changes.items():
            m=self.approved_metadata();m[k]=v
            # Signed XML with signed HDR has no conflict; test reverse conflict.
            if k=='xml_data_type':m['hdr_data_type']=12
            with self.subTest(field=k),self.assertRaises(ValueError): _compatibility(self.path,m,True,EXCEPTION_BYTES)

    def test_corrupt_exception_evidence(self):
        for k,v in [('md5','bad'),('samples',1),('sign_bit_set',1)]:
            e=self.evidence();e[k]=v
            with self.subTest(field=k),patch('src.ingestion.iirs_loader._scan_words',return_value=e),self.assertRaises(ValueError):
                _compatibility(self.path,self.approved_metadata(),True,EXCEPTION_BYTES)

    def test_closed_cube(self):
        self.cube.close()
        with self.assertRaises(ValueError):self.cube.read(0)
        with self.assertRaises(ValueError):self.cube.mean([0])

    def test_explicit_boolean(self):
        with self.assertRaises(ValueError):IIRSCube(self.path,allow_validated_signedness_exception='yes')

    def test_missing_wavelength_fallback(self):
        self.cube.metadata['wavelengths']=[None]*3
        self.cube.metadata['wavelength_units']=None
        self.assertIsNone(self.cube.wavelength(0))
        with self.assertRaises(ValueError):self.cube.nearest_band(1000)
        self.assertEqual(representation_bands(self.cube)['single_band'],1)

    def test_no_usable_bands(self):
        self.cube.metadata['invalid_bands']=[0,1,2]
        with self.assertRaises(ValueError):self.cube.mean([0,1])
        with self.assertRaises(ValueError):representation_bands(self.cube)

    def test_malformed_wavelength_metadata(self):
        self.change('.xml','>900<','>nan<')
        with self.assertRaises(ValueError):read_iirs_metadata(self.path.with_suffix('.xml'))

    def test_raw_metadata_preserved(self):
        self.assertEqual(self.cube.metadata['raw_xml'],self.path.with_suffix('.xml').read_text())
        self.assertEqual(self.cube.metadata['raw_hdr'],self.path.with_suffix('.hdr').read_text())

    def test_xml_header_entry_points(self):
        for suffix in ('.xml','.hdr'):
            with IIRSCube(self.path.with_suffix(suffix)) as cube:
                np.testing.assert_array_equal(cube.read(0).data,self.data[0])

    def test_unsupported_validity_declaration(self):
        self.change('.xml','</Array_3D_Spectrum>','<Special_Constants><missing_constant>0</missing_constant></Special_Constants></Array_3D_Spectrum>')
        with self.assertRaisesRegex(ValueError,'validity'):IIRSCube(self.path)

    def test_processing_float_dn_mask(self):
        from scripts.iirs_wac import processing_image,statistics
        raw=np.array([[0.,100.],[np.nan,300.]],np.float32)
        mask=np.isfinite(raw)
        normalized,report=processing_image(raw,mask)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.isnan(raw[1,0]))
        self.assertEqual(normalized[1,0],0)
        self.assertEqual(statistics(raw,mask)['finite_fraction'],.75)
        self.assertIsNotNone(report['float_dn_input_mapping'])

    def test_repeated_reads(self):
        np.testing.assert_array_equal(self.cube.read([2,0],window=(0,3,0,4)).data,self.cube.read([2,0],window=(0,3,0,4)).data)


if __name__=='__main__':
    unittest.main()
