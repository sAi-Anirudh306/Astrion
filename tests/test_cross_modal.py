"""Offline synthetic M18 integration and conservative engineering gates."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from src.matching.cross_modal import (PairConfig,ReliabilityPolicy,run_pair,
    support_metrics,assess_reliability,prepare_representation)
from src.ingestion.image_loader import ImageData


def scene():
    rng=np.random.default_rng(19)
    a=np.zeros((160,160),np.uint8)
    for x,y,r,v in zip(rng.integers(8,152,90),rng.integers(8,152,90),rng.integers(2,7,90),rng.integers(80,255,90)):
        cv2.circle(a,(int(x),int(y)),int(r),int(v),-1)
    return a


class CrossModalTests(unittest.TestCase):
    def test_sensor_configuration(self):
        for sensor in ('TMC','OHRC','IIRS'):
            self.assertIn('IIRS' if sensor=='IIRS' else 'optical',PairConfig(sensor,'WAC',1,2).modality)

    def test_invalid_configuration(self):
        for args in [('other','WAC',1,1),('IIRS','WAC',0,1),('IIRS','WAC',1,float('nan')),('TMC','WAC',1,1,'new_model')]:
            with self.subTest(args=args),self.assertRaises(ValueError):PairConfig(*args)

    def test_invalid_policy(self):
        for kw in [dict(minimum_inliers=3),dict(minimum_occupied_cells=1),dict(maximum_condition_number=float('inf'))]:
            with self.assertRaises(ValueError):ReliabilityPolicy(**kw)

    def test_intensity_synthetic_success(self):
        a=scene(); b=np.sqrt(a.astype(np.float32)/255)
        r=run_pair(a,b,PairConfig('IIRS','WAC',100,100))
        self.assertEqual(r['metrics']['reliability']['status'],'RELIABLE')
        self.assertGreaterEqual(r['metrics']['spatial_occupied_cells'],6)
        np.testing.assert_allclose(r['transform_original_grids'],[[1,0,0],[0,1,0]],atol=1)

    def test_phase_path(self):
        a=scene();r=run_pair(a,a,PairConfig('TMC','WAC',100,100,'phase_congruency'))
        self.assertGreater(r['metrics']['source_keypoints'],12)
        self.assertEqual(r['metrics']['reliability']['status'],'RELIABLE')

    def test_scale_integration(self):
        a=scene();large=cv2.resize(a,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
        r=run_pair(large,a,PairConfig('OHRC','WAC',50,100))
        self.assertEqual(r['source_scale']['scaled_shape'],(160,160))
        self.assertTrue(r['metrics']['transform_available'])
        np.testing.assert_allclose(np.array(r['transform_original_grids'])[:,:2],np.eye(2)*.5,atol=.02)

    def test_reference_downsample(self):
        a=scene();r=run_pair(a,np.repeat(np.repeat(a,2,0),2,1),PairConfig('IIRS','WAC',100,50))
        np.testing.assert_allclose(np.array(r['transform_original_grids'])[:,:2],np.eye(2)*2,atol=.02)

    def test_three_point_fit_not_reliable(self):
        p=np.array([[2,2],[80,3],[3,80]],float)
        m=support_metrics(p,p,np.array([[1,0,0],[0,1,0]]),np.ones(3,bool),(100,100))
        self.assertEqual(m['rmse'],0)
        self.assertEqual(m['reliability']['status'],'INSUFFICIENT_SUPPORT')

    def test_empty_support(self):
        r=run_pair(np.zeros((32,32),np.uint8),np.zeros((32,32),np.uint8),PairConfig('OHRC','WAC',1,1))
        self.assertEqual(r['metrics']['verified_inliers'],0)
        self.assertIsNone(r['metrics']['rmse'])
        self.assertEqual(r['metrics']['reliability']['status'],'INSUFFICIENT_SUPPORT')

    def test_no_model(self):
        p=np.zeros((20,2));r=support_metrics(p,p,None,None,(100,100))
        self.assertEqual(r['reliability']['status'],'NO_MODEL')

    def test_degenerate_transform(self):
        a=np.array([[i,j] for i in range(4) for j in range(4)],float)
        b=np.zeros_like(a)
        with self.assertRaises(ValueError):support_metrics(a,b,np.zeros((2,3)),np.ones(16,bool),(100,100))

    def test_mask_validation(self):
        with self.assertRaises(ValueError):prepare_representation(scene(),np.ones((1,1),bool))

    def test_nodata_copy(self):
        a=scene().astype(float);a[0,0]=np.nan
        b,m=prepare_representation(a)
        self.assertTrue(np.isnan(a[0,0]));self.assertFalse(m[0,0]);self.assertTrue(np.isfinite(b).all())

    def test_cube_requires_explicit_reduction(self):
        with self.assertRaises(ValueError):prepare_representation(np.zeros((3,4,5)))

    def test_reproducibility(self):
        a=scene();config=PairConfig('TMC','WAC',1,1)
        x,y=run_pair(a,a,config),run_pair(a,a,config)
        self.assertEqual(x['transform_original_grids'],y['transform_original_grids'])
        self.assertEqual(x['source_points_original'],y['source_points_original'])

    def test_inconsistent_saved_mask(self):
        p=np.zeros((4,2))
        with self.assertRaises(ValueError):support_metrics(p,p+10,np.array([[1,0,0],[0,1,0]]),np.ones(4,bool),(100,100))

    def test_finite_statistics_required(self):
        p=np.array([[x,y] for x in (10,35,60,85) for y in (10,35,60,85)],float)
        m=support_metrics(p,p,np.array([[1,0,0],[0,1,0]]),np.ones(16,bool),(100,100))
        self.assertEqual(m['reliability']['status'],'RELIABLE')
        m['rmse']=None
        self.assertNotEqual(assess_reliability(m)['status'],'RELIABLE')

    def test_ingestion_metadata_compatibility(self):
        for sensor in ('TMC','OHRC','IIRS'):
            a=ImageData(scene(),{'selected_bands':[17] if sensor=='IIRS' else None},Path('synthetic'),Path('synthetic.xml'),sensor,'raw')
            r=run_pair(a,scene(),PairConfig(sensor,'WAC',100,100))
            self.assertEqual(r['input_provenance']['source']['sensor'],sensor)
            self.assertTrue(r['metrics']['transform_available'])

    def test_conflicting_sensor_metadata(self):
        a=ImageData(scene(),{},Path('synthetic'),Path('synthetic.xml'),'IIRS','raw')
        with self.assertRaises(ValueError):run_pair(a,scene(),PairConfig('OHRC','WAC',100,100))


if __name__=='__main__':unittest.main()
