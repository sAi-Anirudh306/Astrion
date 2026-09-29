"""Small deterministic M20 configuration/import/live/event contract tests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.evaluation.experiments import (ExperimentConfig,ExperimentRunner,load_configs,
    write_json,json_safe,comparison_records)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.path=self.root/'prior.json'
        self.payload=dict(name='control',config={'source_sensor':'TMC'},metrics=dict(candidate_matches=677,verified_inliers=569,
            spatial_occupied_cells=15,rmse=1.164207,median_residual=.88369,maximum_residual=2.940514,runtime_seconds=10.,
            reliability=dict(status='RELIABLE',reasons=[],policy={'minimum_inliers':12})),evidence_origin='saved control')
        write_json(self.path,dict(experiments=[self.payload],limitations=['Not geographic accuracy']))
        self.config=ExperimentConfig('control','import','prior.json','control')

    def test_config_roundtrip(self):
        self.assertEqual(ExperimentConfig.from_dict(self.config.to_dict()),self.config)

    def test_invalid_config(self):
        for d in [dict(name='',mode='import'),dict(name='x',mode='other'),dict(name='x',mode='live',pair={}),dict(name='x',mode='import',extra=True)]:
            with self.subTest(d=d),self.assertRaises(ValueError):ExperimentConfig.from_dict(d)

    def test_config_file_order(self):
        p=self.root/'config.json'
        write_json(p,dict(schema_version=1,experiments=[dict(self.config.to_dict(),name='z'),dict(self.config.to_dict(),name='a')]))
        self.assertEqual([c.name for c in load_configs(p)],['a','z'])

    def test_duplicate_names(self):
        p=self.root/'config.json';write_json(p,dict(schema_version=1,experiments=[self.config.to_dict()]*2))
        with self.assertRaises(ValueError):load_configs(p)

    def test_preserves_metrics_status(self):
        r=ExperimentRunner(self.root).run(self.config)
        self.assertEqual(r['metrics'],self.payload['metrics']);self.assertEqual(r['status'],'RELIABLE')
        self.assertEqual(r['timing']['historical_seconds'],10)

    def test_standard_serialization(self):
        r=ExperimentRunner(self.root).run(self.config);p=self.root/'result.json';write_json(p,r)
        self.assertEqual(json.loads(p.read_text()),r)
        self.assertIn('report_sha256',r['provenance'])

    def test_nonfinite_sanitization(self):
        r=json_safe(dict(a=np.array([np.nan,np.inf,-np.inf,2]),b=np.int64(4)))
        self.assertEqual(r,dict(a=[None,None,None,2.],b=4))
        json.dumps(r,allow_nan=False)

    def test_invalid_reliable_import_rejected(self):
        doc=json.loads(self.path.read_text());doc['experiments'][0]['metrics']['rmse']=None
        self.path.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):ExperimentRunner(self.root).run(self.config)

    def test_insufficient_preserved(self):
        doc=json.loads(self.path.read_text());doc['experiments'][0]['metrics']['reliability']['status']='INSUFFICIENT_SUPPORT'
        doc['experiments'][0]['metrics']['verified_inliers']=3
        self.path.write_text(json.dumps(doc))
        self.assertEqual(ExperimentRunner(self.root).run(self.config)['status'],'INSUFFICIENT_SUPPORT')

    def test_events(self):
        events=[];r=ExperimentRunner(self.root,events.append).run(self.config)
        self.assertEqual([e.phase for e in events],['started','importing','completed'])
        self.assertEqual([e.sequence for e in events],[0,1,2])
        self.assertEqual(events[-1].fraction,1)
        self.assertEqual(events[-1].detail['status'],r['status'])

    def test_failure_event(self):
        events=[]
        with self.assertRaises(ValueError):ExperimentRunner(self.root,events.append).run(ExperimentConfig('x','import','prior.json','missing'))
        self.assertEqual(events[-1].phase,'failed')

    def test_stable_id_distinct_execution(self):
        runner=ExperimentRunner(self.root);a,b=runner.run(self.config),runner.run(self.config)
        self.assertEqual(a['experiment_id'],b['experiment_id']);self.assertNotEqual(a['execution_id'],b['execution_id'])

    def test_comparison_order_and_missing(self):
        runner=ExperimentRunner(self.root)
        a=runner.run(self.config);b=runner.run(ExperimentConfig('aaa','import','prior.json','control'))
        rows=comparison_records([a,b]);self.assertEqual(rows[0]['name'],'aaa');self.assertIsNone(rows[0]['raw_matches'])

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):write_json(self.path,{})

    def test_artifact_reference(self):
        (self.root/'cross_modal_summary.png').write_bytes(b'test artifact')
        r=ExperimentRunner(self.root).run(self.config)
        self.assertEqual(len(r['artifacts']),1);self.assertEqual(r['artifacts'][0]['origin'],'referenced M18 artifact')

    def test_import_does_not_mutate_cached_report(self):
        runner=ExperimentRunner(self.root);a=runner.run(self.config);a['metrics']['verified_inliers']=0
        self.assertEqual(runner.run(self.config)['metrics']['verified_inliers'],569)

    def test_live_delegation(self):
        config=ExperimentConfig('synthetic','live',pair=dict(source_sensor='IIRS',reference_sensor='WAC',source_gsd=100,reference_gsd=100))
        r=ExperimentRunner(self.root).run(config,np.zeros((24,24),np.uint8),np.zeros((24,24),np.uint8))
        self.assertEqual(r['status'],'INSUFFICIENT_SUPPORT');self.assertIsNone(r['timing']['historical_seconds'])
        self.assertEqual(r['events'][1]['phase'],'executing')


if __name__=='__main__':unittest.main()
