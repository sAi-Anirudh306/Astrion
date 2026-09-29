"""M20 lightweight import demonstration. No scientific pipeline recomputation."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import platform
import subprocess
from time import perf_counter
import numpy as np
import cv2
from src.evaluation.experiments import ExperimentRunner,load_configs,comparison_records,write_json


def main():
    started=perf_counter();root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/milestone_20.json'))
    parser.add_argument('--output',type=Path,default=Path('results/milestone_20_experiments'))
    args=parser.parse_args();config=(root/args.config).resolve();output=(root/args.output).resolve()
    if not output.is_relative_to(root/'results') or output.exists():raise ValueError('Choose a fresh results directory')
    configs=load_configs(config)
    if any(c.mode!='import' for c in configs):raise ValueError('This demonstration imports only; use ExperimentRunner.run for live arrays')
    def progress(event):
        if event.phase=='completed':print(event.experiment_id,event.detail['status'],flush=True)
    runner=ExperimentRunner(root,progress)
    protected={(root/c.report_path).resolve() for c in configs}
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    results=[runner.run(c) for c in configs]
    if before!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}:raise RuntimeError('Imported report changed')
    records=comparison_records(results)
    plan=(root/'PLAN.md').read_text(encoding='utf-8').split('# MILESTONE 20',1)[1].split('# MILESTONE 21',1)[0]
    git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    git_status=subprocess.run(['git','status','--short'],cwd=root,capture_output=True,text=True,check=True).stdout
    report=dict(schema_version=1,milestone=20,status='PASS',exact_plan_section=plan,
        configuration_path=str(config),configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        software=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,git_head=git_head,git_status=git_status),
        experiments=results,comparison=records,imported_reports_unchanged=True,
        runtime_seconds=perf_counter()-started,
        limitations=['All demonstration results are imports, not fresh matching or registration.',
            'Reliability decisions are preserved engineering criteria; fitted residuals are not geographic accuracy.',
            'Progress events are coarse orchestration boundaries; M18 has no internal stage callbacks.',
            'Existing M18 visualization is referenced with its checksum; no unnecessary graphic is regenerated.',
            'Negative cases remain insufficient; one positive TMC case does not establish sensor invariance.'])
    output.mkdir()
    write_json(output/'experiments.json',report)
    write_json(output/'run_config.json',dict(schema_version=1,experiments=[c.to_dict() for c in configs]))
    with (output/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['M20 PASS: nine existing M18 comparisons imported without recomputation.' if len(results)==9 else f'M20: {len(results)} comparisons imported.',
        'Visualization: referenced M18 comparison figure; paths and checksums in experiments.json.',
        'Runtime columns distinguish current import time from historical experiment runtime.']
    lines += [f"{r['name']}: {r['status']}; candidates={r['candidate_matches']}, inliers={r['verified_inliers']}, cells={r['spatial_occupied_cells']}/16, RMSE={r['rmse']} px" for r in records]
    lines += [f"Framework runtime before serialization: {report['runtime_seconds']:.4f} s",*report['limitations']]
    (output/'experiments_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Imported {len(results)} experiments in {report['runtime_seconds']:.4f} s")


if __name__=='__main__':main()
