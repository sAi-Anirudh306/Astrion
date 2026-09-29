"""Focused M18 comparison; prepared TMC, small IIRS windows, saved OHRC control.

No LoFTR inference, source reprojection, global raster crop or whole-cube scan.
IIRS access explicitly reuses M17's approved validation receipt. Its full-cube
digest is NOT revalidated here; labels, size and per-run stat stability are
checked and that weaker freshness assurance is reported. Strict M17 loader
defaults and compatibility policy remain unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from time import perf_counter
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.matching.cross_modal import PairConfig,run_pair,support_metrics,ReliabilityPolicy
from src.ingestion.iirs_loader import read_iirs_metadata,EXCEPTION_STEM,EXCEPTION_MD5,EXCEPTION_BYTES


LIMITATIONS=[
    'RELIABLE denotes explicit engineering support gates, not independently verified registration or geographic accuracy.',
    'Three affine-defining points provide no redundant validation; lower fitted residuals are not absolute geolocation accuracy.',
    'Scale normalization does not solve modality differences; Phase Congruency does not guarantee modality invariance.',
    'LoFTR does not guarantee cross-sensor reliability. The positive learned control reuses saved M19 correspondences without inference.',
    'WAC is not equivalent to Chandrayaan imagery; one observation per sensor does not establish universal performance.',
    'Negative results are valid. OHRC results are saved M16 replays, not fresh feature-extraction benchmarks.',
    'IIRS raw DN and its predetermined spectral reductions are not calibrated radiance or the full hyperspectral cube.',
    'Footprint geolocation remains approximate; no camera/DEM orthorectification or external ground truth.',
    'M17 full-cube validation is trusted as an existing receipt; full raw hashes are not recomputed. Size/mtime checks cannot prove byte identity against external modifications.',
    'Mask endpoints are checked but descriptor neighborhoods can include invalid fill. No registration image is generated from an insufficient-support fit.',
]


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_raster(path):
    with rasterio.open(path) as f:
        a=f.read(1);mask=(f.read_masks(1)>0)&np.isfinite(a)
        return a,mask


def stat_record(path):
    s=path.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def m17_arrays(root,report):
    """Read only approved M17 windows after cross-checking the saved receipt.

    Not a new ingestion validation mode: this experiment consumes an already
    validated product. Fail if labels or receipt differ; never guess signedness.
    """
    p=Path(report['source_paths']['cube']);m=report['product'];e=report['metadata_conflict']
    if (p.stem!=EXCEPTION_STEM or not report['protected_sources_unchanged'] or not e['exception_applied']
            or e['evidence']['md5']!=EXCEPTION_MD5 or e['evidence']['sign_bit_set']!=0
            or e['evidence']['samples']*2!=EXCEPTION_BYTES or p.stat().st_size!=EXCEPTION_BYTES):
        raise ValueError('Missing validated M17 signedness receipt')
    for suffix in ('.hdr','.xml'):
        label=p.with_suffix(suffix);key=str(label.relative_to(root))
        if digest(label)!=report['protected_files'][key]['sha256']:raise ValueError('M17 label changed; revalidate ingestion')
    current=read_iirs_metadata(p.with_suffix('.xml'))
    for key in ('height','width','bands','xml_data_type','hdr_data_type','interleave','header_bytes','byte_order','expected_total_bytes'):
        if current[key]!=m[key]:raise ValueError('Metadata differs from validated M17')
    r0,r1=report['selected_region']['rows'];c0,c1=report['selected_region']['columns']
    selection=report['representation_selection']
    mapped=np.memmap(p,dtype='<u2',mode='r',shape=(m['bands'],m['height'],m['width']))
    try:
        single=np.array(mapped[selection['single_band'],r0:r1,c0:c1],copy=True)
        total=np.zeros(single.shape,np.float64)
        for band in selection['mean_bands']:total+=mapped[band,r0:r1,c0:c1]
        mean=(total/len(selection['mean_bands'])).astype(np.float32)
    finally:mapped._mmap.close()
    return [('single_band',single),('band_mean',mean)]


def main():
    started=perf_counter();root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/milestone_18_cross_modal'))
    args=parser.parse_args();output=(root/args.output).resolve()
    if not output.is_relative_to(root/'results') or output.exists():raise ValueError('Choose a fresh directory under results/')
    files={n:root/p for n,p in dict(m9='results/milestone_09_evaluation/evaluation.json',
        m16='results/milestone_16_ohrc/ohrc.json',m17='results/milestone_17_iirs/iirs.json').items()}
    prior={n:json.loads(p.read_text(encoding='utf-8')) for n,p in files.items()}
    small=set(files.values())
    for folder in ('milestone_08_registration','milestone_16_ohrc','milestone_17_iirs'):
        small.update(p for p in (root/'results'/folder).rglob('*') if p.is_file())
    m9,m16,m17=(prior[n] for n in ('m9','m16','m17'))
    for entry in m9['provenance']['inputs'].values():
        p=root/entry['path']
        if digest(p)!=entry['sha256']:raise ValueError(f'M9 artifact changed: {p}')
        small.add(p)
    for name in ('hdr','xml'):small.add(Path(m17['source_paths'][name]))
    ref_iirs=Path(m17['reference']['output_path']);small.add(ref_iirs)
    hashes={str(p):digest(p) for p in small}
    large=[Path(m17['source_paths']['cube']),root/m16['paths']['image'],root/'data/reference/lroc/WAC_GLOBAL_O000N0000_100M.TIF']
    large_before={str(p):stat_record(p) for p in large}
    rows=[]
    def add(name,result):
        result['name']=name;rows.append(result)
        m=result['metrics'];print(name,m['candidate_matches'],m['verified_inliers'],m['reliability']['status'],flush=True)
    tick=perf_counter()
    with np.load(root/m9['provenance']['inputs']['points']['path']) as points:
        metrics=support_metrics(points['source'],points['destination'],np.array(m9['provenance']['affine_matrix']),
            points['inlier_mask'].astype(bool),tuple(m9['provenance']['shape']),raw_matches=len(points['source']))
    if (metrics['candidate_matches']!=m9['correspondence']['candidate_count'] or
        metrics['verified_inliers']!=m9['correspondence']['inlier_count'] or
        not np.isclose(metrics['rmse'],m9['reprojection_residuals']['wac_pixels']['rmse'],atol=1e-10,rtol=0)):
        raise ValueError('Saved TMC control no longer reproduces M9')
    metrics['runtime_seconds']=perf_counter()-tick
    add('TMC-WAC / saved LoFTR',dict(config=dict(source_sensor='TMC',reference_sensor='WAC',representation='saved M19 LoFTR',source_gsd=100,reference_gsd=100),
        modality='cross-sensor optical',metrics=metrics,evidence_origin='saved M19/M8 support; residual/spatial metrics recomputed, no inference or refit',
        preprocessing=m9['provenance']['preprocessing'],feature=None,matcher='saved M19 LoFTR',verifier=m9['provenance']['ransac']))
    tmc,tm=read_raster(root/m9['provenance']['inputs']['before']['path'])
    reference,rm=read_raster(root/m9['provenance']['inputs']['reference']['path'])
    for representation in ('intensity','phase_congruency'):
        add('TMC-WAC / '+representation,run_pair(tmc,reference,PairConfig('TMC','WAC',100,100,representation),source_mask=tm,reference_mask=rm))
    for name,saved in m16['correspondence'].items():
        if saved['candidates']!=0 or saved['affine_matrix'] is not None:
            raise ValueError('OHRC replay adapter expects the recorded zero-candidate control')
        tick=perf_counter()
        metrics=support_metrics(np.empty((0,2)),np.empty((0,2)),None,None,tuple(m16['reference_crop']['shape']),
            source_keypoints=saved['source_keypoints'],reference_keypoints=saved['reference_keypoints'])
        metrics['runtime_seconds']=perf_counter()-tick
        add('OHRC-WAC / saved '+name,dict(config=dict(source_sensor='OHRC',reference_sensor='WAC',source_gsd=m16['scale']['primary']['source_gsd'],reference_gsd=m16['scale']['primary']['target_gsd'],representation=name),
            modality='cross-sensor optical',metrics=metrics,evidence_origin='M16 saved negative-control replay; no raw image reread',
            scale=m16['scale']['primary'],preprocessing=m16['preprocessing'],feature='SIFT/RootSIFT',matcher='L2 KNN/ratio 0.75',
            verifier=saved['estimator'],saved_raw_knn_neighbors=saved['raw_descriptor_candidates'],saved_runtime_seconds=saved['runtime_seconds']))
    reference,rm=read_raster(ref_iirs)
    for spectral,image in m17_arrays(root,m17):
        for representation in ('intensity','phase_congruency'):
            result=run_pair(image,reference,PairConfig('IIRS','WAC',m17['product']['native_gsd_m'],100,representation),reference_mask=rm)
            result['spectral_representation']=next(r for r in m17['representations'] if r['name']==spectral)['band_indices']
            result['original_cube_offset_xy']=[m17['selected_region']['columns'][0],m17['selected_region']['rows'][0]]
            add('IIRS '+spectral+'-WAC / '+representation,result)
    if hashes!={str(p):digest(p) for p in small}:raise RuntimeError('Protected previous artifacts changed')
    if large_before!={str(p):stat_record(p) for p in large}:raise RuntimeError('Source size/mtime changed')
    output.mkdir()
    fig,ax=plt.subplots(figsize=(15,6));ax.axis('off')
    table=ax.table(cellText=[[r['name'],r['metrics']['candidate_matches'],r['metrics']['verified_inliers'],
        r['metrics']['spatial_occupied_cells'],r['metrics']['reliability']['status']] for r in rows],
        colLabels=['Experiment (controls identified as saved)','Candidates','Inliers','Cells /16','Engineering status'],
        colWidths=[.43,.10,.08,.10,.29],loc='center',cellLoc='left')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    ax.set_title('ASTRION M18 — controlled cross-modal comparison',pad=15)
    fig.text(.03,.04,'RELIABLE = engineering gates only. Three-point fits are insufficient. No absolute geographic accuracy claim.',fontsize=10)
    fig.tight_layout();fig.savefig(output/'cross_modal_summary.png',dpi=150);plt.close(fig)
    plan=(root/'PLAN.md').read_text(encoding='utf-8').split('# MILESTONE 18',1)[1].split('# MILESTONE 19',1)[0]
    report=dict(milestone=18,status='PASS',exact_plan_section=plan,experiments=rows,
        limitations=LIMITATIONS,prior_reports={n:dict(path=str(p),sha256=hashes[str(p)]) for n,p in files.items()},
        protected_artifact_hashes=hashes,protected_artifacts_unchanged=True,raw_source_stat_checks=large_before,
        raw_source_assurance='Unchanged size/mtime during run; historical full-cube validation reused, not rehashed',
        learned_inference_executed=False,runtime_seconds=perf_counter()-started)
    (output/'cross_modal.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['M18 PASS: existing representations/features/geometry integrated with explicit failure analysis.',
        'Policy: >=12 candidates and inliers; >=6/16 reference cells; inlier ratio >=0.25; condition <=10; max residual <=3 pixels; nonsingular affine.',
        'Predetermined intensity and Phase Congruency; SIFT/RootSIFT; no parameter search or learned inference.']
    for r in rows:
        m=r['metrics'];lines.append(f"{r['name']}: {m['reliability']['status']}; {m['candidate_matches']} candidates, {m['verified_inliers']} inliers, {m['spatial_occupied_cells']}/16 cells; RMSE {m['rmse']}; {r['evidence_origin']}.")
        lines.append('  '+ '; '.join(m['reliability']['reasons']))
    lines += [f"Runtime {report['runtime_seconds']:.3f} s. Protected artifact hashes unchanged.",'Limitations:',*LIMITATIONS]
    (output/'cross_modal_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Completed in {report['runtime_seconds']:.3f} s",flush=True)


if __name__=='__main__':main()
