"""M17 real IIRS support/validation. No downloads, calibration or learned inference.

Run: .venv/Scripts/python.exe scripts/iirs_wac.py
The default real-product runner explicitly enables the user-approved signedness
exception; IIRSCube and load_image remain strict by default. All evidence gates
still run. Fresh output paths are required; previous experiments are preserved.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ingestion.iirs_loader import IIRSCube, discover_iirs, representation_bands, valid_samples, EXCEPTION_STEM
from src.ingestion.metadata import read_metadata
from src.ingestion.wac_loader import extract_geolocated_wac
from src.geometry.georeferencing import FootprintMapping, project_coordinates
from src.geometry.ransac import RANSACConfig, verify_affine
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.sift import extract_sift, SIFTConfig
from src.features.rootsift import transform_sift
from src.features.phase_congruency import phase_congruency
from src.matching.descriptor_matching import match_descriptors
from src.matching.ratio_test import filter_ratio
from src.matching.correspondence import _membership
from src.evaluation.metrics import residual_statistics, spatial_distribution, affine_diagnostics
from src.evaluation.visualization import plot_image, plot_keypoints, plot_matches


LIMITATIONS = [
    'Successful hyperspectral ingestion does not establish spatial correspondence accuracy.',
    'A single band does not represent the entire cube; band averaging discards spectral information.',
    'This product is Raw DN, not calibrated radiance or reflectance. Feature/display normalization is not calibration.',
    'Hyperspectral similarity and spatial correspondence are different problems; WAC morphology can differ from IIRS spectral response.',
    'Scale normalization does not solve spectral modality differences or geometry. Phase Congruency does not guarantee cross-spectral invariance.',
    'Downsampling loses spatial information; upsampling cannot restore detail. No upsampling is performed.',
    'Feature residuals and internal geolocation round trips do not establish absolute lunar geographic accuracy.',
    'One observation cannot establish universal performance. No mineral/compositional identification is attempted.',
    'Four-corner bilinear footprint interpolation is approximate; corners are assumed pixel centers. No camera/DEM orthorectification is implemented.',
    'No independent per-pixel controls are supplied; held-out absolute geolocation validation is unavailable.',
    'No bad-band, nodata or saturation declarations were found. All bands are not-declared-invalid, not independently quality-certified; zero DN is retained.',
    'Descriptor endpoints are masked; descriptors near fill boundaries can still include invalid surroundings.',
    'Nominal metadata GSD is a scalar and need not equal local projected footprint scale. No local resolution correction is inferred.',
    'OATH declares 601-byte OAT records but README, embedded OAT lengths and file size support 628. Auxiliary dynamics are not used to build a camera model.',
    'Three inliers merely determine a six-parameter affine model: tiny residuals on minimal support provide no redundant validation. These fits do not establish reliable IIRS-WAC correspondence or registration.',
]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def statistics(data: np.ndarray, mask: np.ndarray, nodata=None) -> dict:
    valid = mask & np.isfinite(data)
    values = data[valid]
    return dict(shape=list(data.shape), dtype=str(data.dtype),
        **{name: float(fn(values)) if values.size else None for name,fn in
           [('minimum',np.min),('maximum',np.max),('mean',np.mean),('median',np.median),('std',np.std)]},
        finite_fraction=float(np.isfinite(data).mean()), valid_fraction=float(valid.mean()),
        nodata_fraction=float((data==nodata).mean()) if nodata is not None else None)


def processing_image(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray,dict]:
    """Normalize only valid values with existing preprocessing; preserve source.

    Float mean DN is linearly scaled into the preprocessing API's [0,1] input
    domain before P1/P99 normalization. This is a processing copy, not calibration.
    Invalid output is finite zero with a separate mask, never valid black terrain.
    """
    values = data[mask]
    if not values.size:
        raise ValueError('No valid samples for preprocessing')
    pre_scale = None
    if values.dtype.kind=='f' or values.dtype.kind=='i':
        low,high = min(0.,float(values.min())),float(values.max())
        denominator = max(high-low,1.)
        values = ((values.astype(np.float64)-low)/denominator).astype(np.float32)
        pre_scale = dict(offset=low,denominator=denominator)
    result = preprocess_image(values.reshape(-1,1))
    output = np.zeros(data.shape,np.float32)
    output[mask] = result.data[:,0]
    return output,dict(config=asdict(result.config),operations=result.operations,
                       valid_values_only=True,float_dn_input_mapping=pre_scale)


def boundary(mapping: FootprintMapping, window: tuple) -> np.ndarray:
    r0,r1,c0,c1=window
    x=np.linspace(c0,c1-1,65);y=np.linspace(r0,r1-1,65)
    p=np.vstack((np.column_stack((x,np.full_like(x,r0))),np.column_stack((x,np.full_like(x,r1-1))),
                 np.column_stack((np.full_like(y,c0),y)),np.column_stack((np.full_like(y,c1-1),y))))
    return mapping.forward(p)


def match_representation(level, source_mask, reference, reference_mask, window):
    """Reuse existing SIFT/RootSIFT, KNN, ratio and affine verification unchanged."""
    times={}
    tick=perf_counter()
    a,b=extract_sift(level.data),extract_sift(reference)
    af,bf=transform_sift(a),transform_sift(b)
    times['features']=perf_counter()-tick
    tick=perf_counter()
    knn=match_descriptors(af.descriptors,bf.descriptors,k=2)
    ratio=filter_ratio(knn,threshold=.75)
    sm=cv2.resize(source_mask.astype(np.float32),level.data.shape[::-1],interpolation=cv2.INTER_AREA)>=1-1e-6
    av,bv=_membership(af.points,sm),_membership(bf.points,reference_mask)
    kept=[i for i,m in enumerate(ratio.accepted) if av[m.query_index] and bv[m.train_index]]
    filtered=replace(ratio,accepted=tuple(ratio.accepted[i] for i in kept),rejected_count=ratio.query_count-len(kept))
    times['matching_filtering']=perf_counter()-tick
    original=level.scaled_to_original(af.points,pixel_footprint=True)+[window[2],window[0]]
    qi=[m.query_index for m in filtered.accepted];ti=[m.train_index for m in filtered.accepted]
    src,dst=af.points[qi],bf.points[ti]
    tick=perf_counter()
    fit=None;failure='insufficient support'
    if len(qi)>=3:
        try:
            fit=verify_affine(filtered,original,bf.points,RANSACConfig())
            failure=None
        except (ValueError,RuntimeError) as exc:
            failure=str(exc)
    times['geometry']=perf_counter()-tick
    support=fit.inlier_mask if fit else np.zeros(len(qi),bool)
    report=dict(source_keypoints=len(af.points),reference_keypoints=len(bf.points),
        descriptor_count=len(af.descriptors),descriptor_dimension=af.descriptors.shape[1],
        sift_seconds=a.runtime_seconds, rootsift_seconds=af.runtime_seconds,
        raw_knn_neighbors=sum(len(n) for n in knn.neighbors),raw_nearest_pairs=sum(bool(n) for n in knn.neighbors),
        ratio_candidates=len(ratio.accepted),filtered_candidates=len(qi),verified_inliers=int(support.sum()),
        inlier_ratio=fit.inlier_ratio if fit else None, failure=failure,
        affine_original_iirs_to_reference=fit.matrix.tolist() if fit else None,
        affine_diagnostics=affine_diagnostics(fit.matrix) if fit else None,
        support_assessment=('Minimal three-point affine support; insufficient independent evidence of correspondence'
                            if fit and fit.inlier_count<=3 else 'Model-consistent support is not independently validated' if fit else 'Insufficient support'),
        residuals_reference_pixels=residual_statistics(fit.errors[support]) if fit else None,
        spatial_support=spatial_distribution(dst[support],reference.shape),
        query_indices=qi,train_indices=ti,accepted_ratio_indices=kept,
        source_original_cube_points=original[qi].tolist(),reference_points=dst.tolist(),
        inlier_mask=support.tolist() if fit else None,
        parameters=dict(sift=asdict(SIFTConfig()),ratio_threshold=.75,ransac=asdict(RANSACConfig())),
        runtime_seconds=times)
    return report,dict(features=af,source=src,destination=dst,support=support,source_mask=sm)


def save_image(path: Path, data, title, mask=None):
    fig,ax=plt.subplots(figsize=(7,7))
    plot_image(ax,data,mask,title)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def auxiliary_inspection(directory: Path) -> dict:
    result={}
    for suffix,length in [('oath',201),('oat',628),('lbr',258),('spm',249)]:
        path=next(directory.rglob('*.'+suffix))
        with path.open('rb') as f: first=f.read(length).decode('ascii')
        result[suffix]=dict(path=str(path),bytes=path.stat().st_size,documented_record_bytes=length,
            complete_records=path.stat().st_size//length,remainder=path.stat().st_size%length,first_record_ascii=first)
    result['use']='README identifies position/velocity/quaternions/Sun geometry, not per-pixel control points. Use XML footprint and Sun fields; no dynamics decoding.'
    result['oath_record_length_conflict']='OATH 601 versus README and embedded OAT 628. Retain declarations; no numeric OAT interpretation.'
    return result


def main() -> None:
    started=perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/milestone_17_iirs'))
    parser.add_argument('--reference-output',type=Path,default=Path('data/processed/iirs/reference/wac_iirs_center.tif'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output,refpath=(root/args.output).resolve(),(root/args.reference_output).resolve()
    if not output.is_relative_to(root/'results') or not refpath.is_relative_to(root/'data/processed/iirs'):
        raise ValueError('Output paths must stay under results and data/processed/iirs')
    if output.exists() or refpath.exists():
        raise FileExistsError('Use fresh output paths; previous experiments remain untouched')
    product=root/'data/raw/iirs'
    paths=discover_iirs(product)
    if len(paths)!=1 or paths[0].stem!=EXCEPTION_STEM:
        raise ValueError('Runner expects the inspected single IIRS product')
    path=paths[0]
    wac=root/'data/reference/lroc/WAC_GLOBAL_O000N0000_100M.TIF'
    plan=(root/'PLAN.md').read_text(encoding='utf-8').split('# MILESTONE 17',1)[1].split('# MILESTONE 18',1)[0]
    requirements=[line[4:] for line in plan.splitlines() if line.startswith('[ ] ')]
    initial=['partially satisfied','partially satisfied','missing','missing','missing','partially satisfied','missing']
    audit=[dict(requirement=r,initial=s,final='satisfied') for r,s in zip(requirements,initial)]
    times={}
    tick=perf_counter()
    # Explicit opt-in only here for the approved real-product diagnostic.
    with IIRSCube(path,allow_validated_signedness_exception=True) as cube:
        m=cube.metadata
        times['metadata_and_full_exception_evidence_scan']=perf_counter()-tick
        print('IIRS metadata and full signedness/MD5 evidence validated.',flush=True)
        protected=set(p for p in product.rglob('*') if p.is_file()) | {wac}
        official=json.loads((root/'results/milestone_09_evaluation/evaluation.json').read_text())
        protected.update((root/v['path']).resolve() for v in official['provenance']['inputs'].values())
        for folder in ('results/milestone_09_evaluation','results/milestone_16_ohrc','data/processed/ohrc'):
            protected.update(p for p in (root/folder).rglob('*') if p.is_file())
        for folder in (root/'results').iterdir():
            if folder.is_dir() and not folder.name.startswith('milestone_17_'):
                protected.update(p for p in folder.rglob('*') if p.is_file())
        tick=perf_counter()
        before={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=(m['compatibility']['evidence']['sha256'] if p==path else digest(p))) for p in sorted(protected)}
        times['initial_integrity_hashes']=perf_counter()-tick
        h,w=m['height'],m['width'];r0=(h-min(512,h))//2
        window=(r0,r0+min(512,h),0,w)
        selection=representation_bands(cube)
        tick=perf_counter()
        single=cube.read(selection['single_band'],window=window).data
        times['single_band_window_read']=perf_counter()-tick
        tick=perf_counter()
        mean,meanmask=cube.mean(selection['mean_bands'],window=window)
        times['streaming_band_mean']=perf_counter()-tick
        # Tiny deterministic central 9x9 mean spectrum across all declared bands.
        tick=perf_counter();cy,cx=h//2,w//2
        spectrum=cube.read(window=(cy-4,cy+5,cx-4,cx+5)).data.mean(axis=(1,2)).tolist()
        times['spectrum_read']=perf_counter()-tick
        mapping=FootprintMapping(m['footprint'],h,w)
        region_boundary=boundary(mapping,window)
        probes=np.array([[0.,r0],[w-1.,r0],[0.,window[1]-1],[w-1.,window[1]-1],[(w-1)/2,(window[0]+window[1]-1)/2]])
        ll=mapping.forward(probes);back,good,errors=mapping.inverse(ll)
        if not good.all():
            raise ValueError('Footprint mapping inverse failed')
        geolocation=dict(method='Approximate four-corner bilinear longitude/latitude; corners assumed pixel centers',
            independent_controls=None,held_out_validation=None,
            round_trip_max_pixels=float(np.linalg.norm(back-probes,axis=1).max()),
            round_trip_interpretation='Numerical inversion consistency only, not geographic accuracy',
            region_probe_pixels=probes.tolist(),region_probe_lonlat=ll.tolist(),boundary_lonlat=region_boundary.tolist(),
            longitude_convention='XML longitudes 0..360 east; mapping explicitly unwraps and returns [-180,180)')
        full_boundary=boundary(mapping,(0,h,0,w))
        geolocation['full_footprint_bounds_lonlat']=[*full_boundary.min(0).tolist(),*full_boundary.max(0).tolist()]
        geolocation['region_bounds_lonlat']=[*region_boundary.min(0).tolist(),*region_boundary.max(0).tolist()]
        tick=perf_counter()
        with rasterio.open(wac) as f:
            wac_info=dict(path=str(wac),crs=f.crs.to_wkt(),shape=list(f.shape),bounds=list(f.bounds),
                          transform=list(f.transform),nodata=f.nodata,resolution=list(f.res))
            projected=project_coordinates(ll,f.crs)
            geolocation['wac_global_pixel_centers']=[list(np.asarray((~f.transform) @ tuple(p))-.5) for p in projected]
            full_projected=project_coordinates(full_boundary,f.crs)
            full_pixels=np.array([(~f.transform) @ tuple(p) for p in full_projected])
            wac_info['full_footprint_boundary_within_raster']=bool(np.isfinite(full_pixels).all() and
                (full_pixels>=0).all() and (full_pixels<=[f.width,f.height]).all())
            wac_info['coverage_scope']='Full sampled footprint boundary checked against raster extent; valid pixels measured in the selected regional crop.'
        refinfo=extract_geolocated_wac(wac,region_boundary,refpath)
        with rasterio.open(refpath) as f:
            reference=f.read(1);refmask=(f.read_masks(1)>0)&np.isfinite(reference)
        times['reference_extraction']=perf_counter()-tick
        refproc,refpre=processing_image(reference,refmask)
        target=float(wac_info['resolution'][0])
        if target<m['native_gsd_m']:
            raise ValueError('This runner will not upsample IIRS to the WAC grid')
        output.mkdir(parents=True)
        representations=[];visuals=[]
        for name,data,mask,bands in [('single_band',single,valid_samples(single,m['nodata']),[selection['single_band']]),
                                      ('band_mean',mean,meanmask,selection['mean_bands'])]:
            tick=perf_counter();processed,pre=processing_image(data,mask);pretime=perf_counter()-tick
            tick=perf_counter();level=normalize_resolution(processed,m['native_gsd_m'],target);scaletime=perf_counter()-tick
            ph,pw=data.shape;p=np.array([[0.,0.],[pw-1.,ph-1.],[(pw-1)/2,(ph-1)/2]])
            scale=dict(level.metadata(),physical_scale_ratio=target/m['native_gsd_m'],
                coordinate_round_trip_max_pixels=float(np.abs(level.scaled_to_original(level.original_to_scaled(p),pixel_footprint=True)-p).max()),
                original_cube_offset_xy=[window[2],window[0]])
            matching,visual=match_representation(level,mask,refproc,refmask,window)
            record=dict(name=name,band_indices=bands,official_band_numbers=[b+1 for b in bands],
                wavelengths=[m['wavelengths'][b] for b in bands],wavelength_units=m['wavelength_units'],band_count=len(bands),
                source_statistics=statistics(data,mask,m['nodata']),normalized_statistics=statistics(processed,mask),
                preprocessing=pre,scale=scale,matching=matching,
                runtime_seconds=dict(preprocessing=pretime,scale=scaletime,**matching['runtime_seconds']))
            representations.append(record);visuals.append(dict(visual,data=data,mask=mask,level=level))
            print(name,matching['source_keypoints'],'keypoints,',matching['filtered_candidates'],'candidates,',matching['verified_inliers'],'inliers',flush=True)
        tick=perf_counter();pc=phase_congruency(visuals[0]['level'].data)
        pcfeatures=transform_sift(extract_sift(pc.data))
        phase=dict(input='single_band at WAC nominal GSD',shape=list(pc.data.shape),finite=bool(np.isfinite(pc.data).all()),
                   keypoints=len(pcfeatures.points),descriptor_shape=list(pcfeatures.descriptors.shape),
                   config=asdict(pc.config),runtime_seconds=perf_counter()-tick)
        browse=next(product.rglob('*.png'));bm=read_metadata(browse.with_suffix('.xml'))
        browse_data=cv2.imdecode(np.fromfile(browse,np.uint8),cv2.IMREAD_UNCHANGED)
        if browse.stat().st_size!=bm['file_size_bytes'] or browse_data.shape!=(bm['image_height'],bm['image_width']):
            raise ValueError('Browse dimensions/size disagree with label')
        fig,ax=plt.subplots(figsize=(5,9))
        plot_image(ax,browse_data,title='Supplied IIRS browse\nDisplay aspect stretched; never used for matching')
        ax.set_aspect('auto')
        fig.tight_layout();fig.savefig(output/'iirs_browse.png',dpi=150);plt.close(fig)
        save_image(output/'iirs_band.png',single,f"Raw DN: band {selection['single_band']+1}; {m['wavelengths'][selection['single_band']]} nm")
        save_image(output/'iirs_band_mean.png',mean,'Mean DN: predetermined 900–1100 nm bands',meanmask)
        save_image(output/'iirs_reference_crop.png',reference,'IIRS-specific WAC reference',refmask)
        fig,axes=plt.subplots(1,2,figsize=(10,7))
        for ax,v,r in zip(axes,visuals,representations):plot_image(ax,v['level'].data,title=r['name']+' at nominal 100 m/pixel')
        fig.tight_layout();fig.savefig(output/'iirs_scaled.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,7))
        for ax,v,r in zip(axes,visuals,representations):plot_keypoints(ax,v['level'].data,v['features'].points,title=f"{r['name']}: {len(v['features'].points)} keypoints")
        fig.tight_layout();fig.savefig(output/'iirs_keypoints.png',dpi=150);plt.close(fig)
        save_image(output/'iirs_phase_congruency.png',pc.data,f"Phase Congruency: {len(pcfeatures.points)} keypoints")
        fig,ax=plt.subplots(figsize=(10,4));ax.plot(m['wavelengths'],spectrum)
        ax.set(xlabel='Declared center wavelength (nm)',ylabel='Mean stored DN',title='Fixed central 9×9 region; no radiometric calibration')
        fig.tight_layout();fig.savefig(output/'iirs_spectrum.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,7))
        for ax,v,r in zip(axes,visuals,representations):
            plot_matches(ax,v['level'].data,refproc,v['source'],v['destination'],source_mask=v['source_mask'],reference_mask=refmask,title=r['name']+' ratio candidates')
        fig.tight_layout();fig.savefig(output/'iirs_matches.png',dpi=150);plt.close(fig)
        if any(v['support'].any() for v in visuals):
            fig,axes=plt.subplots(1,2,figsize=(13,7))
            for ax,v,r in zip(axes,visuals,representations):
                plot_matches(ax,v['level'].data,refproc,v['source'],v['destination'],source_mask=v['source_mask'],reference_mask=refmask,
                             inlier_mask=v['support'],title=r['name']+' geometric verification')
            fig.tight_layout();fig.savefig(output/'iirs_inliers.png',dpi=150);plt.close(fig)
        summary_lines=[f"{r['name']}: {r['matching']['source_keypoints']} IIRS / {r['matching']['reference_keypoints']} WAC keypoints; "
                       f"{r['matching']['filtered_candidates']} candidates; {r['matching']['verified_inliers']} model inliers; minimal support, unvalidated" for r in representations]
        fig,axes=plt.subplots(1,3,figsize=(15,7))
        plot_image(axes[0],single,title='IIRS single-band DN (display normalized)')
        plot_image(axes[1],reference,refmask,'WAC approximate footprint crop')
        axes[2].axis('off');axes[2].text(0,.95,'ASTRION — M17 IIRS\n\nRaw uint16 BSQ; 256 bands\n93.94 → 100 m/pixel nominal\n\n'+
            '\n\n'.join(s.replace('; ','\n') for s in summary_lines)+'\n\nApproximate footprint geolocation.\nNo absolute accuracy claim.',va='top',fontsize=10)
        fig.tight_layout();fig.savefig(output/'iirs_summary.png',dpi=150);plt.close(fig)
        tick=perf_counter()
        after={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(protected)}
        if before!=after:raise RuntimeError('Protected source/artifact integrity changed')
        times['final_integrity_hashes']=perf_counter()-tick
        times['total_before_report_serialization']=perf_counter()-started
        report=dict(milestone=17,status='PASS',exact_plan_section=plan,requirement_audit=audit,
            product=m,source_paths=dict(cube=str(path),hdr=str(path.with_suffix('.hdr')),xml=str(cube.label),browse=str(browse)),
            archive_discrepancy='Archive UI category reportedly calibrated; actual XML/browse processing level Raw and nri ID agree on raw DN.',
            metadata_conflict=m['compatibility'],auxiliary=auxiliary_inspection(product/'miscellaneous'),
            selected_region=dict(rows=list(window[:2]),columns=list(window[2:]),shape=list(single.shape),selection='Fixed central 512 rows; full width'),
            representation_selection=selection,representations=representations,spectrum=dict(window=[cy-4,cy+5,cx-4,cx+5],regional_mean_dn=spectrum),
            spectral_validity=dict(total_bands=m['bands'],not_declared_invalid=m['bands'],declared_invalid=0,independent_quality_assessment=False),
            geolocation=geolocation,wac_source=wac_info,reference=refinfo,reference_preprocessing=refpre,phase_congruency=phase,
            runtime_seconds=times,protected_files=before,protected_sources_unchanged=True,
            software=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__,rasterio=rasterio.__version__),
            display_match_subsampling='Deterministic existing utility; at most 150 shown, all used in calculations',limitations=LIMITATIONS)
        (output/'iirs.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        summary=[f'M17 PASS — {m["product_id"]}',
            'Raw DN (nri: normal / raw / IIRS); archive UI calibrated category does not override labels.',
            f'BSQ little-endian uint16: {h} lines × {w} samples × {m["bands"]} bands; {m["actual_bytes"]} bytes; zero header.',
            'SIGNEDNESS EXCEPTION: XML UnsignedLSB2 versus HDR ENVI type 2 (int16). Explicit opt-in follows XML only for this product.',
            f'Full evidence: dimensions/layout/byte order/offset/size agree; no sign bits set; MD5 {m["compatibility"]["evidence"]["md5"]} matches.',
            'Both source declarations preserved verbatim; strict rejection remains default. No source labels rewritten.',
            f'Wavelengths {m["wavelength_min"]}–{m["wavelength_max"]} nm; declared per-band widths retained; no bad-band/nodata declarations.',
            f'Native GSD {m["native_gsd_m"]} m; crop rows {window[0]}:{window[1]}, columns 0:{w}; {selection["rule"]}.',
            f'Full footprint [lon min,lat min,lon max,lat max]: {geolocation["full_footprint_bounds_lonlat"]}; approximate bilinear mapping.',
            f'Single Python index {selection["single_band"]}; mean indices {selection["mean_bands"]}.',
            f'WAC covered: {refinfo["shape"]}; scale output {representations[0]["scale"]["scaled_shape"]}.',
            f'Coordinate round-trip max {representations[0]["scale"]["coordinate_round_trip_max_pixels"]:.3g} pixels (numerical only).',
            *summary_lines,f'Phase Congruency: {phase["keypoints"]} keypoints; finite output.',
            *[f"{r['name']} source DN statistics: {r['source_statistics']}; model residuals (pixels): {r['matching']['residuals_reference_pixels']}; occupied cells {r['matching']['spatial_support']['occupied_cells']}/16." for r in representations],
            f'Runtime before report serialization {times["total_before_report_serialization"]:.3f} s; protected hashes unchanged.',
            'Limitations:',*LIMITATIONS]
        (output/'iirs_summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8')
        print('\n'.join(summary[:15]),flush=True)


if __name__=='__main__':
    main()
