"""Approximate footprint map projection and one-grid correspondence re-test.

Run with python -m scripts.tycho_map_projected. Never performs final registration.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform
from scripts.real_tmc_wac import save_png
from scripts.tycho_phase_congruency import digest
from scripts.tycho_candidate_generation import geometry_diagnostics
from scripts.tycho_loftr import draw
from src.ingestion.metadata import read_metadata
from src.ingestion.tmc_loader import row_subregion_footprint
from src.geometry.georeferencing import FootprintMapping, project_to_reference, project_coordinates, lunar_geographic_crs, sample_bilinear
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.features.sift import extract_sift, prepare_sift_image, SIFTConfig
from src.features.rootsift import transform_sift
from src.features.phase_congruency import phase_congruency, PhaseCongruencyConfig
from src.features.learned import LoFTRMatcher, LearnedMatches
from src.matching.descriptor_matching import match_descriptors
from src.matching.ratio_test import filter_ratio, RatioMatch, RatioTestResult
from src.geometry.ransac import verify_affine, RANSACConfig


def mask_membership(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Conservative center check: all four surrounding map pixels must be valid."""
    p = np.asarray(points)
    valid = np.isfinite(p).all(axis=1) & (p >= 0).all(axis=1) & (p[:,0] <= mask.shape[1]-1) & (p[:,1] <= mask.shape[0]-1)
    indices = np.flatnonzero(valid)
    x,y = np.floor(p[valid]).astype(int).T
    xx,yy = np.minimum(x+1,mask.shape[1]-1),np.minimum(y+1,mask.shape[0]-1)
    valid[indices] &= mask[y,x] & mask[y,xx] & mask[yy,x] & mask[yy,xx]
    return valid


def verify_and_record(name: str, matches: LearnedMatches, shape: tuple, a: np.ndarray,
                      b: np.ndarray, output: Path) -> dict:
    """Index adapter to existing RANSAC plus same-grid displacement diagnostics."""
    tick = perf_counter()
    pairs = tuple(RatioMatch(i,i,None,None,None) for i in range(len(matches.source)))
    adapter = RatioTestResult(pairs,len(pairs),0,0,0,None,0.)
    geometry, failure = None,None
    try:
        geometry = verify_affine(adapter,matches.source,matches.destination,RANSACConfig())
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    row = {"name":name,"candidates":len(pairs),"inliers":geometry.inlier_count if geometry else 0,
        "inlier_ratio":geometry.inlier_ratio if geometry else None,
        "affine_matrix":geometry.matrix.tolist() if geometry else None,
        "residual_statistics":geometry.inlier_error_statistics if geometry else None,
        "rmse":None,"unique_source_positions":None,"unique_inlier_destination_positions":None,
        "occupied_wac_cells":0,"determinant":None,"linear_singular_values":None,"linear_condition_number":None,
        "same_grid_displacement":None,"failure_reason":failure,"warnings":[failure] if failure else [],
        "ransac_runtime_seconds":perf_counter()-tick}
    keep = geometry.inlier_mask if geometry else np.zeros(len(pairs),bool)
    if geometry:
        row.update(geometry_diagnostics(geometry,matches.destination,shape,100.))
        row['unique_source_positions'] = len(np.unique(matches.source[keep],axis=0))
        if row['unique_source_positions'] < 8:
            row['warnings'].append('fewer_than_8_unique_source_positions')
        displacement = matches.destination[keep]-matches.source[keep]
        magnitude = np.linalg.norm(displacement,axis=1)
        row['same_grid_displacement'] = {"median_dx":float(np.median(displacement[:,0])),
            "median_dy":float(np.median(displacement[:,1])),"median_magnitude":float(np.median(magnitude)),
            "maximum_magnitude":float(magnitude.max())}
    draw(output / f'{name}_candidates.png',a,b,matches)
    draw(output / f'{name}_inliers.png',a,b,LearnedMatches(matches.source[keep],matches.destination[keep],matches.confidence[keep],{}))
    np.savez(output / f'{name}_matches.npz',source=matches.source,destination=matches.destination,
             confidence=matches.confidence,inlier_mask=keep)
    return row


def main() -> None:
    """Map raw DN and a separate preprocessed representation using the same inverse."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    source = root/'data/processed/tycho/ch2_tmc_nrf_20211122T2123225722_d_img_d18_lat_-44_-42.6.npy'
    label = root/'data/raw/tycho/data/raw/20211122/ch2_tmc_nrf_20211122T2123225722_d_img_d18.xml'
    reference = root/'data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif'
    prior = reference.with_suffix('.json')
    folder = root/'data/processed/tycho/map_projected'
    output = root/'results/experiment_tycho_map_projected'
    folder.mkdir(parents=True,exist_ok=False)
    output.mkdir(parents=True,exist_ok=False)
    hashes = {str(p):digest(p) for p in (source,label,reference,prior)}
    raw = np.load(source,mmap_mode='r',allow_pickle=False)
    metadata = read_metadata(label)
    if raw.shape != (124259-115734,metadata['image_width']):
        raise ValueError('Crop dimensions disagree with specified row interval and label')
    corners = row_subregion_footprint(metadata,115734,124259)
    mapping = FootprintMapping(corners,*raw.shape)
    projection_started = perf_counter()
    projected = project_to_reference(raw,mapping,reference,folder/'tmc_on_wac_grid.tif')
    projection_runtime = perf_counter()-projection_started
    with rasterio.open(reference) as dataset:
        wac_raw = dataset.read(1)
        wac_valid = dataset.read_masks(1)>0
        crs, affine = dataset.crs,dataset.transform
        reference_shape = dataset.shape
    with rasterio.open(folder/'tmc_on_wac_grid.tif') as saved:
        assert saved.crs == crs and saved.transform == affine and saved.shape == reference_shape
        np.testing.assert_array_equal(saved.read(1),projected.data)
    points = np.array([[0,0],[1,0],[0,1],[1,1],[.5,.5],[.2,.7],[.8,.1],[.3,.25],[.9,.9]])*[raw.shape[1]-1,raw.shape[0]-1]
    geographic = mapping.forward(points)
    inverse,valid,residual = mapping.inverse(geographic)
    if not valid.all(): raise RuntimeError('Round-trip validation failed')
    projected_points = project_coordinates(geographic,crs)
    lon,lat = transform(crs,lunar_geographic_crs(crs),projected_points[:,0].tolist(),projected_points[:,1].tolist())
    inverse_projected,valid2,_ = mapping.inverse(np.column_stack((lon,lat)))
    if not valid2.all(): raise RuntimeError('Projected round-trip validation failed')
    previous = json.loads(prior.read_text())
    expected = np.array([previous['projected_corners'][k] for k in ('upper_left','upper_right','lower_left','lower_right')])
    validation = {"pixel_roundtrip_max_error":float(np.linalg.norm(points-inverse,axis=1).max()),
        "projected_roundtrip_max_pixel_error":float(np.linalg.norm(points-inverse_projected,axis=1).max()),
        "corner_projected_max_difference_m":float(np.linalg.norm(expected-projected_points[:4],axis=1).max()),
        "inverse_max_residual_degrees":projected.maximum_inverse_residual_degrees,
        "reference_crs_equal":True,"reference_transform_equal":True,"reference_dimensions_equal":True,
        "valid_values_finite":bool(np.isfinite(projected.data[projected.valid]).all()),
        "sample_pixels":points.tolist(),"sample_geographic_lon_lat":geographic.tolist(),
        "sample_projected_xy":projected_points.tolist()}
    config = PreprocessingConfig()
    # Preserve validated uint16 preprocessing behavior and fractional scientific DN.
    # Normalize the native scientific input, then sample its derived representation
    # with exactly the same inverse coordinates. Do not quantize the map TIFF.
    processed = preprocess_image(raw,config)
    processing = sample_bilinear(processed.data,projected.source_pixels,projected.valid.ravel()).reshape(reference_shape)
    wac_processing = preprocess_image(wac_raw,config).data
    common = projected.valid & wac_valid
    tmc_preview = prepare_sift_image(np.where(projected.valid,processing,0).astype(np.float32))
    wac_preview = prepare_sift_image(wac_processing)
    save_png(folder/'tmc_on_wac_grid_preview.png',tmc_preview)
    save_png(folder/'tmc_on_wac_grid_valid_mask.png',projected.valid.astype(np.uint8)*255)
    save_png(output/'wac_reference.png',wac_preview)
    save_png(output/'tmc_map_projected.png',tmc_preview)
    save_png(output/'validity_mask.png',projected.valid.astype(np.uint8)*255)
    save_png(output/'side_by_side.png',np.hstack((tmc_preview,wac_preview)))
    report = {"method":"footprint-based approximate map projection", "sha256":hashes,
        "limitation":"Not rigorous orthorectification: no camera model, DEM or control points. Corner interpolation is approximate.",
        "mapping":"Bilinear longitude/latitude using normalized pixel-center coordinates; Newton inverse; Rasterio/PROJ lunar sphere",
        "crs_wkt":crs.to_wkt(),"crs_proj":crs.to_dict(),"transform":list(affine),
        "shape":list(reference_shape),"resolution":[abs(affine.a),abs(affine.e)],"corners":corners,
        "dtype":"float32 scientific interpolated DN","nodata":"NaN plus GDAL validity mask",
        "valid_pixels":int(projected.valid.sum()),"valid_percent":float(100*projected.valid.mean()),
        "common_valid_pixels":int(common.sum()),"valid_intensity_range":[float(projected.data[projected.valid].min()),float(projected.data[projected.valid].max())],
        "validation":validation,"projection_runtime_seconds":projection_runtime,
        "preprocessing":asdict(config),"sift":asdict(SIFTConfig()),"phase_congruency":asdict(PhaseCongruencyConfig()),
        "ransac":asdict(RANSACConfig()),"ratio_threshold":.75,
        "processing_policy":"Native uint16 default preprocessing then bilinear map sampling; WAC default preprocessing. Common-invalid pixels set black only in processing copies; candidates require four valid neighboring pixels at both endpoints.",
        "warnings":["Bilinear point sampling is not area averaging and can alias when downsampling.",
            "Artificial mask boundaries may influence feature support despite endpoint validity checks.",
            "Same-grid identity displacement is diagnostic, not ground-truth registration error."],"matching":[]}
    (folder/'tmc_on_wac_grid_metadata.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    a,b = np.where(common,processing,0).astype(np.float32),np.where(common,wac_processing,0).astype(np.float32)
    display_a,display_b = prepare_sift_image(a),prepare_sift_image(b)
    for name in ('rootsift_intensity','rootsift_phase_congruency'):
        tick = perf_counter()
        ai,bi = (a,b) if name.endswith('intensity') else (phase_congruency(a).data,phase_congruency(b).data)
        fa,fb = transform_sift(extract_sift(ai)),transform_sift(extract_sift(bi))
        knn = match_descriptors(fa.descriptors,fb.descriptors)
        ratio = filter_ratio(knn,.75)
        source_points = fa.points[[m.query_index for m in ratio.accepted]]
        destination_points = fb.points[[m.train_index for m in ratio.accepted]]
        keep = mask_membership(source_points,common) & mask_membership(destination_points,common)
        matches = LearnedMatches(source_points[keep],destination_points[keep],np.zeros(keep.sum(),np.float32),{})
        row = verify_and_record(name,matches,reference_shape,display_a,display_b,output)
        row.update(tmc_keypoints=len(fa.keypoints),wac_keypoints=len(fb.keypoints),knn_groups=len(knn.neighbors),
            ratio_matches_before_mask=len(ratio.accepted),invalid_endpoint_rejections=int((~keep).sum()),runtime_seconds=perf_counter()-tick,
            saved_confidence_note="RootSIFT has no LoFTR confidence; saved zeros are placeholders only")
        report['matching'].append(row)
        print(json.dumps(row),flush=True)
    matcher = LoFTRMatcher(root/'models/loftr_outdoor_kornia.ckpt')
    raw_matches = matcher.match(a,b)
    keep = mask_membership(raw_matches.source,common) & mask_membership(raw_matches.destination,common)
    usable = LearnedMatches(raw_matches.source[keep],raw_matches.destination[keep],raw_matches.confidence[keep],raw_matches.metadata)
    report['loftr'] = {'raw_matches':len(raw_matches.source),'valid_endpoint_matches':len(usable.source),'inference':raw_matches.metadata}
    for threshold in (.2,.4,.6,.8):
        matches = usable.filter_confidence(threshold)
        row = verify_and_record(f'loftr_{threshold:.2f}',matches,reference_shape,display_a,display_b,output)
        row['confidence_threshold'] = threshold
        report['matching'].append(row)
        print(json.dumps(row),flush=True)
    if any(digest(Path(p)) != value for p,value in hashes.items()): raise RuntimeError('Input content changed')
    report['total_runtime_seconds'] = perf_counter()-started
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    main()
