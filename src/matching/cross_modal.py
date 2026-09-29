"""M18 orchestration of existing methods, with diagnostic support assessment.

RELIABLE means the documented engineering gates pass, not external truth or
universal cross-sensor reliability. No registration, training or model download.
"""
from dataclasses import dataclass, asdict, replace
from time import perf_counter

import cv2
import numpy as np

from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution, ScalePyramid
from src.features.phase_congruency import phase_congruency
from src.features.sift import SIFTConfig
from src.ingestion.image_loader import ImageData
from src.geometry.ransac import verify_affine, RANSACConfig
from src.evaluation.metrics import reprojection_residuals, residual_statistics, spatial_distribution, affine_diagnostics
from .correspondence import match_scale_pyramid


@dataclass(frozen=True)
class PairConfig:
    source_sensor: str
    reference_sensor: str
    source_gsd: float
    reference_gsd: float
    representation: str = 'intensity'

    def __post_init__(self):
        if self.source_sensor not in {'TMC','OHRC','IIRS','WAC'} or self.reference_sensor not in {'TMC','OHRC','IIRS','WAC'}:
            raise ValueError('Unsupported sensor; use canonical TMC/OHRC/IIRS/WAC')
        if self.representation not in {'intensity','phase_congruency'}:
            raise ValueError('Unsupported classical representation')
        for n in (self.source_gsd,self.reference_gsd):
            if isinstance(n,bool) or not isinstance(n,(int,float)) or not np.isfinite(n) or n<=0:
                raise ValueError('GSD must be finite and positive')

    @property
    def modality(self) -> str:
        return 'cross-spectral optical/IIRS' if 'IIRS' in {self.source_sensor,self.reference_sensor} else 'cross-sensor optical'


@dataclass(frozen=True)
class ReliabilityPolicy:
    """Predetermined engineering gates, not scientific laws or calibrated confidence.

    Twelve candidates/inliers give redundancy beyond three affine-defining
    points; six of sixteen reference cells require nonlocal support. The ratio,
    conditioning and 3-pixel gates are conservative diagnostics, not guarantees.
    """
    minimum_candidates: int = 12
    minimum_inliers: int = 12
    minimum_occupied_cells: int = 6
    minimum_inlier_ratio: float = .25
    maximum_condition_number: float = 10.
    maximum_residual: float = 3.

    def __post_init__(self):
        for value in (self.minimum_candidates,self.minimum_inliers,self.minimum_occupied_cells):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('Reliability counts must be positive integers')
        if self.minimum_inliers<4 or self.minimum_candidates<self.minimum_inliers or not 2<=self.minimum_occupied_cells<=16:
            raise ValueError('Policy must require redundant, distributed affine support')
        for value in (self.minimum_inlier_ratio,self.maximum_condition_number,self.maximum_residual):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError('Invalid reliability threshold')
        if self.minimum_inlier_ratio>1 or self.maximum_condition_number<1:
            raise ValueError('Invalid ratio/condition threshold')


def assess_reliability(metrics: dict, policy: ReliabilityPolicy | None = None) -> dict:
    """Shared assessment for new runs and explicitly identified saved-result replays."""
    p=policy or ReliabilityPolicy()
    reasons=[]
    if metrics['candidate_matches']<p.minimum_candidates:reasons.append('too few candidates')
    if metrics['verified_inliers']<p.minimum_inliers:reasons.append('too few inliers; three-point fits have no redundancy')
    if not metrics['transform_available']:
        status='INSUFFICIENT_SUPPORT' if metrics['candidate_matches']<p.minimum_candidates else 'NO_MODEL'
        return dict(status=status,reasons=reasons+['no affine model'],policy=asdict(p))
    if metrics['spatial_occupied_cells']<p.minimum_occupied_cells:reasons.append('insufficient reference-grid support')
    ratio=metrics['inlier_ratio']
    if ratio is None or not np.isfinite(ratio) or ratio<p.minimum_inlier_ratio:reasons.append('low/undefined inlier ratio')
    for key in ('rmse','median_residual','maximum_residual'):
        if metrics[key] is None or not np.isfinite(metrics[key]) or metrics[key]<0:reasons.append('undefined/invalid residuals');break
    if metrics['maximum_residual'] is not None and metrics['maximum_residual']>p.maximum_residual+1e-9:
        reasons.append('residual exceeds reporting threshold')
    d=metrics.get('affine_diagnostics')
    if (not d or d['singular'] or d['condition_number'] is None or not np.isfinite(d['condition_number'])
            or d['condition_number']>p.maximum_condition_number or not np.isfinite(d['determinant']) or abs(d['determinant'])<1e-8):
        reasons.append('degenerate or poorly conditioned transform')
    return dict(status='INSUFFICIENT_SUPPORT' if reasons else 'RELIABLE',reasons=reasons,policy=asdict(p))


def support_metrics(source: np.ndarray, destination: np.ndarray, matrix: np.ndarray | None,
                    inlier_mask: np.ndarray | None, reference_shape: tuple,
                    *, source_keypoints=None, reference_keypoints=None, raw_matches=None,
                    runtime_seconds: float = 0., policy: ReliabilityPolicy | None = None) -> dict:
    """Recompute residual/spatial evidence without fitting or changing saved support.

    Masks must follow candidate order. Final inliers must satisfy the project's
    explicit 3-reference-pixel residual definition; inconsistent artifacts fail.
    """
    a,b=np.asarray(source,float),np.asarray(destination,float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite aligned Nx2 correspondences')
    if not np.isfinite(runtime_seconds) or runtime_seconds<0:raise ValueError('Invalid runtime')
    if matrix is None:
        if inlier_mask is not None and np.asarray(inlier_mask).any():raise ValueError('Inliers require a model')
        keep=np.zeros(len(a),bool);r=np.empty(0);diagnostics=None
    else:
        keep=np.asarray(inlier_mask)
        if keep.dtype!=bool or keep.shape!=(len(a),):raise ValueError('Expected aligned boolean inlier mask')
        residuals=reprojection_residuals(a,b,matrix)
        if np.any(residuals[keep]>3.+1e-9):raise ValueError('Saved inliers exceed the explicit 3-pixel threshold')
        r=residuals[keep];diagnostics=affine_diagnostics(matrix)
    stats=residual_statistics(r);spatial=spatial_distribution(b[keep],reference_shape)
    result=dict(source_keypoints=source_keypoints,reference_keypoints=reference_keypoints,raw_matches=raw_matches,
        candidate_matches=len(a),verified_inliers=int(keep.sum()),inlier_ratio=float(keep.mean()) if len(a) and matrix is not None else None,
        rmse=stats['rmse'],median_residual=stats['median'],maximum_residual=stats['maximum'],residual_statistics=stats,
        spatial_occupied_cells=spatial['occupied_cells'],spatial_occupancy_ratio=spatial['occupied_cells']/16,
        spatial_distribution=spatial,affine_diagnostics=diagnostics,transform_available=matrix is not None,
        transform=np.asarray(matrix).tolist() if matrix is not None else None,runtime_seconds=runtime_seconds)
    result['reliability']=assess_reliability(result,policy)
    return result


def prepare_representation(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray]:
    """Existing percentile preprocessing on finite valid samples only.

    Floating scientific values are shifted/scaled to the preprocessing API's
    unit interval before normalization; this is not radiometric calibration.
    Input arrays stay unchanged. Invalid fill is accompanied by an explicit mask.
    """
    image=np.asarray(image)
    if image.ndim!=2 or not image.size or image.dtype.kind not in 'uif':raise ValueError('Expected nonempty real 2D representation; explicitly reduce IIRS first')
    valid=np.isfinite(image)
    if mask is not None:
        if np.asarray(mask).dtype!=bool or np.shape(mask)!=image.shape:raise ValueError('Invalid validity mask')
        valid &= mask
    if not valid.any():raise ValueError('No finite valid samples')
    values=image[valid]
    if values.dtype.kind in 'if':
        low=min(0.,float(values.min()));denominator=max(float(values.max())-low,1.)
        values=((values.astype(float)-low)/denominator).astype(np.float32)
    data=np.zeros(image.shape,np.float32)
    data[valid]=preprocess_image(values.reshape(-1,1)).data[:,0]
    return data,valid


def run_pair(source: np.ndarray | ImageData, reference: np.ndarray | ImageData, config: PairConfig,
             *, source_mask=None, reference_mask=None) -> dict:
    """Classical 2D pair -> normalization -> common GSD -> optional phase ->
    SIFT/RootSIFT -> KNN/ratio -> affine RANSAC -> spatial/support assessment.

    Coarsen both images to max(GSDs), never upsample. Returned transform maps
    original input source pixels to original input reference pixels. Geometry
    is verified at the common reference scale, with a 3-pixel threshold there.
    Reported residuals/occupancy use that processing reference grid explicitly.
    Learned matching is reused via support_metrics on existing M19 artifacts;
    no implicit learned inference or substitute learned implementation occurs.
    ImageData sensor identity is checked; IIRS must already be reduced to 2D.
    Supply GSDs for the actual arrays, not native GSDs of earlier source products.
    """
    if not isinstance(config,PairConfig):raise ValueError('Expected PairConfig')
    provenance={}
    for name,value,sensor in [('source',source,config.source_sensor),('reference',reference,config.reference_sensor)]:
        if isinstance(value,ImageData):
            canonical={'TMC-2':'TMC','TMC2':'TMC'}.get(value.sensor.upper(),value.sensor.upper())
            if canonical!=sensor:raise ValueError('ImageData sensor conflicts with pair configuration')
            provenance[name]=dict(sensor=value.sensor,path=str(value.source_path),
                selected_bands=value.metadata.get('selected_bands'),subset=value.metadata.get('subset'))
    source=source.data if isinstance(source,ImageData) else source
    reference=reference.data if isinstance(reference,ImageData) else reference
    started=perf_counter()
    a,am=prepare_representation(source,source_mask);b,bm=prepare_representation(reference,reference_mask)
    target=max(config.source_gsd,config.reference_gsd)
    al=normalize_resolution(a,config.source_gsd,target);bl=normalize_resolution(b,config.reference_gsd,target)
    rm=cv2.resize(bm.astype(np.float32),bl.data.shape[::-1],interpolation=cv2.INTER_AREA)>=1-1e-6
    if config.representation=='phase_congruency':
        al=replace(al,data=phase_congruency(al.data).data)
        bl=replace(bl,data=phase_congruency(bl.data).data)
    record=match_scale_pyramid(ScalePyramid((al,),(target,),(0,)),bl.data,source_mask=am,reference_mask=rm)[0]
    fit=None;failure=None
    if len(record.source)>=3:
        try:fit=verify_affine(record.valid_matches,record.source_points_original,record.reference_features.points)
        except (ValueError,RuntimeError) as exc:failure=str(exc)
    metrics=support_metrics(record.source,record.destination,fit.matrix if fit else None,fit.inlier_mask if fit else None,
        bl.data.shape,source_keypoints=len(record.source_features.points),reference_keypoints=len(record.reference_features.points),
        raw_matches=sum(bool(n) for n in record.knn.neighbors),runtime_seconds=perf_counter()-started)
    original_matrix=None
    if fit is not None:original_matrix=(bl.inverse_matrix @ np.vstack((fit.matrix,[0,0,1])))[:2].tolist()
    return dict(config=asdict(config),modality=config.modality,metrics=metrics,failure=failure,input_provenance=provenance,
        transform_original_grids=original_matrix,source_scale=al.metadata(),reference_scale=bl.metadata(),
        preprocessing=asdict(PreprocessingConfig()),feature='SIFT/RootSIFT',feature_config=asdict(SIFTConfig()),matcher='L2 KNN k=2, Lowe ratio 0.75',
        verifier=asdict(RANSACConfig()),residual_grid='common-GSD reference processing pixels',
        evidence_origin='new classical execution',
        source_points_original=record.source.tolist(),reference_points_original=bl.scaled_to_original(record.destination,pixel_footprint=True).tolist())
