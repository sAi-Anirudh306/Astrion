"""M20 experiment records and coarse progress events; no presentation logic.

Imports preserve historical reliability decisions, parameters and evidence.
Live execution delegates to M18 without introducing matching algorithms.
Events describe actual orchestration boundaries, not invented stage progress.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Any
import uuid

import numpy as np

from src.matching.cross_modal import PairConfig, run_pair


def json_safe(value: Any) -> Any:
    """Convert NumPy/container values; nonfinite numbers become explicit null."""
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float):return value if np.isfinite(value) else None
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if value is None or isinstance(value,(str,int,bool)):return value
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def write_json(path: str | Path, value: Any) -> None:
    """Never overwrite an existing experiment artifact."""
    encoded=json.dumps(json_safe(value),indent=2,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(encoded)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mode: str
    report_path: str | None = None
    experiment_name: str | None = None
    pair: dict | None = None

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name.strip():raise ValueError('Experiment name required')
        if self.mode=='import':
            if not isinstance(self.report_path,str) or not self.report_path or not isinstance(self.experiment_name,str) or not self.experiment_name:
                raise ValueError('Import requires report_path and experiment_name')
            if self.pair is not None:raise ValueError('Imported parameters come from the prior result')
        elif self.mode=='live':
            if self.report_path is not None or self.experiment_name is not None:raise ValueError('Live experiments cannot name imported results')
            if not isinstance(self.pair,dict):raise ValueError('Live mode requires PairConfig fields')
            try:PairConfig(**self.pair)
            except TypeError as exc:raise ValueError(f'Invalid PairConfig fields: {exc}') from exc
        else:raise ValueError('Mode must be import or live')

    @classmethod
    def from_dict(cls, value: dict) -> 'ExperimentConfig':
        if not isinstance(value,dict):raise ValueError('Expected configuration object')
        try:return cls(**value)
        except TypeError as exc:raise ValueError(f'Invalid configuration fields: {exc}') from exc

    def to_dict(self) -> dict:
        return asdict(self)


def load_configs(path: str | Path) -> list[ExperimentConfig]:
    document=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document,dict) or set(document)!={'schema_version','experiments'} or document['schema_version']!=1:
        raise ValueError('Expected version-1 experiment configuration file')
    values=document['experiments']
    if not isinstance(values,list) or not values:raise ValueError('Nonempty experiments list required')
    configs=[ExperimentConfig.from_dict(v) for v in values]
    if len({c.name for c in configs})!=len(configs):raise ValueError('Duplicate experiment names')
    return sorted(configs,key=lambda c:c.name)


@dataclass(frozen=True)
class ExperimentEvent:
    experiment_id: str
    sequence: int
    phase: str
    fraction: float
    elapsed_seconds: float
    detail: dict


def comparison_records(results: list[dict]) -> list[dict]:
    """Stable flat records usable by CSV/report/UI consumers; absent metrics null."""
    rows=[]
    for r in sorted(results,key=lambda r:(r['name'],r['experiment_id'])):
        m=r['metrics']
        rows.append(dict(experiment_id=r['experiment_id'],name=r['name'],mode=r['mode'],status=r['status'],
            **{k:m.get(k) for k in ('source_keypoints','reference_keypoints','raw_matches','candidate_matches',
                'verified_inliers','inlier_ratio','rmse','median_residual','maximum_residual','spatial_occupied_cells',
                'spatial_occupancy_ratio','transform_available')},
            execution_seconds=r['timing']['execution_seconds'],historical_seconds=r['timing']['historical_seconds']))
    return json_safe(rows)


class ExperimentRunner:
    """Synchronous API with callback events and per-instance prior-report cache.

    experiment_id identifies a stable configuration; execution_id distinguishes
    repeated runs. Report digests distinguish imported evidence revisions.
    run(config, source, reference, masks...) invokes M18 for live data. Import
    mode reads small JSON only. Cache represents one immutable report snapshot
    per runner, identified by SHA256. Callbacks receive actual start, executing/
    importing, completed or failed boundaries; live M18 has no internal hooks.
    Callback exceptions propagate rather than silently losing caller failures.
    """
    def __init__(self, root: str | Path, on_event: Callable[[ExperimentEvent],None] | None = None):
        self.root=Path(root).resolve()
        self.on_event=on_event
        self._reports={}

    def _report(self, path):
        path=(self.root/path).resolve()
        if path not in self._reports:
            raw=path.read_bytes()
            value=json.loads(raw)
            if not isinstance(value.get('experiments'),list):raise ValueError('Expected M18 experiments report')
            self._reports[path]=(value,hashlib.sha256(raw).hexdigest())
        value,digest=self._reports[path]
        return path,value,digest

    def run(self, config: ExperimentConfig, source=None, reference=None, *, source_mask=None, reference_mask=None) -> dict:
        if not isinstance(config,ExperimentConfig):raise ValueError('Expected ExperimentConfig')
        # Validate again in case a caller mutated the nested pair dictionary.
        config=ExperimentConfig.from_dict(config.to_dict())
        started=perf_counter();events=[]
        canonical=json.dumps(config.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)
        experiment_id='exp-'+hashlib.sha256(canonical.encode()).hexdigest()[:16]
        def emit(phase,fraction,**detail):
            event=ExperimentEvent(experiment_id,len(events),phase,fraction,perf_counter()-started,json_safe(detail))
            events.append(asdict(event))
            if self.on_event:self.on_event(event)
        emit('started',0.)
        try:
            if config.mode=='import':
                if source is not None or reference is not None or source_mask is not None or reference_mask is not None:
                    raise ValueError('Import mode cannot accept live inputs')
                emit('importing',.1)
                path,prior,digest=self._report(config.report_path)
                matches=[r for r in prior['experiments'] if r.get('name')==config.experiment_name]
                if len(matches)!=1:raise ValueError('Prior experiment must resolve uniquely')
                payload=matches[0]
                provenance=dict(report_path=str(path),report_sha256=digest,experiment_name=config.experiment_name,
                    prior_reports=prior.get('prior_reports'),evidence_origin=payload.get('evidence_origin'))
                artifacts=[]
                figure=path.parent/'cross_modal_summary.png'
                if figure.is_file():
                    artifacts.append(dict(kind='comparison_visualization',path=str(figure),origin='referenced M18 artifact',
                                          sha256=hashlib.sha256(figure.read_bytes()).hexdigest()))
                limitations=prior.get('limitations',[])
            else:
                if source is None or reference is None:raise ValueError('Live source and reference are required')
                emit('executing',.1,stage='M18 orchestration; no internal progress hooks')
                payload=run_pair(source,reference,PairConfig(**config.pair),source_mask=source_mask,reference_mask=reference_mask)
                provenance=dict(evidence_origin='new M18 execution',inputs=payload.get('input_provenance'))
                artifacts=[]
                limitations=['Reliability is an engineering diagnostic, not absolute geolocation accuracy.',
                             'Three-point affine fits are not independent validation; insufficient support is not registration success.']
            metrics=payload['metrics'];reliability=metrics['reliability']
            if reliability['status'] not in {'RELIABLE','INSUFFICIENT_SUPPORT','NO_MODEL','FAILED'}:raise ValueError('Unknown reliability status')
            result=dict(schema_version=1,experiment_id=experiment_id,execution_id=str(uuid.uuid4()),name=config.name,mode=config.mode,
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),configuration=config.to_dict(),
                status=reliability['status'],reliability=reliability,metrics=metrics,
                parameters={k:payload.get(k) for k in ('config','preprocessing','feature','feature_config','matcher','verifier')},
                metadata={k:v for k,v in payload.items() if k not in {'metrics','config','preprocessing','feature','feature_config','matcher','verifier'}},
                provenance=provenance,artifacts=artifacts,limitations=limitations,
                timing=dict(execution_seconds=perf_counter()-started,
                    historical_seconds=metrics.get('runtime_seconds') if config.mode=='import' else None),events=events)
            safe=json_safe(result)
            # A malformed imported metric cannot retain RELIABLE after sanitizing.
            if result['status']=='RELIABLE' and any(safe['metrics'].get(k) is None for k in ('rmse','median_residual','maximum_residual')):
                raise ValueError('RELIABLE result contains unavailable/nonfinite required metrics')
            emit('completed',1.,status=result['status'],metrics=metrics,artifacts=artifacts)
            safe['events']=json_safe(events)
            return safe
        except Exception as exc:
            emit('failed',1.,error_type=type(exc).__name__,message=str(exc))
            raise
