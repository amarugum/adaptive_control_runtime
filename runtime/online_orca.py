from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

_SIBLING_POST = Path(__file__).resolve().parents[2] / 'laser_processing_postprocessing'
if str(_SIBLING_POST) not in sys.path:
    sys.path.insert(0, str(_SIBLING_POST))

from postprocess.calibration import GeometryCalibration
from postprocess.cropping import load_crop_specs, resolve_fine_rotations
from postprocess.frame_detection import detect_orca_start
from postprocess.manual_review import apply_manual_override
from postprocess.models import ManualFrameOverride, MotionSettings, RatioSettings, StepMotion
from postprocess.orca import build_orca_detection_sequence, load_orca_stamp_specs, process_orca_pulses_from_files

@dataclass(frozen=True)
class OnlineOrcaResult:
    shock_sequence: np.ndarray
    auto_start_frame_1based: int|None
    used_start_frame_1based: int
    selection_source: str
    confidence: float
    contrast_score: float
    low_confidence: bool
    low_contrast: bool

class OnlineOrcaProcessor:
    def __init__(self, calibration_dir:Path, cfg:dict):
        c=Path(calibration_dir); self.cfg=cfg
        self.geometry=GeometryCalibration.load(c/'grid_definition.csv',c/'orca_grid_calibration.csv',c/'has_side_grid_calibration.csv',c/'orca_processing_origin.csv')
        self.crop_specs=resolve_fine_rotations(load_crop_specs(c/'crop_config.csv'),{'ORCA':self.geometry.physical_to_orca_ref,'HAS_SIDE':self.geometry.physical_to_has_side})
        self.stamp_csv=c/'orca_stamp_time.csv'
        self.available=sorted(self.geometry.orca_rows)
        specs=load_orca_stamp_specs(self.stamp_csv,self.available)
        self.detect_stamps=[s.stamp_frame for s in sorted((x for x in specs if x.use_for_ml),key=lambda x:int(x.ml_channel))]
        self.motion_settings=MotionSettings(float(cfg.get('execution_to_image_y_sign',1.0)),float(cfg.get('execution_to_image_z_sign',-1.0)))
        self.ratio=RatioSettings()

    @staticmethod
    def _strict_override(start0:int,pulse_count:int,n_frames:int):
        events=tuple(start0+2*i for i in range(pulse_count)); refs=tuple(x-1 for x in events)
        if start0<1 or events[-1]>=n_frames: raise ValueError(f'Invalid strict-alternation start frame {start0+1} for {n_frames} frames')
        return ManualFrameOverride(events,refs)

    @staticmethod
    def _contrast(seq,start0,pulse_count):
        vals=[]
        x=np.asarray(seq,dtype=np.float32)
        for i in range(pulse_count):
            e=start0+2*i; b=e-1
            a=x[e]; q=x[b]
            lr=np.log((a+1.0)/(q+1.0)); lr=lr-np.median(lr,axis=(-2,-1),keepdims=True)
            vals.append(float(np.mean(np.abs(lr))))
        return float(np.mean(vals)) if vals else 0.0

    def process(self, raw_dir:Path, request:dict, output_dir:Path):
        step=request['step']; st=request['stage']; k=int(step['completed_control_step']); pulse_count=int(step['pulse_count'])
        dy_exec=float(st['actual_y_um'])-float(st['hole_origin_y_um']); dz_exec=float(st['actual_z_um'])-float(st['hole_origin_z_um'])
        motion=StepMotion(request['request_id'],request['experiment']['experiment_id'],request['experiment']['hole_uid'],k,float(request['executed_action']['energy_uJ']),float(request['executed_action']['action_y_um']),float(request['executed_action']['action_z_um']),int(step['pulse_start']),int(step['pulse_end']),pulse_count,dy_exec,dz_exec,dy_exec*self.motion_settings.execution_to_image_y_sign,dz_exec*self.motion_settings.execution_to_image_z_sign)
        origin=self.geometry.orca_origin_for_motion(motion.delta_z_um,motion.delta_y_um)
        seq,files,stamp_ids=build_orca_detection_sequence(Path(raw_dir),self.geometry,self.crop_specs['ORCA'],origin,expected_frames=int(self.cfg.get('expected_frames',15)),downsample=int(self.cfg.get('detection_downsample',4)),stamp_ids=self.detect_stamps)
        auto=None
        try: auto=detect_orca_start(seq,pulse_count)
        except Exception: auto=None
        auto_start=None if auto is None else int(auto.start_index)+1
        conf=0.0 if auto is None else float(auto.confidence)
        candidate_start0=(int(auto.start_index) if auto is not None else None)
        contrast=0.0 if candidate_start0 is None else self._contrast(seq,candidate_start0,pulse_count)
        low_conf=auto is None or conf < float(self.cfg.get('confidence_threshold',0.05))
        low_contrast=auto is None or contrast < float(self.cfg.get('contrast_threshold',0.0))
        previous=(request.get('frame_selection_context') or {}).get('previous_used_start_frame_1based')
        default=int(self.cfg['session_default_start_frame_1based'])
        if not (low_conf or low_contrast): used1=auto_start; source='auto'
        elif previous is not None: used1=int(previous); source='previous_step'
        else: used1=default; source='session_default'
        override=self._strict_override(int(used1)-1,pulse_count,len(files))
        out=Path(output_dir); pulse_dirs=[]; ml_dirs=[]
        for j in range(pulse_count):
            pd=out/f'pulse_{j+1:02d}'/'processed'; md=out/f'pulse_{j+1:02d}'/'ml'; pd.mkdir(parents=True,exist_ok=True); md.mkdir(parents=True,exist_ok=True); pulse_dirs.append(pd); ml_dirs.append(md)
        selection,_=process_orca_pulses_from_files(seq,files,self.geometry,self.crop_specs['ORCA'],origin,motion,pulse_dirs,ml_dirs,self.ratio,self.stamp_csv,stamp_ids,save_selected_tifs=False,all_available_stamp_ids=self.available,selection_override=override)
        shocks=np.stack([np.load(md/'shock_input.npy').astype(np.float32) for md in ml_dirs],axis=0)
        return OnlineOrcaResult(shocks,auto_start,int(used1),source,conf,contrast,low_conf,low_contrast)
