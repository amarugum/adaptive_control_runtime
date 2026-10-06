from __future__ import annotations
import csv, json, os, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .action_registry import ActionRegistry
from .target_store import TargetStore
from .selector import fractions, candidate_table
from .online_orca import OnlineOrcaProcessor
from .ml_runtime import MLRuntime
from .storage import ensure_experiment_snapshot, resolve_inbox_transfer_dir

@dataclass(frozen=True)
class CandidateAction:
    action_id:str; energy_uJ:float; action_y_um:float; action_z_um:float; next_pulse:int

class AdaptiveEngine:
    def __init__(self,cfg:dict):
        self.cfg=cfg; paths=cfg['paths']; self.output_root=Path(paths['output_root']); self.output_root.mkdir(parents=True,exist_ok=True)
        self.registry=ActionRegistry(Path(paths['action_registry_csv'])); self.inbox_root=Path(paths['inbox_root'])
        self.targets=TargetStore(Path(paths['target_root']),Path(paths['target_registry_csv']))
        self.ml_repo=Path(paths['ml_repo']); self.calibration_root=Path(paths['calibration_root']); self.modes={}

    def _runtime(self,mode):
        if mode not in self.modes: self.modes[mode]=MLRuntime(self.ml_repo,mode,self.cfg['models'][mode],self.cfg.get('device','auto'))
        return self.modes[mode]

    @staticmethod
    def _write_json(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,path)

    def process(self,request:dict):
        t0=time.perf_counter(); rid=request['request_id']; mode=request['model']['mode']; k=int(request['step']['completed_control_step']); next_pulse=int(request['step']['pulse_end'])+5
        if request['control']['action_registry_sha256']!=self.registry.sha256: raise ValueError('ACTION_REGISTRY_HASH_MISMATCH')
        out_dir=self.output_root/request['experiment']['experiment_id']/request['experiment']['hole_uid']/f'k{k:02d}'; out_dir.mkdir(parents=True,exist_ok=True); self._write_json(out_dir/'request.json',request)
        target_id=request['target']['target_id']
        target,target_hash=self.targets.load(target_id)
        ensure_experiment_snapshot(
            cfg=self.cfg,
            experiment_id=request['experiment']['experiment_id'],
            target_id=target_id,
            target_file=self.targets.path_for(target_id),
            action_registry_file=self.registry.path,
            target_registry_file=self.targets.registry_csv,
            mode=mode,
            target_sha256=target_hash,
        )
        rt=self._runtime(mode)
        shock=None; fs=None; pp_ms=0.0
        if mode in {'shock_metadata','shock_action_history'}:
            t=time.perf_counter(); cal_dir=self.calibration_root/request['calibration']['session_calibration_id']; proc=OnlineOrcaProcessor(cal_dir,self.cfg['orca_online']); transfer_dir=resolve_inbox_transfer_dir(self.inbox_root,request['acquisition']['orca']['transfer_rel_dir']); fs=proc.process(transfer_dir/'ORCA',request,out_dir/'online_orca'); shock=fs.shock_sequence; np.save(out_dir/'shock_sequence.npy',shock.astype(np.float32),allow_pickle=False); pp_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); prob,z=rt.infer_state(request,shock); se_ms=(time.perf_counter()-t)*1000; prob_np=prob.cpu().numpy()[0,0]; threshold=float(self.cfg['action_selection']['candidate_threshold']); binary=prob_np>=threshold
        np.save(out_dir/'state_probability.npy',prob_np.astype(np.float16),allow_pickle=False); np.save(out_dir/'state_binary.npy',binary.astype(np.uint8),allow_pickle=False)
        state_eval=fractions(binary,target); sp=self.cfg['stop_policy']; stop=None
        if state_eval['over_removal_fraction']>=float(sp['over_removal_limit']): stop='over_removal'
        elif state_eval['overall_removal_fraction']>=float(sp['overall_removal_target']): stop='target_reached'
        elif k>=int(request['control']['max_control_steps']): stop='max_control_steps'
        base={'protocol_version':'1.0','request_id':rid,'status':'ok','source':{'experiment_id':request['experiment']['experiment_id'],'hole_uid':request['experiment']['hole_uid'],'completed_control_step':k,'next_control_step':k+1},'target':{'target_id':request['target']['target_id'],'target_sha256':target_hash},'state_evaluation':state_eval,'model':{'mode':mode,'state_estimator_checkpoint':str(rt.state_ckpt),'transition_checkpoint':str(rt.transition_ckpt),'inference_domain':'trained_range' if k<=10 else 'extrapolation'}}
        if fs is not None: base['frame_selection']={'auto_start_frame':fs.auto_start_frame_1based,'used_start_frame':fs.used_start_frame_1based,'selection_source':fs.selection_source,'confidence':fs.confidence,'contrast_score':fs.contrast_score,'low_confidence':fs.low_confidence,'low_contrast':fs.low_contrast}
        else: base['frame_selection']=None
        if stop:
            base.update({'decision':'stop','stop_reason':stop,'candidate_evaluation':None,'best_action':None,'timing_ms':{'postprocessing':pp_ms,'state_estimator':se_ms,'transition':0.0,'scoring':0.0,'total':(time.perf_counter()-t0)*1000}}); self._write_json(out_dir/'response.json',base); self._write_json(out_dir/'decision.json',base); return base
        actions=[CandidateAction(a.action_id,a.energy_uJ,a.action_y_um,a.action_z_um,next_pulse) for a in self.registry.actions]
        current=torch.from_numpy(binary.astype(np.float32))[None,None]
        t=time.perf_counter(); preds=rt.predict_candidates(current,z,actions); tr_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); rows,best,safe_count,no_safe=candidate_table(preds,target,self.registry.actions,threshold,float(self.cfg['action_selection']['l1_weight']),float(self.cfg['action_selection']['dice_weight']),float(sp['over_removal_limit'])); score_ms=(time.perf_counter()-t)*1000
        np.save(out_dir/'candidate_probabilities.npy',preds.astype(np.float16),allow_pickle=False); np.save(out_dir/'candidate_binary.npy',(preds>=threshold).astype(np.uint8),allow_pickle=False)
        fields=list(rows[0]);
        with (out_dir/'candidate_scores.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        base['decision']='continue'; base['stop_reason']=None; base['candidate_evaluation']={'safe_candidate_count':safe_count,'no_safe_action':no_safe,'selected_from_safe_set':not no_safe,'fallback_policy':'min_over_then_cost' if no_safe else None}
        base['best_action']={'action_id':best['action_id'],'energy_uJ':best['energy_uJ'],'action_y_um':best['action_y_um'],'action_z_um':best['action_z_um'],'predicted_cost':best['cost_j'],'predicted_overall_removal_fraction':best['predicted_overall_removal_fraction'],'predicted_over_removal_fraction':best['predicted_over_removal_fraction']}
        base['artifacts']={'candidate_probabilities':str(out_dir/'candidate_probabilities.npy'),'candidate_binary':str(out_dir/'candidate_binary.npy'),'candidate_scores':str(out_dir/'candidate_scores.csv'),'decision':str(out_dir/'decision.json')}
        base['timing_ms']={'postprocessing':pp_ms,'state_estimator':se_ms,'transition':tr_ms,'scoring':score_ms,'total':(time.perf_counter()-t0)*1000}; self._write_json(out_dir/'decision.json',base); self._write_json(out_dir/'response.json',base); return base
