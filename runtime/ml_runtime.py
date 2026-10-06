from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class MLRuntime:
    def __init__(self, ml_repo:Path, mode:str, model_cfg:dict, device_name='auto'):
        self.ml_repo=Path(ml_repo).resolve(); sys.path.insert(0,str(self.ml_repo)) if str(self.ml_repo) not in sys.path else None
        from state_estimator.models import StateEstimatorG
        from transition.models import TransitionUNet
        self.StateEstimatorG=StateEstimatorG; self.TransitionUNet=TransitionUNet
        self.mode=mode; self.device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else ('cpu' if device_name=='auto' else device_name))
        self.se_run=Path(model_cfg['state_estimator_run']); self.tr_run=Path(model_cfg['transition_run'])
        self.state_model,self.state_cfg,self.state_ckpt=self._load_state()
        self.transition_model,self.transition_cfg,self.transition_ckpt=self._load_transition()
        expected={'metadata_only':'metadata_only','action_history_only':'action_history_only','shock_metadata':'shock_metadata','shock_action_history':'shock_action_history'}[mode]
        if self.state_cfg['model']['input_mode']!=expected: raise ValueError(f'State checkpoint input_mode mismatch: expected {expected}, got {self.state_cfg["model"]["input_mode"]}')
        variant=self.transition_cfg['model']['variant'].upper(); expected_variant='B' if mode.startswith('shock_') else 'A'
        if variant!=expected_variant: raise ValueError(f'Transition variant mismatch: expected {expected_variant}, got {variant}')

    def _load_state(self):
        p=self.se_run/'checkpoints'/'best.pt'; payload=torch.load(p,map_location='cpu',weights_only=True); cfg=payload['config']; m=cfg['model']; d=cfg['data']
        model=self.StateEstimatorG(family=m['family'],input_mode=m['input_mode'],image_hw=(int(d['image_size']),int(d['image_size'])),encoder_channels=tuple(m['encoder_channels']),meta_hidden_dims=tuple(m['meta_hidden_dims']),meta_dim=int(m['meta_dim']),history_hidden_dim=int(m.get('history_hidden_dim',128)),history_num_layers=int(m.get('history_num_layers',1)),history_dropout=float(m.get('history_dropout',0.0)),convgru_hidden_channels=int(m['convgru_hidden_channels']),convgru_kernel_size=int(m['convgru_kernel_size']),decoder_channels=tuple(m['decoder_channels']),group_norm_groups=int(m['group_norm_groups']))
        model.load_state_dict(payload['model_state_dict'],strict=True); model.to(self.device).eval(); return model,cfg,p

    def _load_transition(self):
        p=self.tr_run/'checkpoints'/'best.pt'; payload=torch.load(p,map_location='cpu',weights_only=True); cfg=payload['config']; m=cfg['model']
        shock_dim=int(self.state_cfg['model']['encoder_channels'][-1]) if m['variant'].upper()=='B' else None
        model=self.TransitionUNet(variant=m['variant'],encoder_channels=tuple(m['encoder_channels']),bottleneck_channels=int(m['bottleneck_channels']),action_hidden_dims=tuple(m['action_hidden_dims']),condition_dim=int(m['condition_dim']),shock_dim=shock_dim,group_norm_groups=int(m['group_norm_groups']))
        model.load_state_dict(payload['model_state_dict'],strict=True); model.to(self.device).eval(); return model,cfg,p

    @staticmethod
    def _norm_meta(meta,cfg):
        x=np.asarray(meta,dtype=np.float32).copy(); r=[(cfg['energy_min'],cfg['energy_max']),(cfg['y_min'],cfg['y_max']),(cfg['z_min'],cfg['z_max']),(cfg['pulse_min'],cfg['pulse_max'])]
        if cfg.get('enabled',True):
            for i,(lo,hi) in enumerate(r): x[i]=2*(x[i]-lo)/(hi-lo)-1
        return x

    @staticmethod
    def _norm_history(history,cfg,block_size):
        x=np.asarray(history,dtype=np.float32).copy(); valid=np.any(x!=0,axis=1)
        if cfg.get('enabled',True):
            for i,(lo,hi) in enumerate([(cfg['energy_min'],cfg['energy_max']),(cfg['y_min'],cfg['y_max']),(cfg['z_min'],cfg['z_max'])]): x[valid,i]=2*(x[valid,i]-lo)/(hi-lo)-1
            if np.any(valid): x[valid,3]=2*(x[valid,3]-1)/(block_size-1)-1 if block_size>1 else 0
        return x

    def infer_state(self, request:dict, shock_sequence:np.ndarray|None):
        d=self.state_cfg['data']; norm=d['metadata_normalization']; hist=request['action_history']; last=request['executed_action']; pulse_end=request['step']['pulse_end']
        metadata=self._norm_meta([last['energy_uJ'],last['action_y_um'],last['action_z_um'],pulse_end],norm)
        hraw=np.asarray([[x['energy_uJ'],x['action_y_um'],x['action_z_um'],x.get('pulse_count',5)] for x in hist],dtype=np.float32)
        hnorm=self._norm_history(hraw,norm,int(d['block_size']))
        meta_t=torch.from_numpy(metadata).unsqueeze(0).to(self.device); h_t=torch.from_numpy(hnorm).unsqueeze(0).to(self.device); hl=torch.tensor([len(hist)],dtype=torch.long,device=self.device)
        shock_t=None
        if self.state_model.uses_shock:
            if shock_sequence is None: raise ValueError('Shock mode requires shock_sequence')
            x=torch.from_numpy(np.asarray(shock_sequence,dtype=np.float32))
            clip0=float(d['shock_normalization']['clip_min']); clip1=float(d['shock_normalization']['clip_max']); x=x.clamp(clip0,clip1)
            if d['shock_normalization']['to_unit_interval']: x=(x-clip0)/(clip1-clip0)
            x=F.interpolate(x,size=(int(d['image_size']),int(d['image_size'])),mode='area')
            shock_t=x.unsqueeze(0).to(self.device)
        with torch.no_grad(): out=self.state_model(shock_t,meta_t,h_t,hl,return_latents=True); prob=torch.sigmoid(out['logits']).float()
        return prob, out.get('z_shock')

    def predict_candidates(self,current_binary:torch.Tensor,z_shock,actions):
        norm=self.transition_cfg['data']['metadata_normalization']; next_pulse=None
        # candidate pulse number is supplied by caller in each action tuple's fifth field
        metas=np.asarray([[a.energy_uJ,a.action_y_um,a.action_z_um,a.next_pulse] for a in actions],dtype=np.float32)
        metas=np.stack([self._norm_meta(x,norm) for x in metas]); act=torch.from_numpy(metas).to(self.device)
        current=current_binary.to(self.device); enc=self.transition_model.encode_state(current)
        probs=[]; chunk=9
        with torch.no_grad():
            for start in range(0,len(actions),chunk):
                n=min(chunk,len(actions)-start); a=act[start:start+n]; z=None
                if self.transition_model.variant=='B': z=z_shock.expand(n,-1)
                out=self.transition_model.decode_from_encoded(current,enc,a,z_shock=z,repeats=n); probs.append(out['next_pred'].float().cpu())
        return torch.cat(probs,dim=0).numpy()[:,0]
