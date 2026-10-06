from __future__ import annotations
import csv, hashlib
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class Action: action_id:str; energy_uJ:float; action_y_um:float; action_z_um:float
class ActionRegistry:
    def __init__(self,path:Path):
        self.path=Path(path)
        with self.path.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        self.actions=[Action(r['action_id'],float(r['energy_uJ']),float(r['action_y_um']),float(r['action_z_um'])) for r in rows]
        if len(self.actions)!=27: raise ValueError(f'Expected 27 actions, got {len(self.actions)}')
    @property
    def sha256(self): return hashlib.sha256(self.path.read_bytes()).hexdigest()
