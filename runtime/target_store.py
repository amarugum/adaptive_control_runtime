from __future__ import annotations
import csv, hashlib
from pathlib import Path
import numpy as np


class TargetStore:
    def __init__(self, root: Path, registry_csv: Path):
        self.root = Path(root)
        self.registry_csv = Path(registry_csv)
        with self.registry_csv.open("r", encoding="utf-8-sig", newline="") as f:
            self.rows = {r["target_id"]: r for r in csv.DictReader(f)}

    def path_for(self, target_id: str) -> Path:
        if target_id not in self.rows:
            raise KeyError(f"Unknown target_id {target_id}")
        return self.root / self.rows[target_id]["file_name"]

    def load(self, target_id: str):
        p = self.path_for(target_id)
        a = np.asarray(np.load(p), dtype=np.float32)
        if a.shape == (1, 256, 256):
            a = a[0]
        if a.shape != (256, 256):
            raise ValueError(f"Target must be 256x256, got {a.shape}")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Target must be binary 0/1")
        if float(a.sum()) <= 0:
            raise ValueError("Target is empty")
        return a, hashlib.sha256(p.read_bytes()).hexdigest()
