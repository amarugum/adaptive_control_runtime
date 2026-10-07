from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from runtime.config import load_runtime_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Register one approved 256x256 binary target in the active PC2 target set")
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--description", required=True)
    args = ap.parse_args()

    cfg = load_runtime_config(Path(args.config).resolve())
    root = Path(cfg["paths"]["target_root"]); registry = Path(cfg["paths"]["target_registry_csv"])
    root.mkdir(parents=True, exist_ok=True)
    src = args.source.resolve(); target_id = args.target_id.strip()
    if not target_id or target_id == "T_SMOKE_ALL":
        raise ValueError("Use a non-smoke target_id")
    a = np.asarray(np.load(src), dtype=np.float32)
    if a.shape == (1, 256, 256): a = a[0]
    if a.shape != (256, 256) or not np.all((a == 0) | (a == 1)) or a.sum() <= 0:
        raise ValueError(f"Target must be non-empty binary 256x256, got {a.shape}")

    dest = root / f"{target_id}.npy"
    if dest.exists():
        raise FileExistsError(f"Target already exists; refusing overwrite: {dest}")
    np.save(dest, a.astype(np.uint8), allow_pickle=False)

    fields = ["target_id", "file_name", "description"]
    rows = []
    if registry.exists():
        with registry.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames != fields:
                raise ValueError(f"Unexpected target registry columns: {r.fieldnames}")
            rows = list(r)
    if any(r["target_id"] == target_id for r in rows):
        dest.unlink(missing_ok=True)
        raise ValueError(f"target_id already exists in registry: {target_id}")
    rows.append({"target_id": target_id, "file_name": dest.name, "description": args.description})
    tmp = registry.with_suffix(registry.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(registry)
    print(f"registered target: {target_id}")
    print(f"file             : {dest}")
    print(f"registry         : {registry}")
    print("Restart server.py so TargetStore reloads the registry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
