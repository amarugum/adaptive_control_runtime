from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from runtime.config import load_runtime_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create a diagnostic all-ones 256x256 target in the active target set. "
            "This target is intended only for exercising the full CONTINUE branch of the PC2 runtime."
        )
    )
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--target-id", default="T_SMOKE_ALL")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_runtime_config(cfg_path)
    target_root = Path(cfg["paths"]["target_root"])
    registry_csv = Path(cfg["paths"]["target_registry_csv"])
    target_root.mkdir(parents=True, exist_ok=True)
    registry_csv.parent.mkdir(parents=True, exist_ok=True)

    target_id = str(args.target_id).strip()
    if not target_id:
        raise ValueError("target_id must not be empty")
    file_name = f"{target_id}.npy"
    target_path = target_root / file_name

    target = np.ones((256, 256), dtype=np.uint8)
    if target_path.exists():
        existing = np.load(target_path)
        if existing.shape == (1, 256, 256):
            existing = existing[0]
        if existing.shape != (256, 256) or not np.array_equal(existing, target):
            raise ValueError(f"Existing target conflicts with diagnostic all-ones target: {target_path}")
        target_status = "exists"
    else:
        np.save(target_path, target, allow_pickle=False)
        target_status = "created"

    fieldnames = ["target_id", "file_name", "description"]
    rows: list[dict[str, str]] = []
    if registry_csv.exists():
        with registry_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fieldnames:
                raise ValueError(
                    f"Unexpected target registry columns: {reader.fieldnames}; expected {fieldnames}"
                )
            rows = list(reader)

    matches = [r for r in rows if r.get("target_id") == target_id]
    expected = {
        "target_id": target_id,
        "file_name": file_name,
        "description": "diagnostic all-ones target for CONTINUE-branch smoke testing",
    }
    if matches:
        if len(matches) != 1:
            raise ValueError(f"Duplicate target_id in registry: {target_id}")
        if matches[0]["file_name"] != file_name:
            raise ValueError(f"Registry target_id conflict for {target_id}: {matches[0]}")
        registry_status = "exists"
    else:
        rows.append(expected)
        tmp = registry_csv.with_suffix(registry_csv.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(registry_csv)
        registry_status = "updated"

    print(f"target root    : {target_root}")
    print(f"target         : {target_path} [{target_status}]")
    print(f"target registry: {registry_csv} [{registry_status}]")
    print("IMPORTANT: restart server.py after this command so TargetStore reloads the registry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
