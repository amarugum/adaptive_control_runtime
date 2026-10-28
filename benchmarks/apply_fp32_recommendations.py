from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml

DEFAULT_MODES = [
    "metadata_only",
    "action_history_only",
    "shock_metadata",
    "shock_action_history",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Apply four-model GPU benchmark chunk recommendations to the runtime config. "
            "V1.10 intentionally forces both Estimator and Transition to FP32."
        )
    )
    ap.add_argument("--runtime-config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--recommendations", required=True, help="benchmark recommendations.json")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.runtime_config).resolve()
    rec_path = Path(args.recommendations).resolve()
    cfg = load_yaml(cfg_path)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    models = cfg.get("models") or {}

    changes = []
    for mode in args.modes:
        if mode not in models:
            raise KeyError(f"Runtime config has no models.{mode}")
        if mode not in rec:
            raise KeyError(f"Recommendations file has no entry for {mode}")
        chunk = rec[mode].get("candidate_chunk_size")
        if chunk is None:
            raise ValueError(f"No candidate_chunk_size recommendation for {mode}")
        chunk = int(chunk)
        if chunk < 1 or chunk > 27:
            raise ValueError(f"Invalid candidate_chunk_size for {mode}: {chunk}")

        old = models[mode].get("performance") or {}
        new = {
            "estimator_amp": False,
            "transition_amp": False,
            "candidate_chunk_size": chunk,
        }
        changes.append((mode, old, new))
        models[mode]["performance"] = new

    print("Four-model FP32 runtime settings to apply:")
    for mode, old, new in changes:
        print(f"  {mode}: {old} -> {new}")

    if args.dry_run:
        print("Dry run only; config was not modified.")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = cfg_path.with_name(cfg_path.name + f".bak_{stamp}")
    shutil.copy2(cfg_path, backup)
    cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Updated: {cfg_path}")
    print(f"Backup : {backup}")
    print("AMP remains disabled for all four modes by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
