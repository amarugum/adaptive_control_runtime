from __future__ import annotations

import argparse
from pathlib import Path

from runtime.config import load_runtime_config
from runtime.storage import initialize_storage


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize one date/session PC2 adaptive-control storage root.")
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_runtime_config(config_path)
    report = initialize_storage(cfg, config_path.parent)

    print(f"session_root: {report['session_root']}")
    print(f"transfer_root: {report['transfer_root']}")
    print(f"inbox: {report['inbox']}")
    print(f"experiments: {report['experiments']}")
    print(f"calibration sessions: {report['calibration_sessions']}")
    print(
        f"action registry: {report['action_registry']['path']} "
        f"[{report['action_registry']['status']}]"
    )
    print(
        f"target registry: {report['target_registry']['path']} "
        f"[{report['target_registry']['status']}]"
    )
    for name, status in report["target_files"].items():
        print(f"target template: {name} [{status}]")


if __name__ == "__main__":
    main()
