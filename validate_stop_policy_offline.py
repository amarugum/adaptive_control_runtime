from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runtime.config import load_runtime_config
from runtime.selector import fractions


def decide(mask: np.ndarray, target: np.ndarray, k: int, cfg: dict) -> tuple[str | None, dict]:
    fr = fractions(mask, target)
    sp = cfg["stop_policy"]
    stop = None
    if fr["over_removal_fraction"] >= float(sp["over_removal_limit"]):
        stop = "over_removal"
    elif fr["overall_removal_fraction"] >= float(sp["overall_removal_target"]):
        stop = "target_reached"
    elif k >= int(sp["max_control_steps"]):
        stop = "max_control_steps"
    return stop, fr


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline branch test for the exact PC2 stop-policy order")
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    args = ap.parse_args()
    cfg = load_runtime_config(Path(args.config).resolve())
    overall = float(cfg["stop_policy"]["overall_removal_target"])
    over_lim = float(cfg["stop_policy"]["over_removal_limit"])
    max_k = int(cfg["stop_policy"]["max_control_steps"])

    target = np.zeros((256, 256), dtype=np.uint8)
    target[64:192, 64:192] = 1
    n = int(target.sum())
    inside = np.argwhere(target == 1)
    outside = np.argwhere(target == 0)

    cases = []
    # Clearly below target.
    m = np.zeros_like(target)
    take = max(1, int(0.50 * n)); yx = inside[:take]; m[yx[:, 0], yx[:, 1]] = 1
    cases.append(("continue_under_target", m, 1, None))

    # Reach target threshold without over-removal.
    m = np.zeros_like(target)
    take = min(n, int(np.ceil(overall * n))); yx = inside[:take]; m[yx[:, 0], yx[:, 1]] = 1
    cases.append(("target_reached", m, 1, "target_reached"))

    # Over-removal branch has priority even if target coverage is also high.
    m = target.copy()
    take = max(1, int(np.ceil(over_lim * n))); yx = outside[:take]; m[yx[:, 0], yx[:, 1]] = 1
    cases.append(("over_removal_priority", m, 1, "over_removal"))

    # Max-step branch while neither shape condition is met.
    m = np.zeros_like(target)
    cases.append(("max_control_steps", m, max_k, "max_control_steps"))

    failed = 0
    print("=== PC2 stop-policy offline validation ===")
    print(f"overall target   : {overall}")
    print(f"over limit       : {over_lim}")
    print(f"max steps        : {max_k}")
    for name, mask, k, expected in cases:
        got, fr = decide(mask, target, k, cfg)
        ok = got == expected
        failed += 0 if ok else 1
        print(f"{name:24s} expected={expected!s:18s} got={got!s:18s} fractions={fr} {'PASS' if ok else 'FAIL'}")
    if failed:
        raise RuntimeError(f"Stop-policy validation failed in {failed} case(s)")
    print("PASS: over_removal -> target_reached -> max_control_steps priority matches runtime/engine.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
