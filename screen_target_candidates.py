from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.selector import candidate_table, fractions


def resize_nearest(mask: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = mask.shape
    if (h, w) == (size, size):
        return mask
    yi = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
    xi = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
    return mask[np.ix_(yi, xi)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, bool); bb = np.asarray(b, bool)
    u = int(np.logical_or(aa, bb).sum())
    return 1.0 if u == 0 else float(np.logical_and(aa, bb).sum()) / u


def load_target(path: Path) -> np.ndarray:
    a = np.asarray(np.load(path), dtype=np.float32)
    if a.shape == (1, 256, 256):
        a = a[0]
    if a.shape != (256, 256):
        raise ValueError(f"Candidate target must be 256x256, got {a.shape}: {path}")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"Candidate target must be binary 0/1: {path}")
    if a.sum() <= 0:
        raise ValueError(f"Empty candidate target: {path}")
    return a


def decision_dirs(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("candidate_probabilities.npy"):
        d = p.parent
        if (d / "state_binary.npy").exists():
            out.append(d)
    return sorted(set(out))


def observed_support_masks(data_root: Path, channel: int, threshold: float, pulses: list[int]) -> list[tuple[str, np.ndarray]]:
    masks = []
    pulse_set = set(pulses)
    for hole in sorted(p for p in data_root.glob("h*") if p.is_dir()):
        for pdir in sorted(p for p in hole.glob("p*") if p.is_dir()):
            try:
                pulse = int(pdir.name[1:])
            except Exception:
                continue
            if pulse not in pulse_set:
                continue
            tp = pdir / "target.npy"
            if not tp.exists():
                continue
            a = np.asarray(np.load(tp), dtype=np.float32)
            if a.ndim != 3 or not 0 <= channel < a.shape[0]:
                continue
            m = resize_nearest((a[channel] >= threshold).astype(np.uint8), 256)
            masks.append((f"{hole.name}/p{pulse:04d}", m))
    return masks


def main() -> int:
    ap = argparse.ArgumentParser(description="Screen target masks against saved one-step transition predictions")
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--decision-root", type=Path, required=True, help="Root containing PC2 kXX decision artifact directories")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None, help="Optional training/observed dataset root for nearest-observed IoU")
    ap.add_argument("--support-pulses", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    ap.add_argument("--pristine-channel", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-support-iou", type=float, default=0.50)
    ap.add_argument("--max-no-safe-fraction", type=float, default=0.20)
    ap.add_argument("--min-mean-safe-actions", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_runtime_config(Path(args.config))
    registry = ActionRegistry(Path(cfg["paths"]["action_registry_csv"]))
    threshold = float(cfg["action_selection"]["candidate_threshold"])
    l1w = float(cfg["action_selection"]["l1_weight"])
    dicew = float(cfg["action_selection"]["dice_weight"])
    over_limit = float(cfg["stop_policy"]["over_removal_limit"])
    overall_target = float(cfg["stop_policy"]["overall_removal_target"])

    candidates = sorted(p for p in args.candidate_dir.resolve().glob("*.npy") if p.is_file())
    if not candidates:
        raise RuntimeError("No candidate .npy files found")
    states = decision_dirs(args.decision_root.resolve())
    if not states:
        raise RuntimeError("No directories with state_binary.npy + candidate_probabilities.npy found")

    support = []
    if args.data_root is not None:
        support = observed_support_masks(args.data_root.resolve(), args.pristine_channel, args.threshold, args.support_pulses)
        if not support:
            raise RuntimeError("--data-root supplied but no observed support masks were found")

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    per_state = []
    summary = []
    min_energy = min(a.energy_uJ for a in registry.actions); max_energy = max(a.energy_uJ for a in registry.actions)

    for cp in candidates:
        target = load_target(cp)
        target_id = cp.stem
        support_iou = np.nan; support_name = ""
        if support:
            vals = [(iou(target, m), name) for name, m in support]
            support_iou, support_name = max(vals, key=lambda x: x[0])

        rows_for_target = []
        for d in states:
            current = np.asarray(np.load(d / "state_binary.npy"), dtype=np.uint8)
            preds = np.asarray(np.load(d / "candidate_probabilities.npy"), dtype=np.float32)
            if current.shape != (256, 256) or preds.shape != (27, 256, 256):
                raise ValueError(f"Unexpected artifacts at {d}: current={current.shape}, preds={preds.shape}")
            current_fr = fractions(current, target)
            cand_rows, best, safe_count, no_safe = candidate_table(
                preds, target, registry.actions, threshold, l1w, dicew, over_limit
            )
            row = {
                "target_id": target_id,
                "decision_dir": str(d),
                "target_area_fraction": float(target.mean()),
                "current_overall_removal_fraction": current_fr["overall_removal_fraction"],
                "current_over_removal_fraction": current_fr["over_removal_fraction"],
                "already_target_reached": int(current_fr["overall_removal_fraction"] >= overall_target),
                "already_over_limit": int(current_fr["over_removal_fraction"] >= over_limit),
                "safe_candidate_count": int(safe_count),
                "no_safe_action": int(no_safe),
                "best_action_id": best["action_id"],
                "best_energy_uJ": best["energy_uJ"],
                "best_action_y_um": best["action_y_um"],
                "best_action_z_um": best["action_z_um"],
                "best_cost_j": best["cost_j"],
                "best_predicted_overall_removal_fraction": best["predicted_overall_removal_fraction"],
                "best_predicted_over_removal_fraction": best["predicted_over_removal_fraction"],
                "best_is_safe": int(best["is_safe"]),
            }
            rows_for_target.append(row); per_state.append(row)

        no_safe_fraction = float(np.mean([r["no_safe_action"] for r in rows_for_target]))
        mean_safe = float(np.mean([r["safe_candidate_count"] for r in rows_for_target]))
        mean_cost = float(np.mean([r["best_cost_j"] for r in rows_for_target]))
        mean_overall = float(np.mean([r["best_predicted_overall_removal_fraction"] for r in rows_for_target]))
        mean_over = float(np.mean([r["best_predicted_over_removal_fraction"] for r in rows_for_target]))
        already_fraction = float(np.mean([r["already_target_reached"] for r in rows_for_target]))
        extreme_fraction = float(np.mean([r["best_energy_uJ"] in {min_energy, max_energy} for r in rows_for_target]))
        unique_actions = len({r["best_action_id"] for r in rows_for_target})
        support_ok = True if not support else bool(support_iou >= args.min_support_iou)
        screen_pass = (
            support_ok
            and no_safe_fraction <= args.max_no_safe_fraction
            and mean_safe >= args.min_mean_safe_actions
            and mean_over < over_limit
        )
        summary.append({
            "target_id": target_id,
            "target_file": str(cp),
            "target_area_fraction": float(target.mean()),
            "n_decision_states": len(rows_for_target),
            "nearest_observed_iou": support_iou,
            "nearest_observed_state": support_name,
            "mean_safe_candidate_count": mean_safe,
            "min_safe_candidate_count": min(r["safe_candidate_count"] for r in rows_for_target),
            "no_safe_fraction": no_safe_fraction,
            "mean_best_cost_j": mean_cost,
            "mean_best_predicted_overall_fraction": mean_overall,
            "mean_best_predicted_over_fraction": mean_over,
            "already_target_reached_fraction": already_fraction,
            "extreme_energy_selection_fraction": extreme_fraction,
            "unique_best_actions": unique_actions,
            "screen_pass": int(screen_pass),
        })

    summary.sort(key=lambda r: (-r["screen_pass"], r["no_safe_fraction"], r["mean_best_cost_j"], -r["mean_safe_candidate_count"]))
    per_path = out / "target_screen_per_state.csv"; sum_path = out / "target_screen_summary.csv"
    with per_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_state[0])); w.writeheader(); w.writerows(per_state)
    with sum_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    print(f"Candidates         : {len(candidates)}")
    print(f"Decision states    : {len(states)}")
    print(f"Observed support   : {len(support)}")
    print(f"Summary            : {sum_path}")
    print(f"Per-state details  : {per_path}")
    print("Top candidates:")
    for r in summary[:10]:
        print(
            f"  {r['target_id']}: pass={r['screen_pass']} support_iou={r['nearest_observed_iou']} "
            f"safe_mean={r['mean_safe_candidate_count']:.1f} no_safe={r['no_safe_fraction']:.2f} "
            f"cost={r['mean_best_cost_j']:.4f} over={r['mean_best_predicted_over_fraction']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
