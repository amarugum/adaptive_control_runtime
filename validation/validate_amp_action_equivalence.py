from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Allow direct execution from adaptive_control_runtime/.
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.selector import candidate_table


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_target_root(cfg: Dict[str, Any], explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    session_root = Path(cfg["storage"]["session_root"])
    offline_root = session_root / "offline_evaluation"
    candidates = sorted(
        [p for p in offline_root.glob("target_reachability_*") if (p / "targets_256").is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find <session_root>/offline_evaluation/target_reachability_*/targets_256. "
            "Pass --target-root explicitly."
        )
    return candidates[-1] / "targets_256"


def load_targets(target_root: Path) -> Dict[str, np.ndarray]:
    files = sorted(target_root.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy targets in {target_root}")
    out: Dict[str, np.ndarray] = {}
    for p in files:
        a = np.asarray(np.load(p), dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.shape != (256, 256):
            raise ValueError(f"Target must be 256x256: {p}, got {a.shape}")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError(f"Target is not exact binary 0/1: {p}")
        out[p.stem] = a
    return out


def add_ml_repo(ml_repo: Path) -> None:
    p = str(ml_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def physical_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return str(row["hole_uid"]), int(row["current_pulse"]), int(row["next_pulse"])


def load_transition_metadata(run_dir: Path, ml_repo: Path, split: str = "test") -> Dict[str, Any]:
    add_ml_repo(ml_repo)
    from transition.dataset import load_dataset_split, load_manifest

    ckpt = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    d = cfg["data"]
    root = Path(d["root"])
    rows = load_manifest(root / d["manifest_name"])
    holes = sorted({str(r["hole_uid"]) for r in rows})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)
    wanted = set(splits[split])
    rows = [r for r in rows if str(r["hole_uid"]) in wanted]
    return {
        "run_dir": run_dir,
        "checkpoint": ckpt,
        "cfg": cfg,
        "rows_by_key": {physical_key(r): r for r in rows},
    }


def select_common_keys(
    metadata: Sequence[Dict[str, Any]],
    states_per_pulse: int,
    seed: int,
    pulses: Sequence[int],
) -> List[Tuple[str, int, int]]:
    common = set.intersection(*(set(x["rows_by_key"]) for x in metadata))
    rng = np.random.default_rng(seed)
    selected: List[Tuple[str, int, int]] = []
    for pulse in pulses:
        group = sorted([k for k in common if int(k[1]) == int(pulse)])
        if len(group) > states_per_pulse:
            idx = sorted(rng.choice(len(group), size=states_per_pulse, replace=False).tolist())
            group = [group[i] for i in idx]
        selected.extend(group)
    if not selected:
        raise RuntimeError("No common test states selected")
    return selected


def load_model(meta: Dict[str, Any], ml_repo: Path, device: torch.device):
    add_ml_repo(ml_repo)
    from transition.dataset import load_z_shock_npz
    from transition.models import TransitionUNet

    payload = torch.load(meta["checkpoint"], map_location="cpu", weights_only=True)
    cfg = payload["config"]
    m = cfg["model"]
    variant = str(m["variant"]).upper()
    z_lookup = None
    shock_dim = None
    if variant == "B":
        z_lookup = load_z_shock_npz(Path(cfg["data"]["z_shock_npz"]))
        shock_dim = int(next(iter(z_lookup.values())).shape[0])

    model = TransitionUNet(
        variant=variant,
        encoder_channels=tuple(m["encoder_channels"]),
        bottleneck_channels=int(m["bottleneck_channels"]),
        action_hidden_dims=tuple(m["action_hidden_dims"]),
        condition_dim=int(m["condition_dim"]),
        shock_dim=shock_dim,
        group_norm_groups=int(m["group_norm_groups"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg, z_lookup


def load_current(row: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
    d = cfg["data"]
    p = Path(d["root"]) / row["current_target_path"]
    a = np.asarray(np.load(p), dtype=np.float32)
    ch = int(d["pristine_channel"])
    thr = float(d["mask_threshold"])
    mask = (a[ch] >= thr).astype(np.float32)
    size = int(d.get("image_size", 256))
    if mask.shape != (size, size):
        t = torch.from_numpy(mask)[None, None]
        mask = F.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy()
    return torch.from_numpy(mask.astype(np.float32))[None, None]


def normalize_actions(actions, next_pulse: int, norm_cfg: Dict[str, Any]) -> np.ndarray:
    from transition.dataset import normalize_metadata

    arr = []
    for a in actions:
        raw = np.asarray([a.energy_uJ, a.action_y_um, a.action_z_um, float(next_pulse)], dtype=np.float32)
        arr.append(normalize_metadata(raw, norm_cfg))
    return np.stack(arr, axis=0).astype(np.float32)


def predict_27(
    model,
    current: torch.Tensor,
    z_base: Optional[torch.Tensor],
    actions_t: torch.Tensor,
    chunk: int,
    amp: bool,
    device: torch.device,
) -> np.ndarray:
    current = current.to(device)
    use_amp = bool(amp and device.type == "cuda")
    probs = []
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=use_amp):
            enc = model.encode_state(current)
            for start in range(0, len(actions_t), chunk):
                n = min(chunk, len(actions_t) - start)
                a = actions_t[start:start + n]
                z = None
                if str(model.variant).upper() == "B":
                    if z_base is None:
                        raise RuntimeError("Transition B requires z_shock")
                    z = z_base.expand(n, -1)
                out = model.decode_from_encoded(
                    current,
                    enc,
                    a,
                    z_shock=z,
                    repeats=n,
                )
                probs.append(out["next_pred"].float().cpu())
    return torch.cat(probs, dim=0).numpy()[:, 0]


def safe_ranked_actions(rows: Sequence[Dict[str, Any]]) -> List[str]:
    safe = [r for r in rows if bool(r["is_safe"])]
    return [str(r["action_id"]) for r in sorted(safe, key=lambda r: (float(r["cost_j"]), int(r["candidate_index"])))]


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    both_continue = [r for r in rows if r["fp32_decision"] == "continue" and r["amp_decision"] == "continue"]
    action_mismatch = [r for r in both_continue if not bool(r["selected_action_match"])]
    decision_mismatch = [r for r in rows if not bool(r["decision_match"])]
    safe_count_mismatch = [r for r in rows if not bool(r["safe_count_match"])]
    top3_mismatch = [r for r in both_continue if not bool(r["top3_set_match"])]
    return {
        "n_state_target_cases": n,
        "decision_match_fraction": 1.0 - len(decision_mismatch) / n,
        "decision_mismatch_count": len(decision_mismatch),
        "both_continue_count": len(both_continue),
        "selected_action_match_fraction_on_both_continue": (
            1.0 - len(action_mismatch) / len(both_continue) if both_continue else 1.0
        ),
        "selected_action_mismatch_count": len(action_mismatch),
        "safe_candidate_count_match_fraction": 1.0 - len(safe_count_mismatch) / n,
        "top3_set_match_fraction_on_both_continue": (
            1.0 - len(top3_mismatch) / len(both_continue) if both_continue else 1.0
        ),
        "candidate_safe_flag_disagreement_fraction_mean": float(np.mean([r["candidate_safe_flag_disagreement_fraction"] for r in rows])),
        "candidate_cost_max_abs_diff_max": float(np.max([r["candidate_cost_max_abs_diff"] for r in rows])),
        "candidate_probability_max_abs_diff_max": float(np.max([r["candidate_probability_max_abs_diff"] for r in rows])),
        "fp32_selected_margin_median": float(np.nanmedian([r["fp32_selected_margin"] for r in both_continue])) if both_continue else None,
        "mismatch_fp32_selected_margin_max": float(np.nanmax([r["fp32_selected_margin"] for r in action_mismatch])) if action_mismatch else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate AMP vs FP32 final action-selection equivalence under runtime B=1 conditions."
    )
    ap.add_argument("--runtime-config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--target-root", default=None, help="Directory containing prepared 256x256 target .npy files. Default: latest reachability run.")
    ap.add_argument("--modes", nargs="+", default=["metadata_only", "action_history_only", "shock_metadata", "shock_action_history"])
    ap.add_argument("--states-per-pulse", type=int, default=20)
    ap.add_argument("--pulses", nargs="+", type=int, default=[5,10,15,20,25,30,35,40,45])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for AMP equivalence validation")
    device = torch.device("cuda")

    cfg = load_runtime_config(Path(args.runtime_config))
    ml_repo = Path(cfg["paths"]["ml_repo"])
    target_root = discover_target_root(cfg, args.target_root)
    targets = load_targets(target_root)
    registry = ActionRegistry(Path(cfg["paths"]["action_registry_csv"]))

    model_meta = []
    for mode in args.modes:
        model_cfg = cfg["models"][mode]
        model_meta.append(load_transition_metadata(Path(model_cfg["transition_run"]), ml_repo))
    keys = select_common_keys(model_meta, args.states_per_pulse, args.seed, args.pulses)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["storage"]["session_root"]) / "offline_evaluation" / f"amp_action_equivalence_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    threshold = float(cfg["action_selection"]["candidate_threshold"])
    l1_weight = float(cfg["action_selection"]["l1_weight"])
    dice_weight = float(cfg["action_selection"]["dice_weight"])
    over_limit = float(cfg["stop_policy"]["over_removal_limit"])

    all_rows: List[Dict[str, Any]] = []
    mode_summaries: Dict[str, Any] = {}

    for mode, meta in zip(args.modes, model_meta):
        perf = (cfg["models"][mode].get("performance") or {})
        chunk = int(perf.get("candidate_chunk_size", 9))
        model, tr_cfg, z_lookup = load_model(meta, ml_repo, device)
        rows_by_key = meta["rows_by_key"]
        norm_cfg = tr_cfg["data"]["metadata_normalization"]

        iterator = tqdm(keys, desc=f"{mode}: FP32 vs AMP", unit="state", dynamic_ncols=True)
        for state_key in iterator:
            row = rows_by_key[state_key]
            current = load_current(row, tr_cfg)
            act_np = normalize_actions(registry.actions, int(row["next_pulse"]), norm_cfg)
            actions_t = torch.from_numpy(act_np).to(device)

            z_base = None
            if str(model.variant).upper() == "B":
                zid = str(row["current_g5_sample_id"])
                if z_lookup is None or zid not in z_lookup:
                    raise KeyError(f"Missing z_shock: {zid}")
                z_base = torch.from_numpy(np.asarray(z_lookup[zid], dtype=np.float32))[None].to(device)

            fp32 = predict_27(model, current, z_base, actions_t, chunk, False, device)
            amp = predict_27(model, current, z_base, actions_t, chunk, True, device)

            prob_diff = np.abs(fp32.astype(np.float32) - amp.astype(np.float32))
            for target_id, target in targets.items():
                fp_rows, fp_best, fp_safe_count, fp_no_safe = candidate_table(
                    fp32, target, registry.actions, threshold, l1_weight, dice_weight, over_limit
                )
                am_rows, am_best, am_safe_count, am_no_safe = candidate_table(
                    amp, target, registry.actions, threshold, l1_weight, dice_weight, over_limit
                )

                fp_decision = "stop" if fp_no_safe else "continue"
                am_decision = "stop" if am_no_safe else "continue"
                fp_action = None if fp_no_safe else str(fp_best["action_id"])
                am_action = None if am_no_safe else str(am_best["action_id"])

                fp_ranked = safe_ranked_actions(fp_rows)
                am_ranked = safe_ranked_actions(am_rows)
                fp_top3 = fp_ranked[:3]
                am_top3 = am_ranked[:3]

                fp_costs = np.asarray([r["cost_j"] for r in fp_rows], dtype=np.float64)
                am_costs = np.asarray([r["cost_j"] for r in am_rows], dtype=np.float64)
                fp_safe_flags = np.asarray([bool(r["is_safe"]) for r in fp_rows], dtype=bool)
                am_safe_flags = np.asarray([bool(r["is_safe"]) for r in am_rows], dtype=bool)

                if len(fp_ranked) >= 2:
                    by_id = {str(r["action_id"]): float(r["cost_j"]) for r in fp_rows}
                    fp_margin = by_id[fp_ranked[1]] - by_id[fp_ranked[0]]
                else:
                    fp_margin = float("nan")

                all_rows.append({
                    "mode": mode,
                    "hole_uid": state_key[0],
                    "current_pulse": state_key[1],
                    "next_pulse": state_key[2],
                    "target_id": target_id,
                    "candidate_chunk_size": chunk,
                    "fp32_decision": fp_decision,
                    "amp_decision": am_decision,
                    "decision_match": int(fp_decision == am_decision),
                    "fp32_safe_candidate_count": int(fp_safe_count),
                    "amp_safe_candidate_count": int(am_safe_count),
                    "safe_count_match": int(fp_safe_count == am_safe_count),
                    "fp32_action_id": fp_action,
                    "amp_action_id": am_action,
                    "selected_action_match": int(fp_action == am_action),
                    "fp32_top3": ";".join(fp_top3),
                    "amp_top3": ";".join(am_top3),
                    "top3_set_match": int(set(fp_top3) == set(am_top3)),
                    "fp32_selected_margin": fp_margin,
                    "candidate_safe_flag_disagreement_fraction": float(np.mean(fp_safe_flags != am_safe_flags)),
                    "candidate_cost_max_abs_diff": float(np.max(np.abs(fp_costs - am_costs))),
                    "candidate_probability_mean_abs_diff": float(np.mean(prob_diff)),
                    "candidate_probability_max_abs_diff": float(np.max(prob_diff)),
                })

        mode_rows = [r for r in all_rows if r["mode"] == mode]
        summary = summarize(mode_rows)
        summary.update({
            "mode": mode,
            "candidate_chunk_size": chunk,
            "n_unique_states": len(keys),
            "n_targets": len(targets),
            "transition_checkpoint": str(meta["checkpoint"]),
        })
        mode_summaries[mode] = summary
        del model
        torch.cuda.empty_cache()

    write_csv(all_rows, out_dir / "per_case_equivalence.csv")
    write_json(mode_summaries, out_dir / "summary.json")

    summary_rows = []
    for mode in args.modes:
        summary_rows.append(mode_summaries[mode])
    write_csv(summary_rows, out_dir / "summary.csv")

    overall_pass = True
    for mode, s in mode_summaries.items():
        # Fail closed: final decision and final selected action must be identical.
        if s.get("decision_mismatch_count", 0) != 0 or s.get("selected_action_mismatch_count", 0) != 0:
            overall_pass = False

    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "criterion": (
            "PASS requires zero AMP/FP32 decision mismatches and zero selected-action mismatches "
            "for all tested state-target cases."
        ),
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "target_root": str(target_root.resolve()),
        "states_per_pulse_requested": int(args.states_per_pulse),
        "pulses": [int(x) for x in args.pulses],
        "n_common_states": len(keys),
        "targets": sorted(targets),
        "modes": mode_summaries,
    }
    write_json(report, out_dir / "validation_report.json")

    print("\nAMP / FP32 action-selection equivalence validation complete.")
    print(f"Status : {report['status']}")
    print(f"States : {len(keys)}")
    print(f"Targets: {len(targets)}")
    for mode in args.modes:
        s = mode_summaries[mode]
        print(
            f"{mode}: decision mismatches={s['decision_mismatch_count']}, "
            f"action mismatches={s['selected_action_mismatch_count']}, "
            f"safe-count match={s['safe_candidate_count_match_fraction']:.6f}"
        )
    print(f"Output : {out_dir}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
