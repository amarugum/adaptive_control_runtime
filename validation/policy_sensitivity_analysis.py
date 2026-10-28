from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.ml_runtime import MLRuntime

# Reuse the exact same dataset discovery / runtime-input logic as the v1.11 replay.
from validation import replay_end_to_end_offline as replay


DEFAULT_MODES = [
    "metadata_only",
    "action_history_only",
    "shock_metadata",
    "shock_action_history",
]
DEFAULT_PULSES = [5, 10, 15, 20, 25, 30, 35, 40, 45]
DEFAULT_OVERALL_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_OVER_LIMITS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


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


def _finite_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _finite_median(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _finite_percentile(x: np.ndarray, q: float) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _candidate_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    binary_threshold: float,
    l1_weight: float,
    dice_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cost, overall, over arrays with shape [target, action]."""
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"candidate probabilities must be [A,H,W], got {p.shape}")
    tstack = np.asarray(targets, dtype=np.float32)
    if tstack.ndim != 3 or tstack.shape[1:] != p.shape[1:]:
        raise ValueError(f"target stack shape {tstack.shape} incompatible with predictions {p.shape}")

    pb = p >= float(binary_threshold)
    p_sum = p.sum(axis=(1, 2), dtype=np.float64)
    nt = tstack.shape[0]
    na = p.shape[0]
    cost = np.empty((nt, na), dtype=np.float32)
    overall = np.empty((nt, na), dtype=np.float32)
    over = np.empty((nt, na), dtype=np.float32)

    for ti in range(nt):
        t = tstack[ti]
        tb = t >= 0.5
        denom = max(int(tb.sum()), 1)
        overall[ti] = np.logical_and(pb, tb[None]).sum(axis=(1, 2)) / denom
        over[ti] = np.logical_and(pb, ~tb[None]).sum(axis=(1, 2)) / denom

        l1 = np.mean(np.abs(p - t[None]), axis=(1, 2), dtype=np.float64)
        inter = (p * t[None]).sum(axis=(1, 2), dtype=np.float64)
        t_sum = float(t.sum(dtype=np.float64))
        dice_loss = 1.0 - (2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
        cost[ti] = (float(l1_weight) * l1 + float(dice_weight) * dice_loss).astype(np.float32)

    return cost, overall, over


def _current_fractions(mask: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    tstack = np.asarray(targets, dtype=np.float32)
    overall = np.empty((tstack.shape[0],), dtype=np.float32)
    over = np.empty_like(overall)
    for ti, t in enumerate(tstack):
        tb = t >= 0.5
        denom = max(int(tb.sum()), 1)
        overall[ti] = np.logical_and(m, tb).sum() / denom
        over[ti] = np.logical_and(m, ~tb).sum() / denom
    return overall, over


def _setup_replay_objects(args, cfg: Dict[str, Any]):
    ml_repo = Path(cfg["paths"]["ml_repo"])
    replay.add_ml_repo(ml_repo)
    registry = ActionRegistry(Path(cfg["paths"]["action_registry_csv"]))
    target_root = replay.discover_target_root(cfg, args.target_root)
    target_dict = replay.load_targets(target_root, args.target_id)
    target_ids = list(target_dict.keys())
    target_stack = np.stack([target_dict[k] for k in target_ids], axis=0).astype(np.float32)
    exclude_path = Path(args.exclude_csv) if args.exclude_csv else None
    excluded_holes = replay.load_excluded_holes(exclude_path)

    runtimes: Dict[str, MLRuntime] = {}
    state_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    transition_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    all_maps: List[Dict[Tuple[str, int], Any]] = []
    state_cache: Dict[Tuple[Any, ...], Dict[Tuple[str, int], Dict[str, Any]]] = {}
    transition_cache: Dict[Tuple[Any, ...], Dict[Tuple[str, int], Dict[str, Any]]] = {}

    for mode_i, mode in enumerate(args.modes, start=1):
        prefix = f"[Setup {mode_i}/{len(args.modes)}] {mode}"
        with replay.setup_activity(f"{prefix} | load State Estimator + Transition checkpoints"):
            rt = MLRuntime(ml_repo, mode, cfg["models"][mode], cfg.get("device", "auto"))
        runtimes[mode] = rt

        skey = replay.state_map_cache_key(rt, args.split)
        if skey in state_cache:
            state_maps[mode] = state_cache[skey]
            tqdm.write(f"{prefix} | State Estimator sample map: cache hit ({len(state_maps[mode])} states)")
        else:
            family = str(rt.state_cfg["model"]["family"]).upper()
            with replay.setup_activity(
                f"{prefix} | build {family} State Estimator sample map (loads dataset metadata .npy files)"
            ):
                state_maps[mode] = replay.build_state_sample_map(rt, args.split)
            state_cache[skey] = state_maps[mode]
            tqdm.write(f"{prefix} | State Estimator sample map: {len(state_maps[mode])} states")

        tkey = replay.transition_map_cache_key(rt, args.split)
        if tkey in transition_cache:
            transition_maps[mode] = transition_cache[tkey]
            tqdm.write(f"{prefix} | Transition row map: cache hit ({len(transition_maps[mode])} transitions)")
        else:
            with replay.setup_activity(f"{prefix} | build Transition row map"):
                transition_maps[mode] = replay.build_transition_row_map(rt, args.split)
            transition_cache[tkey] = transition_maps[mode]
            tqdm.write(f"{prefix} | Transition row map: {len(transition_maps[mode])} transitions")
        all_maps.extend([state_maps[mode], transition_maps[mode]])

    with replay.setup_activity("[Setup] intersect common held-out physical states"):
        keys = replay.select_common_keys(
            all_maps,
            args.pulses,
            excluded_holes,
            args.max_states_per_pulse,
            args.seed,
        )
    return (
        ml_repo,
        registry,
        target_root,
        target_ids,
        target_stack,
        excluded_holes,
        runtimes,
        state_maps,
        transition_maps,
        keys,
    )


def build_cache(args, cfg: Dict[str, Any], out_dir: Path, cache_path: Path) -> Dict[str, Any]:
    (
        _ml_repo,
        registry,
        target_root,
        target_ids,
        target_stack,
        excluded_holes,
        runtimes,
        state_maps,
        transition_maps,
        keys,
    ) = _setup_replay_objects(args, cfg)

    modes = list(args.modes)
    nm, ns, nt, na = len(modes), len(keys), len(target_ids), len(registry.actions)
    print("=== Sensitivity cache setup complete ===")
    print(f"Modes          : {nm}")
    print(f"States         : {ns}")
    print(f"Targets        : {nt}")
    print(f"Actions        : {na}")
    print(f"Threshold pairs: {len(args.overall_thresholds) * len(args.over_limits)}")
    print("Building candidate-metric cache once; the threshold sweep after this is CPU-only.\n")

    shape_cur = (nm, ns, nt)
    shape_cand = (nm, ns, nt, na)
    est_cur_overall = np.empty(shape_cur, dtype=np.float32)
    est_cur_over = np.empty(shape_cur, dtype=np.float32)
    oracle_cur_overall = np.empty(shape_cur, dtype=np.float32)
    oracle_cur_over = np.empty(shape_cur, dtype=np.float32)
    est_cost = np.empty(shape_cand, dtype=np.float32)
    est_cand_overall = np.empty(shape_cand, dtype=np.float32)
    est_cand_over = np.empty(shape_cand, dtype=np.float32)
    oracle_cost = np.empty(shape_cand, dtype=np.float32)
    oracle_cand_overall = np.empty(shape_cand, dtype=np.float32)
    oracle_cand_over = np.empty(shape_cand, dtype=np.float32)

    binary_threshold = float(cfg["action_selection"]["candidate_threshold"])
    l1_weight = float(cfg["action_selection"]["l1_weight"])
    dice_weight = float(cfg["action_selection"]["dice_weight"])
    max_control_steps = int(cfg["stop_policy"]["max_control_steps"])

    hole_uids = np.asarray([k[0] for k in keys], dtype="U32")
    current_pulses = np.asarray([int(k[1]) for k in keys], dtype=np.int16)
    next_pulses = np.empty((nm, ns), dtype=np.int16)

    for mi, mode in enumerate(modes):
        rt = runtimes[mode]
        smap = state_maps[mode]
        tmap = transition_maps[mode]
        iterator = tqdm(keys, desc=f"[Cache {mi+1}/{nm}] {mode}", unit="state", dynamic_ncols=True)
        for si, (hole_uid, pulse) in enumerate(iterator):
            sample = smap[(hole_uid, pulse)]
            tr_row = tmap[(hole_uid, pulse)]
            next_pulse = int(tr_row["next_pulse"])
            next_pulses[mi, si] = next_pulse
            request = replay.build_runtime_request(sample, max_control_steps)
            shock = replay.load_raw_shock_sequence(sample, rt.state_model.uses_shock)
            prob_t, z = rt.infer_state(request, shock)
            prob = prob_t.detach().cpu().numpy()[0, 0].astype(np.float32)
            est_binary = (prob >= binary_threshold).astype(np.float32)
            true_current, _true_next = replay.load_true_masks(tr_row, rt.transition_cfg)

            ecur_o, ecur_x = _current_fractions(est_binary, target_stack)
            ocur_o, ocur_x = _current_fractions(true_current, target_stack)
            est_cur_overall[mi, si] = ecur_o
            est_cur_over[mi, si] = ecur_x
            oracle_cur_overall[mi, si] = ocur_o
            oracle_cur_over[mi, si] = ocur_x

            actions = [replay.CandidateAction(a, next_pulse) for a in registry.actions]
            est_t = torch.from_numpy(est_binary)[None, None]
            est_preds = rt.predict_candidates(est_t, z, actions)
            true_t = torch.from_numpy(true_current.astype(np.float32))[None, None]
            oracle_preds = rt.predict_candidates(true_t, z, actions)

            c, o, x = _candidate_metrics(est_preds, target_stack, binary_threshold, l1_weight, dice_weight)
            est_cost[mi, si], est_cand_overall[mi, si], est_cand_over[mi, si] = c, o, x
            c, o, x = _candidate_metrics(oracle_preds, target_stack, binary_threshold, l1_weight, dice_weight)
            oracle_cost[mi, si], oracle_cand_overall[mi, si], oracle_cand_over[mi, si] = c, o, x

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        modes=np.asarray(modes, dtype="U32"),
        target_ids=np.asarray(target_ids, dtype="U64"),
        action_ids=np.asarray([a.action_id for a in registry.actions], dtype="U16"),
        hole_uids=hole_uids,
        current_pulses=current_pulses,
        next_pulses=next_pulses,
        est_cur_overall=est_cur_overall,
        est_cur_over=est_cur_over,
        oracle_cur_overall=oracle_cur_overall,
        oracle_cur_over=oracle_cur_over,
        est_cost=est_cost,
        est_cand_overall=est_cand_overall,
        est_cand_over=est_cand_over,
        oracle_cost=oracle_cost,
        oracle_cand_overall=oracle_cand_overall,
        oracle_cand_over=oracle_cand_over,
    )

    meta = {
        "cache_version": "v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "modes": modes,
        "target_root": str(target_root),
        "target_ids": target_ids,
        "action_ids": [a.action_id for a in registry.actions],
        "split": args.split,
        "pulses": list(args.pulses),
        "n_common_states": len(keys),
        "excluded_holes_count": len(excluded_holes),
        "excluded_holes": sorted(excluded_holes),
        "fixed_candidate_scoring": {
            "candidate_threshold": binary_threshold,
            "l1_weight": l1_weight,
            "dice_weight": dice_weight,
        },
        "note": (
            "This cache contains current-state fractions and all 27 candidate costs/overall/over fractions "
            "for both estimated-current and oracle-current paths. Stop thresholds are intentionally not baked in."
        ),
    }
    write_json(meta, out_dir / "policy_cache_metadata.json")
    print(f"Candidate cache saved: {cache_path}")
    return meta


def load_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    z = np.load(cache_path, allow_pickle=False)
    required = {
        "modes", "target_ids", "action_ids", "hole_uids", "current_pulses", "next_pulses",
        "est_cur_overall", "est_cur_over", "oracle_cur_overall", "oracle_cur_over",
        "est_cost", "est_cand_overall", "est_cand_over",
        "oracle_cost", "oracle_cand_overall", "oracle_cand_over",
    }
    missing = required - set(z.files)
    if missing:
        raise KeyError(f"Sensitivity cache missing arrays: {sorted(missing)}")
    return {k: z[k] for k in z.files}


def _branch_policy(
    cur_overall: np.ndarray,
    cur_over: np.ndarray,
    cand_cost: np.ndarray,
    cand_over: np.ndarray,
    overall_thr: float,
    over_limit: float,
    current_pulses: np.ndarray,
    max_control_steps: int,
) -> Dict[str, np.ndarray]:
    """Evaluate one branch for [state,target] arrays and [state,target,action] candidates."""
    stop_over = cur_over >= float(over_limit)
    stop_reached = (~stop_over) & (cur_overall >= float(overall_thr))
    step2d = (np.asarray(current_pulses, dtype=np.int32) // 5)[:, None]
    stop_max = (~stop_over) & (~stop_reached) & (step2d >= int(max_control_steps))
    active = ~(stop_over | stop_reached | stop_max)

    safe = cand_over < float(over_limit)
    safe_count = safe.sum(axis=-1).astype(np.int16)
    no_safe = active & (safe_count == 0)
    cont = active & (safe_count > 0)

    masked_cost = np.where(safe, cand_cost, np.inf)
    selected_idx = np.argmin(masked_cost, axis=-1).astype(np.int16)
    selected_idx = np.where(cont, selected_idx, -1).astype(np.int16)

    reason = np.zeros(cur_overall.shape, dtype=np.int8)  # 0=continue, 1=over,2=reached,3=max,4=no-safe
    reason[stop_over] = 1
    reason[stop_reached] = 2
    reason[stop_max] = 3
    reason[no_safe] = 4
    return {
        "continue": cont,
        "reason": reason,
        "safe_count": safe_count,
        "selected_idx": selected_idx,
        "safe": safe,
    }


def _take_candidate(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    safe_idx = np.clip(idx, 0, a.shape[-1] - 1)
    return np.take_along_axis(a, safe_idx[..., None], axis=-1)[..., 0]


def _aggregate_slice(
    mode: str,
    overall_thr: float,
    over_limit: float,
    current_pulses: np.ndarray,
    est_cur_overall: np.ndarray,
    est_cur_over: np.ndarray,
    oracle_cur_overall: np.ndarray,
    oracle_cur_over: np.ndarray,
    est_cost: np.ndarray,
    est_cand_overall: np.ndarray,
    est_cand_over: np.ndarray,
    oracle_cost: np.ndarray,
    oracle_cand_overall: np.ndarray,
    oracle_cand_over: np.ndarray,
    max_control_steps: int,
) -> Dict[str, Any]:
    e = _branch_policy(
        est_cur_overall, est_cur_over, est_cost, est_cand_over,
        overall_thr, over_limit, current_pulses, max_control_steps,
    )
    o = _branch_policy(
        oracle_cur_overall, oracle_cur_over, oracle_cost, oracle_cand_over,
        overall_thr, over_limit, current_pulses, max_control_steps,
    )
    ec, oc = e["continue"], o["continue"]
    decision_match = ec == oc
    false_continue = ec & (~oc)
    false_stop = (~ec) & oc
    both_continue = ec & oc
    both_n = int(both_continue.sum())

    e_idx, o_idx = e["selected_idx"], o["selected_idx"]
    action_match = both_continue & (e_idx == o_idx)

    oracle_over_of_est = _take_candidate(oracle_cand_over, e_idx)
    oracle_overall_of_est = _take_candidate(oracle_cand_overall, e_idx)
    oracle_cost_of_est = _take_candidate(oracle_cost, e_idx)
    oracle_cost_of_oracle = _take_candidate(oracle_cost, o_idx)
    est_over_sel = _take_candidate(est_cand_over, e_idx)
    est_overall_sel = _take_candidate(est_cand_overall, e_idx)
    oracle_unsafe_est_action = both_continue & (oracle_over_of_est >= float(over_limit))
    est_action_oracle_safe = both_continue & (~oracle_unsafe_est_action)
    regret = oracle_cost_of_est - oracle_cost_of_oracle

    n = int(ec.size)
    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if n else float("nan")

    # Reason codes: 1 over-removal, 2 target reached, 3 max steps, 4 no-safe.
    row: Dict[str, Any] = {
        "mode": mode,
        "overall_threshold": float(overall_thr),
        "over_limit": float(over_limit),
        "n": n,
        "estimated_continue_fraction": frac(ec),
        "oracle_continue_fraction": frac(oc),
        "decision_match_fraction": frac(decision_match),
        "false_continue_fraction": frac(false_continue),
        "false_stop_fraction": frac(false_stop),
        "false_continue_due_oracle_over_fraction": frac(false_continue & (o["reason"] == 1)),
        "false_continue_due_oracle_target_reached_fraction": frac(false_continue & (o["reason"] == 2)),
        "false_continue_due_oracle_no_safe_fraction": frac(false_continue & (o["reason"] == 4)),
        "estimated_over_removal_stop_fraction": frac(e["reason"] == 1),
        "oracle_over_removal_stop_fraction": frac(o["reason"] == 1),
        "estimated_target_reached_fraction": frac(e["reason"] == 2),
        "oracle_target_reached_fraction": frac(o["reason"] == 2),
        "estimated_no_safe_stop_fraction": frac(e["reason"] == 4),
        "oracle_no_safe_stop_fraction": frac(o["reason"] == 4),
        "estimated_max_steps_stop_fraction": frac(e["reason"] == 3),
        "oracle_max_steps_stop_fraction": frac(o["reason"] == 3),
        "estimated_safe_candidate_count_mean": _finite_mean(np.where(ec | (e["reason"] == 4), e["safe_count"], np.nan)),
        "oracle_safe_candidate_count_mean": _finite_mean(np.where(oc | (o["reason"] == 4), o["safe_count"], np.nan)),
        "both_continue_count": both_n,
        "selected_action_match_fraction_on_both_continue": (
            float(action_match.sum() / both_n) if both_n else float("nan")
        ),
        "estimated_action_oracle_unsafe_fraction_on_both_continue": (
            float(oracle_unsafe_est_action.sum() / both_n) if both_n else float("nan")
        ),
        "oracle_regret_mean_when_est_action_oracle_safe": _finite_mean(np.where(est_action_oracle_safe, regret, np.nan)),
        "oracle_regret_median_when_est_action_oracle_safe": _finite_median(np.where(est_action_oracle_safe, regret, np.nan)),
        "oracle_regret_p95_when_est_action_oracle_safe": _finite_percentile(np.where(est_action_oracle_safe, regret, np.nan), 95),
        "estimated_selected_predicted_overall_mean_on_continue": _finite_mean(np.where(ec, est_overall_sel, np.nan)),
        "estimated_selected_predicted_over_mean_on_continue": _finite_mean(np.where(ec, est_over_sel, np.nan)),
        "oracle_eval_of_est_selected_overall_mean_on_both_continue": _finite_mean(np.where(both_continue, oracle_overall_of_est, np.nan)),
        "oracle_eval_of_est_selected_over_mean_on_both_continue": _finite_mean(np.where(both_continue, oracle_over_of_est, np.nan)),
    }
    return row


def continuous_fraction_summary(cache: Dict[str, np.ndarray]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    modes = [str(x) for x in cache["modes"].tolist()]
    targets = [str(x) for x in cache["target_ids"].tolist()]
    rows_all: List[Dict[str, Any]] = []
    rows_target: List[Dict[str, Any]] = []
    for mi, mode in enumerate(modes):
        eo = cache["est_cur_overall"][mi]
        ex = cache["est_cur_over"][mi]
        oo = cache["oracle_cur_overall"][mi]
        ox = cache["oracle_cur_over"][mi]
        rows_all.append({
            "mode": mode,
            "n": int(eo.size),
            "estimated_overall_mean": float(np.mean(eo)),
            "oracle_overall_mean": float(np.mean(oo)),
            "overall_bias_mean_est_minus_oracle": float(np.mean(eo - oo)),
            "overall_mae_est_vs_oracle": float(np.mean(np.abs(eo - oo))),
            "overall_correlation": _corr(eo, oo),
            "estimated_over_mean": float(np.mean(ex)),
            "oracle_over_mean": float(np.mean(ox)),
            "over_bias_mean_est_minus_oracle": float(np.mean(ex - ox)),
            "over_mae_est_vs_oracle": float(np.mean(np.abs(ex - ox))),
            "over_correlation": _corr(ex, ox),
        })
        for ti, target_id in enumerate(targets):
            eot, ext, oot, oxt = eo[:, ti], ex[:, ti], oo[:, ti], ox[:, ti]
            rows_target.append({
                "mode": mode,
                "target_id": target_id,
                "n": int(eot.size),
                "estimated_overall_mean": float(np.mean(eot)),
                "oracle_overall_mean": float(np.mean(oot)),
                "overall_bias_mean_est_minus_oracle": float(np.mean(eot - oot)),
                "overall_mae_est_vs_oracle": float(np.mean(np.abs(eot - oot))),
                "overall_correlation": _corr(eot, oot),
                "estimated_over_mean": float(np.mean(ext)),
                "oracle_over_mean": float(np.mean(oxt)),
                "over_bias_mean_est_minus_oracle": float(np.mean(ext - oxt)),
                "over_mae_est_vs_oracle": float(np.mean(np.abs(ext - oxt))),
                "over_correlation": _corr(ext, oxt),
            })
    return rows_all, rows_target


def analyze_cache(args, cfg: Dict[str, Any], cache: Dict[str, np.ndarray], out_dir: Path) -> None:
    modes = [str(x) for x in cache["modes"].tolist()]
    target_ids = [str(x) for x in cache["target_ids"].tolist()]
    current_pulses = cache["current_pulses"].astype(np.int32)
    max_steps = int(cfg["stop_policy"]["max_control_steps"])

    cont_all, cont_target = continuous_fraction_summary(cache)
    write_csv(cont_all, out_dir / "continuous_current_fraction_summary.csv")
    write_csv(cont_target, out_dir / "continuous_current_fraction_by_target.csv")

    summary_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    pulse_rows: List[Dict[str, Any]] = []
    total_pairs = len(args.overall_thresholds) * len(args.over_limits) * len(modes)
    pbar = tqdm(total=total_pairs, desc="[Sweep] policy grid", unit="mode-pair", dynamic_ncols=True)

    for mi, mode in enumerate(modes):
        arrays = {
            "est_cur_overall": cache["est_cur_overall"][mi],
            "est_cur_over": cache["est_cur_over"][mi],
            "oracle_cur_overall": cache["oracle_cur_overall"][mi],
            "oracle_cur_over": cache["oracle_cur_over"][mi],
            "est_cost": cache["est_cost"][mi],
            "est_cand_overall": cache["est_cand_overall"][mi],
            "est_cand_over": cache["est_cand_over"][mi],
            "oracle_cost": cache["oracle_cost"][mi],
            "oracle_cand_overall": cache["oracle_cand_overall"][mi],
            "oracle_cand_over": cache["oracle_cand_over"][mi],
        }
        for overall_thr in args.overall_thresholds:
            for over_limit in args.over_limits:
                row = _aggregate_slice(
                    mode, overall_thr, over_limit, current_pulses,
                    arrays["est_cur_overall"], arrays["est_cur_over"],
                    arrays["oracle_cur_overall"], arrays["oracle_cur_over"],
                    arrays["est_cost"], arrays["est_cand_overall"], arrays["est_cand_over"],
                    arrays["oracle_cost"], arrays["oracle_cand_overall"], arrays["oracle_cand_over"],
                    max_steps,
                )
                summary_rows.append(row)

                for ti, target_id in enumerate(target_ids):
                    tr = _aggregate_slice(
                        mode, overall_thr, over_limit, current_pulses,
                        arrays["est_cur_overall"][:, ti:ti+1], arrays["est_cur_over"][:, ti:ti+1],
                        arrays["oracle_cur_overall"][:, ti:ti+1], arrays["oracle_cur_over"][:, ti:ti+1],
                        arrays["est_cost"][:, ti:ti+1], arrays["est_cand_overall"][:, ti:ti+1], arrays["est_cand_over"][:, ti:ti+1],
                        arrays["oracle_cost"][:, ti:ti+1], arrays["oracle_cand_overall"][:, ti:ti+1], arrays["oracle_cand_over"][:, ti:ti+1],
                        max_steps,
                    )
                    tr["target_id"] = target_id
                    target_rows.append(tr)

                for pulse in sorted(set(int(x) for x in current_pulses.tolist())):
                    keep = current_pulses == pulse
                    pr = _aggregate_slice(
                        mode, overall_thr, over_limit, current_pulses[keep],
                        arrays["est_cur_overall"][keep], arrays["est_cur_over"][keep],
                        arrays["oracle_cur_overall"][keep], arrays["oracle_cur_over"][keep],
                        arrays["est_cost"][keep], arrays["est_cand_overall"][keep], arrays["est_cand_over"][keep],
                        arrays["oracle_cost"][keep], arrays["oracle_cand_overall"][keep], arrays["oracle_cand_over"][keep],
                        max_steps,
                    )
                    pr["current_pulse"] = int(pulse)
                    pulse_rows.append(pr)
                pbar.update(1)
    pbar.close()

    write_csv(summary_rows, out_dir / "policy_grid_summary.csv")
    write_csv(target_rows, out_dir / "policy_grid_by_target.csv")
    write_csv(pulse_rows, out_dir / "policy_grid_by_pulse.csv")

    # A compact comparison table at the currently configured policy, if that exact pair was swept.
    cur_overall = float(cfg["stop_policy"]["overall_removal_target"])
    cur_over = float(cfg["stop_policy"]["over_removal_limit"])
    cur_rows = [
        r for r in summary_rows
        if math.isclose(float(r["overall_threshold"]), cur_overall, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(r["over_limit"]), cur_over, rel_tol=0, abs_tol=1e-12)
    ]
    write_csv(cur_rows, out_dir / "current_policy_comparison.csv")

    # No automatic best-policy selection: scientific/safety preference is not encoded in one scalar.
    resolved = {
        "analysis_version": "v1.12",
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "cache_path": str(Path(args.cache).resolve()),
        "overall_thresholds": [float(x) for x in args.overall_thresholds],
        "over_limits": [float(x) for x in args.over_limits],
        "current_runtime_policy": {
            "overall_removal_target": cur_overall,
            "over_removal_limit": cur_over,
            "max_control_steps": max_steps,
        },
        "interpretation": [
            "Continuous overall/over fractions are kept as model-comparison outcomes; the sweep is only for the hard runtime stop policy.",
            "The same cached candidate predictions are reused for every threshold pair, so threshold comparisons are not confounded by repeated stochastic inference.",
            "Oracle-current replaces only the current mask with the measured mask; Transition and, for Transition B, z_shock remain model outputs as in the end-to-end replay.",
            "Historical fixed-plan trajectories were not optimized toward the eight targets; target-reached frequency must not be maximized blindly to choose thresholds.",
            "No automatic best threshold pair is declared. Prioritize low false-continue / oracle-unsafe-action rates, then inspect premature-stop and target-specific behavior.",
        ],
    }
    write_json(resolved, out_dir / "resolved_policy_sensitivity.json")
    make_heatmaps(summary_rows, out_dir / "plots")


def make_heatmaps(rows: Sequence[Dict[str, Any]], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped: matplotlib unavailable ({exc})")
        return
    if not rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    modes = sorted({str(r["mode"]) for r in rows})
    metrics = [
        ("decision_match_fraction", "Decision match"),
        ("false_continue_fraction", "False continue"),
        ("estimated_action_oracle_unsafe_fraction_on_both_continue", "Estimated action oracle-unsafe"),
        ("estimated_target_reached_fraction", "Estimated target reached"),
    ]
    for mode in modes:
        rr = [r for r in rows if str(r["mode"]) == mode]
        xs = sorted({float(r["over_limit"]) for r in rr})
        ys = sorted({float(r["overall_threshold"]) for r in rr})
        for metric, title in metrics:
            mat = np.full((len(ys), len(xs)), np.nan, dtype=float)
            for r in rr:
                yi = ys.index(float(r["overall_threshold"]))
                xi = xs.index(float(r["over_limit"]))
                mat[yi, xi] = float(r.get(metric, np.nan))
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
            ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
            ax.set_xlabel("Over-removal limit")
            ax.set_ylabel("Overall target threshold")
            ax.set_title(f"{mode}: {title}")
            for yi in range(len(ys)):
                for xi in range(len(xs)):
                    if np.isfinite(mat[yi, xi]):
                        ax.text(xi, yi, f"{mat[yi, xi]:.3f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{mode}_{metric}.png", dpi=160)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for overall/over-removal stop policy. Builds one reusable cache of "
            "all 27 candidate metrics for estimated-current and oracle-current paths, then sweeps stop thresholds."
        )
    )
    ap.add_argument("--runtime-config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pulses", nargs="+", type=int, default=DEFAULT_PULSES)
    ap.add_argument("--max-states-per-pulse", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-root", default=None)
    ap.add_argument("--target-id", nargs="*", default=None)
    ap.add_argument(
        "--exclude-csv",
        default=str(Path(__file__).with_name("artifact_exclusions_20261008.csv")),
        help="Hole-level exclusion CSV. Pass empty string to disable.",
    )
    ap.add_argument("--overall-thresholds", nargs="+", type=float, default=DEFAULT_OVERALL_THRESHOLDS)
    ap.add_argument("--over-limits", nargs="+", type=float, default=DEFAULT_OVER_LIMITS)
    ap.add_argument("--cache", default=None, help="Existing/new .npz candidate cache. If omitted, output_dir/policy_candidate_cache.npz is used.")
    ap.add_argument("--reuse-cache", action="store_true", help="Require and reuse --cache instead of running GPU inference.")
    ap.add_argument("--cache-only", action="store_true", help="Build cache and stop before threshold sweep.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if any(not (0.0 <= x <= 1.0) for x in args.overall_thresholds):
        ap.error("--overall-thresholds must be within [0,1]")
    if any(x < 0.0 for x in args.over_limits):
        ap.error("--over-limits must be >= 0")
    args.overall_thresholds = sorted(set(float(x) for x in args.overall_thresholds))
    args.over_limits = sorted(set(float(x) for x in args.over_limits))
    return args


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(Path(args.runtime_config))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["storage"]["session_root"]) / "offline_evaluation" / f"policy_sensitivity_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)

    cache_path = Path(args.cache) if args.cache else (out_dir / "policy_candidate_cache.npz")
    args.cache = str(cache_path)

    print("=== Policy sensitivity analysis ===")
    print(f"Runtime config     : {Path(args.runtime_config).resolve()}")
    print(f"Modes              : {', '.join(args.modes)}")
    print(f"Overall thresholds : {args.overall_thresholds}")
    print(f"Over limits        : {args.over_limits}")
    print(f"Max/pulse          : {args.max_states_per_pulse if args.max_states_per_pulse is not None else 'all'}")
    print(f"Output             : {out_dir}")
    print(f"Cache              : {cache_path}")
    print()

    if args.reuse_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f"--reuse-cache requested but cache does not exist: {cache_path}")
        print("Reusing candidate cache; no GPU inference will run.")
    else:
        build_cache(args, cfg, out_dir, cache_path)

    if args.cache_only:
        print("Cache build complete (--cache-only).")
        return 0

    cache = load_cache(cache_path)
    print("Running threshold sweep from cache...")
    analyze_cache(args, cfg, cache, out_dir)
    print("Policy sensitivity analysis complete.")
    print(f"Main summary : {out_dir / 'policy_grid_summary.csv'}")
    print(f"By target    : {out_dir / 'policy_grid_by_target.csv'}")
    print(f"Continuous   : {out_dir / 'continuous_current_fraction_summary.csv'}")
    print(f"Plots        : {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
