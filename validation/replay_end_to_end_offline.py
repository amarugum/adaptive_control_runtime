from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import threading
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Direct execution from adaptive_control_runtime/.
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.ml_runtime import MLRuntime
from runtime.selector import candidate_table, fractions




# ---------------------------------------------------------------------------
# Setup progress helpers
# ---------------------------------------------------------------------------

@contextmanager
def setup_activity(desc: str, refresh_s: float = 1.0):
    """Show an elapsed-time heartbeat while a setup stage performs opaque I/O.

    Some dataset builders internally load thousands of small .npy files and do not
    expose item-level progress.  This heartbeat makes it clear which preparation
    stage is active without modifying the training/data repository.
    """
    bar = tqdm(
        total=None,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        bar_format="{desc} [{elapsed}]",
    )
    stop = threading.Event()

    def _refresh() -> None:
        while not stop.wait(max(0.2, float(refresh_s))):
            try:
                bar.refresh()
            except Exception:
                return

    worker = threading.Thread(target=_refresh, daemon=True)
    worker.start()
    t0 = time.perf_counter()
    try:
        yield
    except BaseException:
        stop.set()
        worker.join(timeout=2.0)
        elapsed = time.perf_counter() - t0
        bar.set_description_str(f"{desc} FAILED ({elapsed:.1f}s)")
        bar.refresh()
        bar.close()
        raise
    else:
        stop.set()
        worker.join(timeout=2.0)
        elapsed = time.perf_counter() - t0
        bar.set_description_str(f"{desc} done ({elapsed:.1f}s)")
        bar.refresh()
        bar.close()


def state_map_cache_key(rt: MLRuntime, split: str) -> Tuple[Any, ...]:
    """Key only on fields that affect build_g1_samples/build_g5_samples output."""
    cfg = rt.state_cfg
    d = cfg["data"]
    m = cfg["model"]
    family = str(m["family"]).upper()
    family_option = (
        str(d.get("g1_sampling", "block_end"))
        if family == "G1"
        else bool(d["require_same_action_within_g5_block"])
    )
    return (
        family,
        str(Path(d["root"]).resolve()),
        str(d["shock_filename"]),
        str(d["target_filename"]),
        str(d["metadata_filename"]),
        str(Path(d["split_csv"]).resolve()),
        int(d["block_size"]),
        int(d.get("history_max_steps", 10)),
        family_option,
        str(split),
    )


def transition_map_cache_key(rt: MLRuntime, split: str) -> Tuple[Any, ...]:
    d = rt.transition_cfg["data"]
    return (
        str(Path(d["root"]).resolve()),
        str(d["manifest_name"]),
        str(Path(d["split_csv"]).resolve()),
        str(split),
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


def add_ml_repo(ml_repo: Path) -> None:
    p = str(Path(ml_repo).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


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
            "No target_reachability_*/targets_256 found. Pass --target-root explicitly."
        )
    return candidates[-1] / "targets_256"


def load_targets(target_root: Path, target_ids: Optional[Sequence[str]]) -> Dict[str, np.ndarray]:
    wanted = set(target_ids or [])
    files = sorted(target_root.glob("*.npy"))
    if wanted:
        files = [p for p in files if p.stem in wanted]
    if not files:
        raise RuntimeError(f"No requested .npy targets found in {target_root}")
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
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"Targets not found: {sorted(missing)}")
    return out


def load_excluded_holes(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "hole_uid" not in reader.fieldnames:
            raise ValueError("Exclusion CSV must contain hole_uid")
        return {str(r["hole_uid"]).strip() for r in reader if str(r.get("hole_uid", "")).strip()}


# ---------------------------------------------------------------------------
# Dataset / model metadata
# ---------------------------------------------------------------------------

def physical_key(hole_uid: str, current_pulse: int) -> Tuple[str, int]:
    return str(hole_uid), int(current_pulse)


def build_state_sample_map(rt: MLRuntime, split: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    add_ml_repo(rt.ml_repo)
    from state_estimator.dataset import (
        build_g1_samples,
        build_g5_samples,
        discover_pulses,
        load_dataset_split,
    )

    cfg = rt.state_cfg
    d = cfg["data"]
    m = cfg["model"]
    data_root = Path(d["root"])
    records = discover_pulses(
        data_root,
        d["shock_filename"],
        d["target_filename"],
        d["metadata_filename"],
    )
    holes = sorted({r.hole_uid for r in records})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)
    allowed = splits[split]
    common = dict(
        block_size=int(d["block_size"]),
        history_max_steps=int(d.get("history_max_steps", 10)),
    )
    if str(m["family"]).upper() == "G1":
        samples = build_g1_samples(
            records,
            allowed,
            sampling=d.get("g1_sampling", "block_end"),
            **common,
        )
    else:
        samples = build_g5_samples(
            records,
            allowed,
            require_same_action=bool(d["require_same_action_within_g5_block"]),
            **common,
        )
    return {physical_key(s["hole_uid"], s["pulse_number"]): s for s in samples}


def build_transition_row_map(rt: MLRuntime, split: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    add_ml_repo(rt.ml_repo)
    from transition.dataset import load_dataset_split, load_manifest

    d = rt.transition_cfg["data"]
    root = Path(d["root"])
    rows = load_manifest(root / d["manifest_name"])
    holes = sorted({str(r["hole_uid"]) for r in rows})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)
    allowed = set(splits[split])
    rows = [r for r in rows if str(r["hole_uid"]) in allowed]
    return {physical_key(r["hole_uid"], int(r["current_pulse"])): r for r in rows}


def select_common_keys(
    maps: Sequence[Dict[Tuple[str, int], Any]],
    pulses: Sequence[int],
    excluded_holes: set[str],
    max_states_per_pulse: Optional[int],
    seed: int,
) -> List[Tuple[str, int]]:
    common = set.intersection(*(set(m) for m in maps))
    common = {k for k in common if k[0] not in excluded_holes and int(k[1]) in set(pulses)}
    rng = np.random.default_rng(seed)
    selected: List[Tuple[str, int]] = []
    for pulse in pulses:
        group = sorted(k for k in common if int(k[1]) == int(pulse))
        if max_states_per_pulse is not None and len(group) > max_states_per_pulse:
            idx = sorted(rng.choice(len(group), size=max_states_per_pulse, replace=False).tolist())
            group = [group[i] for i in idx]
        selected.extend(group)
    if not selected:
        raise RuntimeError("No common test states selected")
    return selected


# ---------------------------------------------------------------------------
# Runtime-like input preparation
# ---------------------------------------------------------------------------

def build_runtime_request(sample: Dict[str, Any], max_control_steps: int) -> Dict[str, Any]:
    pulse_end = int(sample["pulse_number"])
    meta = np.asarray(sample["metadata_raw"], dtype=np.float32).reshape(-1)
    history_raw = np.asarray(sample["action_history_raw"], dtype=np.float32)
    history_len = int(sample["history_length"])
    history = []
    for row in history_raw[:history_len]:
        history.append({
            "energy_uJ": float(row[0]),
            "action_y_um": float(row[1]),
            "action_z_um": float(row[2]),
            "pulse_count": int(round(float(row[3]))),
        })
    return {
        "executed_action": {
            "energy_uJ": float(meta[0]),
            "action_y_um": float(meta[1]),
            "action_z_um": float(meta[2]),
        },
        "action_history": history,
        "step": {
            "completed_control_step": pulse_end // 5,
            "pulse_end": pulse_end,
        },
        "control": {"max_control_steps": int(max_control_steps)},
    }


def load_raw_shock_sequence(sample: Dict[str, Any], uses_shock: bool) -> Optional[np.ndarray]:
    if not uses_shock:
        return None
    xs = [np.asarray(np.load(r.shock_path), dtype=np.float32) for r in sample["records"]]
    if len(xs) == 1:
        return xs[0]
    return np.stack(xs, axis=0)


def load_mask(path: Path, channel: int, threshold: float, size: int) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"target.npy must be [C,H,W]: {path}, got {arr.shape}")
    m = (arr[channel] >= threshold).astype(np.float32)
    if m.shape != (size, size):
        t = torch.from_numpy(m)[None, None]
        m = F.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy()
    return m.astype(np.float32, copy=False)


def load_true_masks(row: Dict[str, Any], tr_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    d = tr_cfg["data"]
    root = Path(d["root"])
    ch = int(d["pristine_channel"])
    thr = float(d["mask_threshold"])
    size = int(d.get("image_size", 256))
    cur = load_mask(root / row["current_target_path"], ch, thr, size)
    nxt_raw = load_mask(root / row["next_target_path"], ch, thr, size)
    nxt = np.maximum(cur, nxt_raw).astype(np.float32, copy=False)
    return cur, nxt


class CandidateAction:
    def __init__(self, a, next_pulse: int):
        self.action_id = a.action_id
        self.energy_uJ = a.energy_uJ
        self.action_y_um = a.action_y_um
        self.action_z_um = a.action_z_um
        self.next_pulse = int(next_pulse)


def actual_action_index(row: Dict[str, Any], registry: ActionRegistry) -> int:
    key = (
        round(float(row["energy_uJ"]), 6),
        round(float(row["action_y_um"]), 6),
        round(float(row["action_z_um"]), 6),
    )
    matches = [
        i for i, a in enumerate(registry.actions)
        if (
            round(float(a.energy_uJ), 6),
            round(float(a.action_y_um), 6),
            round(float(a.action_z_um), 6),
        ) == key
    ]
    if len(matches) != 1:
        raise ValueError(f"Observed action does not uniquely match registry: {key}, matches={matches}")
    return int(matches[0])


# ---------------------------------------------------------------------------
# Metrics and decision logic
# ---------------------------------------------------------------------------

def binary_metrics(pred_prob: np.ndarray, true_mask: np.ndarray, threshold: float) -> Dict[str, float]:
    p = np.asarray(pred_prob, dtype=np.float32)
    t = np.asarray(true_mask, dtype=np.float32)
    pb = p >= threshold
    tb = t >= 0.5
    inter = int(np.logical_and(pb, tb).sum())
    union = int(np.logical_or(pb, tb).sum())
    ps = int(pb.sum())
    ts = int(tb.sum())
    dice = (2.0 * inter) / max(ps + ts, 1)
    iou = inter / max(union, 1)
    return {
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        "dice": float(dice),
        "iou": float(iou),
        "binary_disagreement_fraction": float(np.mean(pb != tb)),
        "processed_area_bias_fraction": float((ps - ts) / max(ts, 1)),
    }


def stop_from_current(mask: np.ndarray, target: np.ndarray, cfg: Dict[str, Any], control_step: int) -> Tuple[Optional[str], Dict[str, float]]:
    fr = fractions(mask, target)
    sp = cfg["stop_policy"]
    reason = None
    if fr["over_removal_fraction"] >= float(sp["over_removal_limit"]):
        reason = "over_removal"
    elif fr["overall_removal_fraction"] >= float(sp["overall_removal_target"]):
        reason = "target_reached"
    elif int(control_step) >= int(sp["max_control_steps"]):
        reason = "max_control_steps"
    return reason, fr


def top3_safe_set(rows: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    safe = sorted(
        [r for r in rows if bool(r["is_safe"])],
        key=lambda r: (float(r["cost_j"]), int(r["candidate_index"])),
    )
    return tuple(str(r["action_id"]) for r in safe[:3])


def evaluate_target_decision(
    current_mask: np.ndarray,
    candidate_probs: np.ndarray,
    target: np.ndarray,
    registry: ActionRegistry,
    cfg: Dict[str, Any],
    control_step: int,
) -> Dict[str, Any]:
    current_stop, current_fr = stop_from_current(current_mask, target, cfg, control_step)
    out: Dict[str, Any] = {
        "decision": "stop" if current_stop else "continue",
        "stop_reason": current_stop,
        "safe_candidate_count": None,
        "selected_action_id": None,
        "top3_safe": tuple(),
        **{f"current_{k}": v for k, v in current_fr.items()},
    }
    if current_stop:
        return out

    a = cfg["action_selection"]
    sp = cfg["stop_policy"]
    rows, best, safe_count, no_safe = candidate_table(
        candidate_probs,
        target,
        registry.actions,
        float(a["candidate_threshold"]),
        float(a["l1_weight"]),
        float(a["dice_weight"]),
        float(sp["over_removal_limit"]),
    )
    out["safe_candidate_count"] = int(safe_count)
    out["top3_safe"] = top3_safe_set(rows)
    if no_safe:
        out["decision"] = "stop"
        out["stop_reason"] = "no_safe_action"
        return out
    out["decision"] = "continue"
    out["stop_reason"] = None
    out["selected_action_id"] = str(best["action_id"])
    out["selected_cost"] = float(best["cost_j"])
    out["selected_over_fraction"] = float(best["predicted_over_removal_fraction"])
    return out


def summarize_numeric(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_median"] = float(np.median(vals))
            out[f"{key}_p95"] = float(np.percentile(vals, 95))
    return out


def aggregate_control(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    both_continue = [r for r in rows if r["estimated_decision"] == "continue" and r["oracle_decision"] == "continue"]
    action_match = [r for r in both_continue if bool(r["selected_action_match"])]
    top3_match = [r for r in both_continue if bool(r["top3_set_match"])]
    return {
        "n": len(rows),
        "decision_match_fraction": float(np.mean([bool(r["decision_match"]) for r in rows])),
        "both_continue_count": len(both_continue),
        "selected_action_match_fraction_on_both_continue": (
            len(action_match) / len(both_continue) if both_continue else 1.0
        ),
        "top3_set_match_fraction_on_both_continue": (
            len(top3_match) / len(both_continue) if both_continue else 1.0
        ),
        "estimated_continue_fraction": float(np.mean([r["estimated_decision"] == "continue" for r in rows])),
        "oracle_continue_fraction": float(np.mean([r["oracle_decision"] == "continue" for r in rows])),
        "estimated_no_safe_fraction": float(np.mean([r["estimated_stop_reason"] == "no_safe_action" for r in rows])),
        "oracle_no_safe_fraction": float(np.mean([r["oracle_stop_reason"] == "no_safe_action" for r in rows])),
        "estimated_over_removal_stop_fraction": float(np.mean([r["estimated_stop_reason"] == "over_removal" for r in rows])),
        "oracle_over_removal_stop_fraction": float(np.mean([r["oracle_stop_reason"] == "over_removal" for r in rows])),
        "estimated_target_reached_fraction": float(np.mean([r["estimated_stop_reason"] == "target_reached" for r in rows])),
        "oracle_target_reached_fraction": float(np.mean([r["oracle_stop_reason"] == "target_reached" for r in rows])),
        "safe_candidate_count_abs_diff_mean": float(np.nanmean([
            abs(float(r["estimated_safe_candidate_count"]) - float(r["oracle_safe_candidate_count"]))
            if r["estimated_safe_candidate_count"] is not None and r["oracle_safe_candidate_count"] is not None
            else np.nan
            for r in rows
        ])),
    }


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end offline replay using the live State Estimator -> Transition -> "
            "target selector path on held-out physical states."
        )
    )
    ap.add_argument("--runtime-config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--modes", nargs="+", default=["metadata_only", "action_history_only", "shock_metadata", "shock_action_history"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pulses", nargs="+", type=int, default=[5,10,15,20,25,30,35,40,45])
    ap.add_argument("--max-states-per-pulse", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-root", default=None)
    ap.add_argument("--target-id", nargs="*", default=None)
    ap.add_argument(
        "--exclude-csv",
        default=str(Path(__file__).with_name("artifact_exclusions_20261008.csv")),
        help="Hole-level exclusion CSV. Pass empty string to disable.",
    )
    ap.add_argument("--skip-oracle", action="store_true", help="Skip true-current-state oracle control comparison.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    print("=== End-to-end replay setup ===")
    print(f"Runtime config : {Path(args.runtime_config).resolve()}")
    print(f"Modes          : {', '.join(args.modes)}")
    print(f"Split          : {args.split}")
    print(f"Pulses         : {args.pulses}")
    print(f"Max/pulse      : {args.max_states_per_pulse if args.max_states_per_pulse is not None else 'all'}")
    print()

    with setup_activity("[Setup] load runtime config / registry / targets"):
        cfg = load_runtime_config(Path(args.runtime_config))
        ml_repo = Path(cfg["paths"]["ml_repo"])
        add_ml_repo(ml_repo)
        registry = ActionRegistry(Path(cfg["paths"]["action_registry_csv"]))
        target_root = discover_target_root(cfg, args.target_root)
        targets = load_targets(target_root, args.target_id)
        exclude_path = Path(args.exclude_csv) if args.exclude_csv else None
        excluded_holes = load_excluded_holes(exclude_path)

    print(f"Targets loaded : {len(targets)} from {target_root}")
    print(f"Excluded holes : {len(excluded_holes)}")
    print()

    runtimes: Dict[str, MLRuntime] = {}
    state_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    transition_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    all_maps: List[Dict[Tuple[str, int], Any]] = []
    state_cache: Dict[Tuple[Any, ...], Dict[Tuple[str, int], Dict[str, Any]]] = {}
    transition_cache: Dict[Tuple[Any, ...], Dict[Tuple[str, int], Dict[str, Any]]] = {}

    for mode_i, mode in enumerate(args.modes, start=1):
        prefix = f"[Setup {mode_i}/{len(args.modes)}] {mode}"

        with setup_activity(f"{prefix} | load State Estimator + Transition checkpoints"):
            rt = MLRuntime(ml_repo, mode, cfg["models"][mode], cfg.get("device", "auto"))
        runtimes[mode] = rt

        skey = state_map_cache_key(rt, args.split)
        if skey in state_cache:
            state_maps[mode] = state_cache[skey]
            tqdm.write(
                f"{prefix} | State Estimator sample map: cache hit "
                f"({len(state_maps[mode])} states)"
            )
        else:
            family = str(rt.state_cfg["model"]["family"]).upper()
            with setup_activity(
                f"{prefix} | build {family} State Estimator sample map "
                f"(loads dataset metadata .npy files)"
            ):
                state_maps[mode] = build_state_sample_map(rt, args.split)
            state_cache[skey] = state_maps[mode]
            tqdm.write(f"{prefix} | State Estimator sample map: {len(state_maps[mode])} states")

        tkey = transition_map_cache_key(rt, args.split)
        if tkey in transition_cache:
            transition_maps[mode] = transition_cache[tkey]
            tqdm.write(
                f"{prefix} | Transition row map: cache hit "
                f"({len(transition_maps[mode])} transitions)"
            )
        else:
            with setup_activity(f"{prefix} | build Transition row map"):
                transition_maps[mode] = build_transition_row_map(rt, args.split)
            transition_cache[tkey] = transition_maps[mode]
            tqdm.write(f"{prefix} | Transition row map: {len(transition_maps[mode])} transitions")

        all_maps.extend([state_maps[mode], transition_maps[mode]])
        print()

    with setup_activity("[Setup] intersect common held-out physical states"):
        keys = select_common_keys(
            all_maps,
            args.pulses,
            excluded_holes,
            args.max_states_per_pulse,
            args.seed,
        )

    counts_by_pulse = {
        int(p): sum(1 for _, q in keys if int(q) == int(p))
        for p in args.pulses
    }
    print("=== Setup complete ===")
    print(f"Common states  : {len(keys)}")
    print(f"By pulse       : {counts_by_pulse}")
    print(f"State-map builds actually executed: {len(state_cache)} "
          f"(instead of {len(args.modes)} when compatible maps can be reused)")
    print(f"Transition-map builds actually executed: {len(transition_cache)}")
    print("Starting GPU replay...\n")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["storage"]["session_root"]) / "offline_evaluation" / f"end_to_end_replay_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    threshold = float(cfg["action_selection"]["candidate_threshold"])
    state_rows: List[Dict[str, Any]] = []
    next_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []

    for mode_i, mode in enumerate(args.modes, start=1):
        rt = runtimes[mode]
        smap = state_maps[mode]
        tmap = transition_maps[mode]
        iterator = tqdm(keys, desc=f"[{mode_i}/{len(args.modes)}] {mode}", unit="state", dynamic_ncols=True)

        for hole_uid, pulse in iterator:
            sample = smap[(hole_uid, pulse)]
            tr_row = tmap[(hole_uid, pulse)]
            next_pulse = int(tr_row["next_pulse"])
            control_step = int(pulse // 5)
            request = build_runtime_request(sample, int(cfg["stop_policy"]["max_control_steps"]))
            shock = load_raw_shock_sequence(sample, rt.state_model.uses_shock)

            t0 = time.perf_counter()
            prob_t, z = rt.infer_state(request, shock)
            se_ms = (time.perf_counter() - t0) * 1000.0
            prob = prob_t.detach().cpu().numpy()[0, 0].astype(np.float32)
            est_binary = (prob >= threshold).astype(np.float32)
            true_current, true_next = load_true_masks(tr_row, rt.transition_cfg)

            sm = binary_metrics(prob, true_current, threshold)
            state_rows.append({
                "mode": mode,
                "hole_uid": hole_uid,
                "current_pulse": int(pulse),
                "next_pulse": next_pulse,
                "state_sample_id": str(sample["sample_id"]),
                "transition_sample_id": str(tr_row["sample_id"]),
                "state_estimator_ms": se_ms,
                **{f"state_{k}": v for k, v in sm.items()},
            })

            actions = [CandidateAction(a, next_pulse) for a in registry.actions]
            current_est_t = torch.from_numpy(est_binary)[None, None]
            t1 = time.perf_counter()
            est_preds = rt.predict_candidates(current_est_t, z, actions)
            est_tr_ms = (time.perf_counter() - t1) * 1000.0

            oracle_preds = None
            oracle_tr_ms = None
            if not args.skip_oracle:
                true_t = torch.from_numpy(true_current.astype(np.float32))[None, None]
                t2 = time.perf_counter()
                oracle_preds = rt.predict_candidates(true_t, z, actions)
                oracle_tr_ms = (time.perf_counter() - t2) * 1000.0

            actual_idx = actual_action_index(tr_row, registry)
            em = binary_metrics(est_preds[actual_idx], true_next, threshold)
            nrow = {
                "mode": mode,
                "hole_uid": hole_uid,
                "current_pulse": int(pulse),
                "next_pulse": next_pulse,
                "actual_action_id": registry.actions[actual_idx].action_id,
                "estimated_state_transition_ms": est_tr_ms,
                **{f"end_to_end_next_{k}": v for k, v in em.items()},
            }
            if oracle_preds is not None:
                om = binary_metrics(oracle_preds[actual_idx], true_next, threshold)
                nrow.update({f"oracle_current_next_{k}": v for k, v in om.items()})
                nrow["state_estimation_rmse_penalty"] = float(em["rmse"] - om["rmse"])
                nrow["state_estimation_dice_penalty"] = float(om["dice"] - em["dice"])
                nrow["oracle_transition_ms"] = oracle_tr_ms
            next_rows.append(nrow)

            for target_id, target in targets.items():
                est_dec = evaluate_target_decision(
                    est_binary, est_preds, target, registry, cfg, control_step
                )
                crow: Dict[str, Any] = {
                    "mode": mode,
                    "hole_uid": hole_uid,
                    "current_pulse": int(pulse),
                    "next_pulse": next_pulse,
                    "target_id": target_id,
                    "estimated_decision": est_dec["decision"],
                    "estimated_stop_reason": est_dec["stop_reason"],
                    "estimated_selected_action_id": est_dec["selected_action_id"],
                    "estimated_safe_candidate_count": est_dec["safe_candidate_count"],
                    "estimated_current_overall_fraction": est_dec["current_overall_removal_fraction"],
                    "estimated_current_over_fraction": est_dec["current_over_removal_fraction"],
                    "estimated_top3_safe": "|".join(est_dec["top3_safe"]),
                }
                if oracle_preds is not None:
                    oracle_dec = evaluate_target_decision(
                        true_current, oracle_preds, target, registry, cfg, control_step
                    )
                    both_continue = est_dec["decision"] == "continue" and oracle_dec["decision"] == "continue"
                    crow.update({
                        "oracle_decision": oracle_dec["decision"],
                        "oracle_stop_reason": oracle_dec["stop_reason"],
                        "oracle_selected_action_id": oracle_dec["selected_action_id"],
                        "oracle_safe_candidate_count": oracle_dec["safe_candidate_count"],
                        "oracle_current_overall_fraction": oracle_dec["current_overall_removal_fraction"],
                        "oracle_current_over_fraction": oracle_dec["current_over_removal_fraction"],
                        "oracle_top3_safe": "|".join(oracle_dec["top3_safe"]),
                        "decision_match": est_dec["decision"] == oracle_dec["decision"],
                        "selected_action_match": (
                            est_dec["selected_action_id"] == oracle_dec["selected_action_id"]
                            if both_continue else True
                        ),
                        "top3_set_match": (
                            set(est_dec["top3_safe"]) == set(oracle_dec["top3_safe"])
                            if both_continue else True
                        ),
                    })
                control_rows.append(crow)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    write_csv(state_rows, out_dir / "state_estimator_metrics.csv")
    write_csv(next_rows, out_dir / "actual_action_next_state_metrics.csv")
    write_csv(control_rows, out_dir / "control_policy_replay.csv")

    state_summary: List[Dict[str, Any]] = []
    next_summary: List[Dict[str, Any]] = []
    control_summary: List[Dict[str, Any]] = []
    by_pulse: List[Dict[str, Any]] = []

    for mode in args.modes:
        sr = [r for r in state_rows if r["mode"] == mode]
        state_summary.append({
            "mode": mode,
            **summarize_numeric(sr, [
                "state_rmse", "state_mae", "state_dice", "state_iou",
                "state_binary_disagreement_fraction", "state_processed_area_bias_fraction",
                "state_estimator_ms",
            ]),
        })
        nr = [r for r in next_rows if r["mode"] == mode]
        next_summary.append({
            "mode": mode,
            **summarize_numeric(nr, [
                "end_to_end_next_rmse", "end_to_end_next_dice", "end_to_end_next_iou",
                "oracle_current_next_rmse", "oracle_current_next_dice", "oracle_current_next_iou",
                "state_estimation_rmse_penalty", "state_estimation_dice_penalty",
                "estimated_state_transition_ms", "oracle_transition_ms",
            ]),
        })
        for target_id in targets:
            cr = [r for r in control_rows if r["mode"] == mode and r["target_id"] == target_id]
            row = {"mode": mode, "target_id": target_id}
            if args.skip_oracle:
                row.update({
                    "n": len(cr),
                    "estimated_continue_fraction": float(np.mean([r["estimated_decision"] == "continue" for r in cr])) if cr else None,
                    "estimated_no_safe_fraction": float(np.mean([r["estimated_stop_reason"] == "no_safe_action" for r in cr])) if cr else None,
                })
            else:
                row.update(aggregate_control(cr))
            control_summary.append(row)

        for pulse in args.pulses:
            cr = [r for r in control_rows if r["mode"] == mode and int(r["current_pulse"]) == int(pulse)]
            row = {"mode": mode, "current_pulse": int(pulse)}
            if not args.skip_oracle:
                row.update(aggregate_control(cr))
            else:
                row["n"] = len(cr)
            by_pulse.append(row)

    write_csv(state_summary, out_dir / "summary_state_estimator.csv")
    write_csv(next_summary, out_dir / "summary_actual_action_next_state.csv")
    write_csv(control_summary, out_dir / "summary_control_by_target.csv")
    write_csv(by_pulse, out_dir / "summary_control_by_pulse.csv")

    resolved = {
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "runtime_performance": {m: runtimes[m].performance_settings() for m in args.modes},
        "target_root": str(target_root),
        "target_ids": list(targets.keys()),
        "split": args.split,
        "pulses": list(args.pulses),
        "n_common_states": len(keys),
        "excluded_holes_count": len(excluded_holes),
        "excluded_holes": sorted(excluded_holes),
        "skip_oracle": bool(args.skip_oracle),
        "policy": {
            "candidate_threshold": float(cfg["action_selection"]["candidate_threshold"]),
            "l1_weight": float(cfg["action_selection"]["l1_weight"]),
            "dice_weight": float(cfg["action_selection"]["dice_weight"]),
            "no_safe_action_policy": cfg["action_selection"]["no_safe_action_policy"],
            "overall_removal_target": float(cfg["stop_policy"]["overall_removal_target"]),
            "over_removal_limit": float(cfg["stop_policy"]["over_removal_limit"]),
            "max_control_steps": int(cfg["stop_policy"]["max_control_steps"]),
        },
        "interpretation": [
            "This replay uses the live MLRuntime State Estimator and Transition path on held-out physical states.",
            "The end-to-end next-state metric is evaluated only for the action actually observed in the dataset; unobserved selected actions have no physical counterfactual ground truth.",
            "The oracle-current comparison replaces only the estimated current binary mask with the measured current mask. For Transition B it keeps the same live State-Estimator z_shock, isolating current-shape estimation error rather than replacing the shock latent.",
            "No automatic scientific PASS/FAIL threshold is encoded. Review the state-estimation metrics, actual-action next-state metrics, and estimated-vs-oracle control-policy agreement before real-laser closed-loop operation.",
        ],
    }
    write_json(resolved, out_dir / "resolved_replay.json")

    print("End-to-end offline replay complete.")
    print(f"Modes         : {', '.join(args.modes)}")
    print(f"Targets       : {len(targets)}")
    print(f"Common states : {len(keys)}")
    print(f"Excluded holes: {len(excluded_holes)}")
    print(f"Output        : {out_dir}")
    print(f"Main summaries: {out_dir / 'summary_state_estimator.csv'}")
    print(f"                {out_dir / 'summary_actual_action_next_state.csv'}")
    print(f"                {out_dir / 'summary_control_by_target.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
