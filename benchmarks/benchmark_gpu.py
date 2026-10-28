from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return obj


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def gb(x: int | float) -> float:
    return float(x) / (1024.0 ** 3)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cuda_memory_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "cuda_total_gb": gb(total_b),
        "cuda_free_gb": gb(free_b),
        "cuda_used_visible_gb": gb(total_b - free_b),
        "torch_allocated_gb": gb(torch.cuda.memory_allocated()),
        "torch_reserved_gb": gb(torch.cuda.memory_reserved()),
    }


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
        "end_free_gb": gb(free_b),
        "end_cuda_used_visible_gb": gb(total_b - free_b),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare_probability_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compare two probability tensors on CPU in float32."""
    aa = a.detach().float().cpu()
    bb = b.detach().float().cpu()
    diff = (aa - bb).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rmse_diff": float(torch.sqrt(torch.mean((aa - bb).square())).item()),
        "binary_disagreement_fraction": float(
            ((aa >= threshold) != (bb >= threshold)).float().mean().item()
        ),
    }


def benchmark_callable(
    fn,
    *,
    warmup: int,
    repeats: int,
    desc: str,
) -> Tuple[Dict[str, float], Any]:
    """Benchmark a CUDA callable. Returns timing stats and last output."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    last = None
    for _ in range(max(0, warmup)):
        last = fn()
    cuda_sync()

    reset_peak_memory()
    times_ms: List[float] = []
    for _ in tqdm(range(repeats), desc=desc, unit="run", dynamic_ncols=True):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    stats = {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "p95_ms": percentile(times_ms, 95),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }
    stats.update(peak_memory_snapshot())
    return stats, last


def add_ml_repo_to_path(ml_repo: Path) -> None:
    p = str(ml_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Runtime config helpers
# ---------------------------------------------------------------------------

def get_path(cfg: Dict[str, Any], *keys: str, default=None):
    x: Any = cfg
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    estimator_run: Path
    transition_run: Path


def resolve_modes(runtime_cfg: Dict[str, Any], requested: Sequence[str]) -> List[ModeSpec]:
    models = runtime_cfg.get("models", {})
    if not isinstance(models, dict):
        raise KeyError("runtime config has no 'models' mapping")

    out: List[ModeSpec] = []
    for mode in requested:
        if mode not in models:
            raise KeyError(f"Mode {mode!r} not found in runtime config models.")
        m = models[mode]
        est = m.get("estimator_run") or m.get("state_estimator_run")
        tr = m.get("transition_run")
        if not est or not tr:
            raise KeyError(
                f"Mode {mode!r} must define estimator_run/state_estimator_run and transition_run."
            )
        out.append(ModeSpec(mode, Path(est), Path(tr)))
    return out


# ---------------------------------------------------------------------------
# State Estimator loading and sample preparation
# ---------------------------------------------------------------------------

def load_estimator_bundle(
    run_dir: Path,
    ml_repo: Path,
    device: torch.device,
    preferred_pulse: int,
):
    add_ml_repo_to_path(ml_repo)

    from state_estimator.dataset import (
        LaserStateDataset,
        build_g1_samples,
        build_g5_samples,
        discover_pulses,
        infer_image_hw,
        load_dataset_split,
    )
    from state_estimator.models import StateEstimatorG

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    d = cfg["data"]
    m = cfg["model"]

    records = discover_pulses(
        Path(d["root"]),
        d["shock_filename"],
        d["target_filename"],
        d["metadata_filename"],
    )
    holes = sorted({r.hole_uid for r in records})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)
    test_holes = splits["test"]

    common = dict(
        block_size=int(d["block_size"]),
        history_max_steps=int(d.get("history_max_steps", 10)),
    )
    if str(m["family"]).upper() == "G1":
        samples = build_g1_samples(
            records,
            test_holes,
            sampling=d.get("g1_sampling", "block_end"),
            **common,
        )
    else:
        samples = build_g5_samples(
            records,
            test_holes,
            require_same_action=bool(d["require_same_action_within_g5_block"]),
            **common,
        )
    if not samples:
        raise RuntimeError(f"No estimator test samples for {run_dir}")

    dataset = LaserStateDataset(
        samples=samples,
        family=m["family"],
        input_mode=m["input_mode"],
        pristine_channel=int(d["pristine_channel"]),
        metadata_norm_cfg=d["metadata_normalization"],
        image_size=int(d["image_size"]),
        shock_clip_min=float(d["shock_normalization"]["clip_min"]),
        shock_clip_max=float(d["shock_normalization"]["clip_max"]),
        normalize_shock_to_unit=bool(d["shock_normalization"]["to_unit_interval"]),
        block_size=int(d["block_size"]),
    )
    image_hw = infer_image_hw(
        samples,
        int(d["pristine_channel"]),
        image_size=int(d["image_size"]),
    )

    model = StateEstimatorG(
        family=m["family"],
        input_mode=m["input_mode"],
        image_hw=image_hw,
        encoder_channels=tuple(m["encoder_channels"]),
        meta_hidden_dims=tuple(m["meta_hidden_dims"]),
        meta_dim=int(m["meta_dim"]),
        history_hidden_dim=int(m.get("history_hidden_dim", 128)),
        history_num_layers=int(m.get("history_num_layers", 1)),
        history_dropout=float(m.get("history_dropout", 0.0)),
        convgru_hidden_channels=int(m["convgru_hidden_channels"]),
        convgru_kernel_size=int(m["convgru_kernel_size"]),
        decoder_channels=tuple(m["decoder_channels"]),
        group_norm_groups=int(m["group_norm_groups"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    # Pick a representative runtime-like B=1 test item nearest the preferred pulse.
    def sample_pulse(s) -> int:
        for attr in ("pulse_number", "pulse_end", "current_pulse"):
            if hasattr(s, attr):
                return int(getattr(s, attr))
        return preferred_pulse

    idx = min(
        range(len(samples)),
        key=lambda i: abs(sample_pulse(samples[i]) - preferred_pulse),
    )
    item = dataset[idx]

    shock = item["shock"].unsqueeze(0).to(device)
    shock_arg = None if shock.numel() == 0 else shock
    metadata = item["metadata"].unsqueeze(0).to(device)
    history = item["action_history"].unsqueeze(0).to(device)
    history_length = item["history_length"].reshape(1).to(device)

    sample_id = str(item.get("sample_id", f"index_{idx}"))
    return model, cfg, {
        "shock": shock_arg,
        "metadata": metadata,
        "history": history,
        "history_length": history_length,
        "sample_id": sample_id,
    }


def benchmark_estimator(
    *,
    mode_name: str,
    model,
    batch: Dict[str, Any],
    amp_options: Sequence[bool],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    outputs: Dict[bool, torch.Tensor] = {}

    for amp in amp_options:
        def run():
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(amp and device.type == "cuda"),
                ):
                    return model(
                        batch["shock"],
                        batch["metadata"],
                        batch["history"],
                        batch["history_length"],
                    )

        try:
            stats, output = benchmark_callable(
                run,
                warmup=warmup,
                repeats=repeats,
                desc=f"{mode_name} estimator AMP={amp}",
            )
            row = {
                "component": "estimator",
                "mode": mode_name,
                "amp": bool(amp),
                "sample_id": batch["sample_id"],
                "status": "PASS",
                **stats,
            }
            if torch.is_tensor(output):
                row["output_shape"] = str(tuple(output.shape))
                # Compare probabilities, because runtime thresholds sigmoid output.
                outputs[bool(amp)] = torch.sigmoid(output.detach()).float().cpu()
            rows.append(row)
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda()
            rows.append({
                "component": "estimator",
                "mode": mode_name,
                "amp": bool(amp),
                "sample_id": batch["sample_id"],
                "status": "OOM",
                "error": str(exc),
            })

    consistency: Dict[str, Any] = {}
    if False in outputs and True in outputs:
        cmp = compare_probability_tensors(outputs[True], outputs[False])
        consistency = {
            "reference": "AMP=False (FP32)",
            "comparison": "AMP=True vs AMP=False",
            **cmp,
        }
        for row in rows:
            if row.get("status") == "PASS" and bool(row.get("amp")):
                for k, v in cmp.items():
                    row[f"vs_fp32_{k}"] = v
    return rows, consistency


# ---------------------------------------------------------------------------
# Transition loading and representative sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    action_id: str
    energy_uJ: float
    y_um: float
    z_um: float


def load_actions(runtime_cfg: Dict[str, Any]) -> List[Action]:
    p = get_path(runtime_cfg, "paths", "action_registry_csv")
    if not p:
        # Strictly preserve the known 27-action grid if no registry path exists.
        out = []
        idx = 1
        for y in (-3.0, 0.0, 3.0):
            for z in (0.0, -5.0, -10.0):
                for e in (10.0, 15.0, 20.0):
                    out.append(Action(f"A{idx:02d}", e, y, z))
                    idx += 1
        return out

    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"action_registry_csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(Action(
            str(r["action_id"]),
            float(r["energy_uJ"]),
            float(r["action_y_um"]),
            float(r["action_z_um"]),
        ))
    if len(out) != 27:
        raise ValueError(f"Expected 27 actions, got {len(out)} from {path}")
    return out


def load_transition_bundle(
    run_dir: Path,
    ml_repo: Path,
    device: torch.device,
    preferred_current_pulse: int,
):
    add_ml_repo_to_path(ml_repo)

    from transition.dataset import (
        load_dataset_split,
        load_manifest,
        load_z_shock_npz,
    )
    from transition.models import TransitionUNet

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    d = cfg["data"]
    m = cfg["model"]

    rows = load_manifest(Path(d["root"]) / d["manifest_name"])
    holes = sorted({str(r["hole_uid"]) for r in rows})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)
    test_holes = set(splits["test"])
    rows = [r for r in rows if str(r["hole_uid"]) in test_holes]
    if not rows:
        raise RuntimeError(f"No transition test rows for {run_dir}")

    row = min(
        rows,
        key=lambda r: abs(int(r["current_pulse"]) - preferred_current_pulse),
    )

    variant = str(m["variant"]).upper()
    z_lookup = None
    shock_dim = None
    if variant == "B":
        z_lookup = load_z_shock_npz(Path(d["z_shock_npz"]))
        z = np.asarray(z_lookup[str(row["current_g5_sample_id"])], dtype=np.float32)
        shock_dim = int(z.shape[0])
    else:
        z = None

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

    # Load current binary mask exactly from the transition manifest.
    current_path = Path(d["root"]) / row["current_target_path"]
    arr = np.asarray(np.load(current_path), dtype=np.float32)
    pristine_channel = int(d["pristine_channel"])
    threshold = float(d["mask_threshold"])
    mask = (arr[pristine_channel] >= threshold).astype(np.float32)

    image_size = int(d.get("image_size", 256))
    if mask.shape != (image_size, image_size):
        t = torch.from_numpy(mask)[None, None]
        mask = (
            F.interpolate(t, size=(image_size, image_size), mode="nearest")[0, 0]
            .numpy()
            .astype(np.float32)
        )

    current = torch.from_numpy(mask)[None, None].to(device)
    z_t = None if z is None else torch.from_numpy(z)[None].to(device)

    return model, cfg, row, current, z_t


def normalize_action_grid(
    actions: Sequence[Action],
    next_pulse: int,
    norm_cfg: Dict[str, Any],
) -> np.ndarray:
    from transition.dataset import normalize_metadata

    out = []
    for a in actions:
        raw = np.asarray(
            [a.energy_uJ, a.y_um, a.z_um, float(next_pulse)],
            dtype=np.float32,
        )
        out.append(normalize_metadata(raw, norm_cfg))
    return np.stack(out, axis=0).astype(np.float32)


def collect_transition_predictions(
    *,
    model,
    current: torch.Tensor,
    z_base: Optional[torch.Tensor],
    actions_t: torch.Tensor,
    chunk: int,
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return all 27 next-state probabilities as [27,1,H,W] on CPU."""
    n_actions = int(actions_t.shape[0])
    preds: List[torch.Tensor] = []
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            enabled=bool(amp and device.type == "cuda"),
        ):
            encoded = model.encode_state(current)
            for j0 in range(0, n_actions, chunk):
                j1 = min(j0 + chunk, n_actions)
                c = j1 - j0
                z_chunk = None if z_base is None else z_base.expand(c, -1)
                out = model.decode_from_encoded(
                    current_mask=current,
                    encoded_state=encoded,
                    action=actions_t[j0:j1],
                    z_shock=z_chunk,
                    repeats=c,
                )
                preds.append(out["next_pred"].detach().float().cpu())
    return torch.cat(preds, dim=0)


def benchmark_transition_chunks(
    *,
    mode_name: str,
    model,
    cfg: Dict[str, Any],
    row: Dict[str, Any],
    current: torch.Tensor,
    z_base: Optional[torch.Tensor],
    actions: Sequence[Action],
    chunk_sizes: Sequence[int],
    amp_options: Sequence[bool],
    warmup: int,
    repeats: int,
    device: torch.device,
    memory_margin_gb: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    norm_cfg = cfg["data"]["metadata_normalization"]
    action_np = normalize_action_grid(
        actions,
        int(row["next_pulse"]),
        norm_cfg,
    )
    actions_t = torch.from_numpy(action_np).to(device)
    n_actions = len(actions)

    rows: List[Dict[str, Any]] = []
    outputs: Dict[Tuple[bool, int], torch.Tensor] = {}
    total_gb = gb(torch.cuda.get_device_properties(device).total_memory)
    normalized_chunks = sorted({min(max(1, int(c)), n_actions) for c in chunk_sizes})

    for amp in amp_options:
        for chunk in normalized_chunks:
            def one_decision():
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=device.type,
                        enabled=bool(amp and device.type == "cuda"),
                    ):
                        encoded = model.encode_state(current)
                        last = None
                        for j0 in range(0, n_actions, chunk):
                            j1 = min(j0 + chunk, n_actions)
                            c = j1 - j0
                            action_chunk = actions_t[j0:j1]
                            z_chunk = None
                            if z_base is not None:
                                z_chunk = z_base.expand(c, -1)
                            out = model.decode_from_encoded(
                                current_mask=current,
                                encoded_state=encoded,
                                action=action_chunk,
                                z_shock=z_chunk,
                                repeats=c,
                            )
                            last = out["next_pred"].sum()
                        return last

            try:
                stats, _ = benchmark_callable(
                    one_decision,
                    warmup=warmup,
                    repeats=repeats,
                    desc=f"{mode_name} transition AMP={amp} chunk={chunk}",
                )
                safe_by_margin = (
                    float(stats["peak_reserved_gb"])
                    <= total_gb - float(memory_margin_gb)
                )
                rows.append({
                    "component": "transition_27_action_sweep",
                    "mode": mode_name,
                    "variant": str(model.variant),
                    "candidate_chunk_size": chunk,
                    "amp": bool(amp),
                    "current_pulse": int(row["current_pulse"]),
                    "next_pulse": int(row["next_pulse"]),
                    "status": "PASS",
                    "safe_by_cuda_margin": int(safe_by_margin),
                    "requested_memory_margin_gb": float(memory_margin_gb),
                    "gpu_total_gb": total_gb,
                    **stats,
                })
                outputs[(bool(amp), chunk)] = collect_transition_predictions(
                    model=model,
                    current=current,
                    z_base=z_base,
                    actions_t=actions_t,
                    chunk=chunk,
                    amp=bool(amp),
                    device=device,
                )
            except torch.cuda.OutOfMemoryError as exc:
                clear_cuda()
                rows.append({
                    "component": "transition_27_action_sweep",
                    "mode": mode_name,
                    "variant": str(model.variant),
                    "candidate_chunk_size": chunk,
                    "amp": bool(amp),
                    "current_pulse": int(row["current_pulse"]),
                    "next_pulse": int(row["next_pulse"]),
                    "status": "OOM",
                    "safe_by_cuda_margin": 0,
                    "requested_memory_margin_gb": float(memory_margin_gb),
                    "gpu_total_gb": total_gb,
                    "error": str(exc),
                })

    # Numerical consistency: AMP vs FP32 at the same chunk.
    for row_out in rows:
        if row_out.get("status") != "PASS":
            continue
        amp = bool(row_out["amp"])
        chunk = int(row_out["candidate_chunk_size"])
        if amp and (True, chunk) in outputs and (False, chunk) in outputs:
            cmp = compare_probability_tensors(
                outputs[(True, chunk)],
                outputs[(False, chunk)],
            )
            for k, v in cmp.items():
                row_out[f"vs_fp32_same_chunk_{k}"] = v

    # Chunk consistency under FP32. Smallest FP32 chunk is the reference.
    fp32_chunks = [c for c in normalized_chunks if (False, c) in outputs]
    if fp32_chunks:
        ref_chunk = min(fp32_chunks)
        ref = outputs[(False, ref_chunk)]
        for row_out in rows:
            if row_out.get("status") != "PASS" or bool(row_out.get("amp")):
                continue
            chunk = int(row_out["candidate_chunk_size"])
            cmp = compare_probability_tensors(outputs[(False, chunk)], ref)
            for k, v in cmp.items():
                row_out[f"vs_fp32_chunk{ref_chunk}_{k}"] = v

    # Performance recommendation:
    # 1) only configurations with requested dedicated-memory margin,
    # 2) fastest median latency,
    # 3) if FP32 is within 3% of fastest, prefer FP32 for numerical conservatism.
    candidates = [
        r for r in rows
        if r.get("status") == "PASS" and int(r.get("safe_by_cuda_margin", 0)) == 1
    ]
    if not candidates:
        candidates = [r for r in rows if r.get("status") == "PASS"]

    recommendation: Dict[str, Any] = {}
    if candidates:
        fastest = min(candidates, key=lambda r: float(r["median_ms"]))
        fp32 = [r for r in candidates if not bool(r.get("amp"))]
        chosen = fastest
        if fp32:
            best_fp32 = min(fp32, key=lambda r: float(r["median_ms"]))
            if float(best_fp32["median_ms"]) <= float(fastest["median_ms"]) * 1.03:
                chosen = best_fp32
        recommendation = {
            "transition_inference_amp": bool(chosen["amp"]),
            "candidate_chunk_size": int(chosen["candidate_chunk_size"]),
            "median_ms": float(chosen["median_ms"]),
            "p95_ms": float(chosen["p95_ms"]),
            "peak_reserved_gb": float(chosen["peak_reserved_gb"]),
            "fastest_raw_amp": bool(fastest["amp"]),
            "fastest_raw_chunk_size": int(fastest["candidate_chunk_size"]),
            "fastest_raw_median_ms": float(fastest["median_ms"]),
            "selection_rule": (
                "fastest safe setting; prefer FP32 when its median latency is within 3% "
                "of the raw fastest setting"
            ),
        }
        if bool(chosen["amp"]):
            k = "vs_fp32_same_chunk_binary_disagreement_fraction"
            recommendation["amp_vs_fp32_binary_disagreement_fraction"] = chosen.get(k)
            recommendation["amp_vs_fp32_max_abs_diff"] = chosen.get(
                "vs_fp32_same_chunk_max_abs_diff"
            )

    return rows, recommendation


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend_estimator_amp(rows: Sequence[Dict[str, Any]], mode: str) -> Optional[bool]:
    cand = [
        r for r in rows
        if r.get("component") == "estimator"
        and r.get("mode") == mode
        and r.get("status") == "PASS"
    ]
    if not cand:
        return None
    cand = sorted(cand, key=lambda r: float(r["median_ms"]))
    fastest = cand[0]
    fp32 = [r for r in cand if not bool(r.get("amp"))]
    if fp32:
        best_fp32 = min(fp32, key=lambda r: float(r["median_ms"]))
        if float(best_fp32["median_ms"]) <= float(fastest["median_ms"]) * 1.03:
            return False
    return bool(fastest["amp"])


def selected_estimator_row(
    rows: Sequence[Dict[str, Any]], mode: str, amp: Optional[bool]
) -> Optional[Dict[str, Any]]:
    matches = [
        r for r in rows
        if r.get("component") == "estimator"
        and r.get("mode") == mode
        and r.get("status") == "PASS"
        and bool(r.get("amp")) == bool(amp)
    ]
    return min(matches, key=lambda r: float(r["median_ms"])) if matches else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark runtime-like B=1 State Estimator latency/VRAM and "
            "27-action Transition AMP x candidate-chunk combinations."
        )
    )
    ap.add_argument(
        "--runtime-config",
        default="adaptive_runtime_config.yaml",
        help="Current PC2 adaptive runtime YAML.",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["metadata_only", "action_history_only", "shock_metadata", "shock_action_history"],
    )
    ap.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12, 18, 27],
    )
    ap.add_argument("--preferred-pulse", type=int, default=25)
    ap.add_argument("--estimator-warmup", type=int, default=5)
    ap.add_argument("--estimator-repeats", type=int, default=30)
    ap.add_argument("--transition-warmup", type=int, default=3)
    ap.add_argument("--transition-repeats", type=int, default=20)
    ap.add_argument(
        "--estimator-amp-options",
        nargs="+",
        default=["false"],
        choices=["true", "false"],
        help="Estimator precision modes to benchmark. V1.10 defaults to FP32 only.",
    )
    ap.add_argument(
        "--transition-amp-options",
        nargs="+",
        default=["false"],
        choices=["true", "false"],
        help="Transition precision modes to benchmark. V1.10 defaults to FP32 only because live AMP was rejected by the action-equivalence gate.",
    )
    ap.add_argument(
        "--memory-margin-gb",
        type=float,
        default=0.75,
        help=(
            "Required CUDA dedicated-memory headroom for auto recommendation. "
            "Windows Shared GPU Memory is not exposed by PyTorch and must be "
            "checked in Task Manager for the recommended chunk."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Default: benchmarks/results/runtime_gpu_benchmark_YYYYMMDD_HHMMSS",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")
    device = torch.device("cuda")

    runtime_path = Path(args.runtime_config).resolve()
    runtime_cfg = load_yaml(runtime_path)
    ml_repo_raw = get_path(runtime_cfg, "paths", "ml_repo")
    if not ml_repo_raw:
        raise KeyError("adaptive_runtime_config.yaml: paths.ml_repo is required")
    ml_repo = Path(ml_repo_raw)
    if not ml_repo.exists():
        raise FileNotFoundError(ml_repo)

    modes = resolve_modes(runtime_cfg, args.modes)
    actions = load_actions(runtime_cfg)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("benchmarks") / "results" / f"runtime_gpu_benchmark_{stamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    gpu = torch.cuda.get_device_properties(device)
    system_report = {
        "runtime_config": str(runtime_path),
        "ml_repo": str(ml_repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu.name,
        "benchmark_version": "v3_four_model_fp32",
        "gpu_total_memory_gb": gb(gpu.total_memory),
        "initial_memory": cuda_memory_snapshot(),
        "important_note": (
            "PyTorch reports CUDA/dedicated-memory usage, not Windows WDDM "
            "Shared GPU Memory. Confirm Shared GPU Memory manually in Task Manager "
            "for the recommended runtime setting."
        ),
    }
    write_json(system_report, out_dir / "system.json")

    all_rows: List[Dict[str, Any]] = []
    recommendations: Dict[str, Any] = {}

    amp_options = [x == "true" for x in args.estimator_amp_options]
    transition_amp_options = [x == "true" for x in args.transition_amp_options]

    print(f"GPU: {gpu.name} ({gb(gpu.total_memory):.2f} GB)")
    print(f"Benchmark output: {out_dir}")
    print()

    for mi, spec in enumerate(modes, start=1):
        print(f"=== [{mi}/{len(modes)}] {spec.mode} ===")
        clear_cuda()

        # Estimator stays resident while the matching Transition is loaded.
        # This is closer to runtime than benchmarking the Transition in isolation.
        est_model, est_cfg, est_batch = load_estimator_bundle(
            spec.estimator_run,
            ml_repo,
            device,
            preferred_pulse=args.preferred_pulse,
        )
        print("Estimator loaded:", cuda_memory_snapshot())

        est_rows, est_consistency = benchmark_estimator(
            mode_name=spec.mode,
            model=est_model,
            batch=est_batch,
            amp_options=amp_options,
            warmup=args.estimator_warmup,
            repeats=args.estimator_repeats,
            device=device,
        )
        all_rows.extend(est_rows)
        recommended_amp = recommend_estimator_amp(est_rows, spec.mode)

        # Keep estimator loaded. Load the matching transition as well.
        tr_model, tr_cfg, tr_row, current, z_base = load_transition_bundle(
            spec.transition_run,
            ml_repo,
            device,
            preferred_current_pulse=args.preferred_pulse,
        )
        print("Estimator + Transition resident:", cuda_memory_snapshot())

        tr_rows, transition_rec = benchmark_transition_chunks(
            mode_name=spec.mode,
            model=tr_model,
            cfg=tr_cfg,
            row=tr_row,
            current=current,
            z_base=z_base,
            actions=actions,
            chunk_sizes=args.chunk_sizes,
            amp_options=transition_amp_options,
            warmup=args.transition_warmup,
            repeats=args.transition_repeats,
            device=device,
            memory_margin_gb=args.memory_margin_gb,
        )
        all_rows.extend(tr_rows)

        est_selected = selected_estimator_row(est_rows, spec.mode, recommended_amp)
        end_to_end_median_ms = None
        if est_selected and transition_rec:
            end_to_end_median_ms = (
                float(est_selected["median_ms"]) + float(transition_rec["median_ms"])
            )

        recommendations[spec.mode] = {
            "estimator_inference_amp": recommended_amp,
            "transition_inference_amp": transition_rec.get("transition_inference_amp"),
            "candidate_chunk_size": transition_rec.get("candidate_chunk_size"),
            "estimator_median_ms": (
                float(est_selected["median_ms"]) if est_selected else None
            ),
            "transition_27_action_median_ms": transition_rec.get("median_ms"),
            "estimated_ml_pipeline_median_ms": end_to_end_median_ms,
            "estimator_amp_consistency": est_consistency,
            "transition_recommendation_detail": transition_rec,
            "memory_margin_gb": float(args.memory_margin_gb),
            "estimator_run": str(spec.estimator_run),
            "transition_run": str(spec.transition_run),
            "manual_check_required": (
                "Confirm Windows Task Manager Shared GPU Memory stays near zero "
                "while repeatedly using the recommended FP32 setting."
            ),
        }

        del tr_model, current, z_base
        del est_model
        clear_cuda()
        print()

    write_csv(all_rows, out_dir / "benchmark_results.csv")
    write_json(recommendations, out_dir / "recommendations.json")

    # Emit a human-copyable YAML snippet.  Do not modify runtime config automatically.
    snippet = {
        "runtime_performance_recommendation": recommendations,
    }
    (out_dir / "recommended_runtime_performance.yaml").write_text(
        yaml.safe_dump(snippet, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("=== Recommended settings ===")
    for mode, rec in recommendations.items():
        print(
            f"{mode}: estimator_amp={rec['estimator_inference_amp']}, "
            f"transition_amp={rec['transition_inference_amp']}, "
            f"candidate_chunk_size={rec['candidate_chunk_size']}, "
            f"estimated_pipeline={rec['estimated_ml_pipeline_median_ms']:.3f} ms"
            if rec.get("estimated_ml_pipeline_median_ms") is not None
            else (
                f"{mode}: estimator_amp={rec['estimator_inference_amp']}, "
                f"transition_amp={rec['transition_inference_amp']}, "
                f"candidate_chunk_size={rec['candidate_chunk_size']}"
            )
        )
    print()
    print("IMPORTANT:")
    print(
        "The automatic recommendation uses CUDA/dedicated-memory statistics only. "
        "Windows Shared GPU Memory cannot be read reliably from PyTorch. "
        "After choosing a chunk, verify Task Manager Shared GPU Memory manually."
    )
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
