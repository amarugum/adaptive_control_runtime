from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.ml_runtime import MLRuntime
from runtime.online_orca import OnlineOrcaProcessor


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    energy_uJ: float
    action_y_um: float
    action_z_um: float
    next_pulse: int


def _json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_calibration_dir(path: Path) -> None:
    required = [
        "grid_definition.csv",
        "orca_grid_calibration.csv",
        "has_side_grid_calibration.csv",
        "orca_processing_origin.csv",
        "crop_config.csv",
        "orca_stamp_time.csv",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Calibration directory is incomplete. Missing: " + ", ".join(missing)
        )


def _frame_stats(shock: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for p in range(shock.shape[0]):
        for c in range(shock.shape[1]):
            x = np.asarray(shock[p, c], dtype=np.float32)
            rows.append(
                {
                    "pulse_index_1based": p + 1,
                    "channel": c,
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "finite": bool(np.isfinite(x).all()),
                }
            )
    return rows


def _run_mode(
    cfg: dict,
    ml_repo: Path,
    registry: ActionRegistry,
    mode: str,
    request: dict,
    shock: np.ndarray,
    output_dir: Path,
) -> dict:
    t0 = time.perf_counter()
    rt = MLRuntime(ml_repo, mode, cfg["models"][mode], cfg.get("device", "auto"))

    with torch.no_grad():
        prob, z_shock = rt.infer_state(request, shock)
    prob_np = prob.detach().float().cpu().numpy()[0, 0]
    threshold = float(cfg["action_selection"]["candidate_threshold"])
    current_binary = prob_np >= threshold

    next_pulse = int(request["step"]["pulse_end"]) + int(request["step"]["pulse_count"])
    actions = [
        CandidateAction(
            a.action_id,
            a.energy_uJ,
            a.action_y_um,
            a.action_z_um,
            next_pulse,
        )
        for a in registry.actions
    ]
    current_t = torch.from_numpy(current_binary.astype(np.float32))[None, None]
    preds = rt.predict_candidates(current_t, z_shock, actions)

    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    np.save(mode_dir / "state_probability.npy", prob_np.astype(np.float16), allow_pickle=False)
    np.save(mode_dir / "state_binary.npy", current_binary.astype(np.uint8), allow_pickle=False)
    np.save(mode_dir / "candidate_probabilities.npy", preds.astype(np.float16), allow_pickle=False)
    np.save(mode_dir / "candidate_binary.npy", (preds >= threshold).astype(np.uint8), allow_pickle=False)

    finite = bool(np.isfinite(prob_np).all() and np.isfinite(preds).all())
    monotonic = bool(
        np.all((preds >= threshold) | (~current_binary[None, :, :]))
    )
    # The model itself enforces next >= current. Check in probability space as a softer diagnostic too.
    prob_monotonic = bool(np.all(preds + 1e-6 >= current_binary[None, :, :].astype(np.float32)))

    result = {
        "mode": mode,
        "device": str(rt.device),
        "state_shape": list(prob_np.shape),
        "state_range": [float(prob_np.min()), float(prob_np.max())],
        "state_mean": float(prob_np.mean()),
        "state_binary_fraction": float(current_binary.mean()),
        "z_shock_shape": None if z_shock is None else list(z_shock.shape),
        "candidate_shape": list(preds.shape),
        "candidate_range": [float(preds.min()), float(preds.max())],
        "finite": finite,
        "binary_monotonic_check": monotonic,
        "probability_monotonic_check": prob_monotonic,
        "state_estimator_checkpoint": str(rt.state_ckpt),
        "transition_checkpoint": str(rt.transition_ckpt),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    _json_dump(mode_dir / "summary.json", result)

    del rt, prob, z_shock
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Smoke-test the real ORCA online path: raw ORCA -> existing postprocessing -> "
            "5x3 shock tensor -> shock State Estimator -> matching Transition -> 27 candidates."
        )
    )
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--orca-dir", required=True, help="Directory containing one step of 15 ORCA TIFF files")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibration-id", help="Folder name under session resources/calibration_sessions")
    group.add_argument("--calibration-dir", help="Explicit calibration directory (diagnostic use)")
    ap.add_argument(
        "--modes",
        default="shock_metadata,shock_action_history",
        help="Comma-separated shock modes. Default: shock_metadata,shock_action_history",
    )
    ap.add_argument("--label", default="real_orca_k01")
    ap.add_argument("--experiment-id", default="smoke_real_orca")
    ap.add_argument("--hole-uid", default="h0001")
    ap.add_argument("--hole-index", type=int, default=1)
    ap.add_argument("--control-step", type=int, default=1)
    ap.add_argument("--energy-uJ", type=float, default=10.0)
    ap.add_argument("--action-y-um", type=float, default=0.0)
    ap.add_argument("--action-z-um", type=float, default=0.0)
    ap.add_argument("--hole-origin-y-um", type=float, default=0.0)
    ap.add_argument("--hole-origin-z-um", type=float, default=0.0)
    ap.add_argument("--actual-y-um", type=float, default=0.0)
    ap.add_argument("--actual-z-um", type=float, default=0.0)
    ap.add_argument("--previous-start-frame-1based", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_runtime_config(cfg_path)
    paths = cfg["paths"]
    orca_dir = Path(args.orca_dir).resolve()
    if not orca_dir.is_dir():
        raise FileNotFoundError(f"ORCA directory not found: {orca_dir}")

    if args.calibration_id:
        calibration_dir = Path(paths["calibration_root"]) / args.calibration_id
        calibration_id = args.calibration_id
    else:
        calibration_dir = Path(args.calibration_dir).resolve()
        calibration_id = calibration_dir.name
    _validate_calibration_dir(calibration_dir)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    allowed = {"shock_metadata", "shock_action_history"}
    unknown = [m for m in modes if m not in allowed]
    if unknown:
        raise ValueError(f"This real-ORCA smoke test only accepts shock modes. Invalid: {unknown}")

    k = int(args.control_step)
    pulse_count = 5
    pulse_start = 5 * (k - 1) + 1
    pulse_end = 5 * k
    request_id = f"{args.experiment_id}_{args.hole_uid}_k{k:02d}_smoke"

    # For this smoke test, history is synthetic when k > 1: it repeats the supplied current action.
    # k01 is therefore the recommended first real-data test.
    history = [
        {
            "control_step": i,
            "action_id": "SMOKE",
            "energy_uJ": float(args.energy_uJ),
            "action_y_um": float(args.action_y_um),
            "action_z_um": float(args.action_z_um),
            "pulse_count": pulse_count,
        }
        for i in range(1, k + 1)
    ]
    request = {
        "protocol_version": "1.0",
        "request_id": request_id,
        "experiment": {
            "experiment_id": args.experiment_id,
            "hole_uid": args.hole_uid,
            "hole_index": int(args.hole_index),
        },
        "step": {
            "completed_control_step": k,
            "next_control_step": k + 1,
            "pulse_start": pulse_start,
            "pulse_end": pulse_end,
            "pulse_count": pulse_count,
        },
        "model": {"mode": modes[0]},
        "target": {"target_id": "SMOKE_ONLY"},
        "executed_action": {
            "action_id": "SMOKE",
            "energy_uJ": float(args.energy_uJ),
            "action_y_um": float(args.action_y_um),
            "action_z_um": float(args.action_z_um),
        },
        "action_history": history,
        "stage": {
            "hole_origin_y_um": float(args.hole_origin_y_um),
            "hole_origin_z_um": float(args.hole_origin_z_um),
            "actual_y_um": float(args.actual_y_um),
            "actual_z_um": float(args.actual_z_um),
            "actual_nd_mm": 0.0,
        },
        "calibration": {"session_calibration_id": calibration_id},
        "frame_selection_context": {
            "previous_used_start_frame_1based": args.previous_start_frame_1based
        },
    }

    session_root = Path(cfg["storage"]["session_root"])
    output_dir = session_root / "diagnostics" / "real_orca_smoke" / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "request.json", request)

    print("=== Real ORCA online-path smoke test ===")
    print(f"config          : {cfg_path}")
    print(f"session_root    : {session_root}")
    print(f"ORCA raw        : {orca_dir}")
    print(f"calibration     : {calibration_dir}")
    print(f"output          : {output_dir}")
    print(f"control_step    : {k}  pulses={pulse_start}-{pulse_end}")
    if k > 1:
        print("WARNING: action history is synthetic in this smoke test; k01 is recommended first.")

    t0 = time.perf_counter()
    processor = OnlineOrcaProcessor(calibration_dir, cfg["orca_online"])
    orca_result = processor.process(orca_dir, request, output_dir / "online_orca")
    shock = np.asarray(orca_result.shock_sequence, dtype=np.float32)
    if shock.shape[0] != 5 or shock.shape[1] != 3:
        raise ValueError(f"Unexpected shock sequence shape: {shock.shape}; expected [5,3,H,W]")
    if not np.isfinite(shock).all():
        raise ValueError("shock_sequence contains NaN/Inf")
    np.save(output_dir / "shock_sequence.npy", shock, allow_pickle=False)

    frame_selection = {
        "auto_start_frame_1based": orca_result.auto_start_frame_1based,
        "used_start_frame_1based": orca_result.used_start_frame_1based,
        "selection_source": orca_result.selection_source,
        "confidence": orca_result.confidence,
        "contrast_score": orca_result.contrast_score,
        "low_confidence": orca_result.low_confidence,
        "low_contrast": orca_result.low_contrast,
    }
    _json_dump(output_dir / "frame_selection.json", frame_selection)
    _json_dump(output_dir / "shock_stats.json", {"shape": list(shock.shape), "frames": _frame_stats(shock)})

    print("\n--- ORCA postprocessing ---")
    print(f"shock_sequence  : {shock.shape}  dtype={shock.dtype}")
    print(f"range           : [{shock.min():.6f}, {shock.max():.6f}]")
    print(f"auto start      : {orca_result.auto_start_frame_1based}")
    print(f"used start      : {orca_result.used_start_frame_1based}")
    print(f"selection source: {orca_result.selection_source}")
    print(f"confidence      : {orca_result.confidence:.6f}")
    print(f"contrast        : {orca_result.contrast_score:.6f}")
    print(f"low confidence  : {orca_result.low_confidence}")
    print(f"low contrast    : {orca_result.low_contrast}")

    registry = ActionRegistry(Path(paths["action_registry_csv"]))
    ml_repo = Path(paths["ml_repo"])
    summaries: list[dict] = []
    failures: list[tuple[str, str]] = []

    for mode in modes:
        print(f"\n--- {mode} ---")
        try:
            request["model"]["mode"] = mode
            result = _run_mode(cfg, ml_repo, registry, mode, request, shock, output_dir)
            summaries.append(result)
            print("PASS")
            print(f"  state      : {tuple(result['state_shape'])}, range={result['state_range']}")
            print(f"  z_shock    : {result['z_shock_shape']}")
            print(f"  candidates : {tuple(result['candidate_shape'])}, range={result['candidate_range']}")
            print(f"  finite     : {result['finite']}")
            print(f"  elapsed    : {result['elapsed_s']:.3f} s")
        except Exception as exc:
            failures.append((mode, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {type(exc).__name__}: {exc}")

    final = {
        "orca": frame_selection,
        "shock_shape": list(shock.shape),
        "modes": summaries,
        "failures": [{"mode": m, "error": e} for m, e in failures],
        "elapsed_s": float(time.perf_counter() - t0),
    }
    _json_dump(output_dir / "summary.json", final)

    print("\n=== Summary ===")
    print(f"passed: {len(summaries)}/{len(modes)}")
    for x in summaries:
        print(f"  PASS {x['mode']}")
    for m, e in failures:
        print(f"  FAIL {m}: {e}")
    print(f"artifacts: {output_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
