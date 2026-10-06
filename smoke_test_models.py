from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config
from runtime.ml_runtime import MLRuntime


ALL_MODES = [
    "metadata_only",
    "action_history_only",
    "shock_metadata",
    "shock_action_history",
]
SHOCK_MODES = {"shock_metadata", "shock_action_history"}


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_request(mode: str) -> dict:
    # One completed seed step: 10 uJ, y=0, z=0, pulses 1-5.
    return {
        "request_id": f"smoke_test_{mode}_k01",
        "model": {"mode": mode},
        "step": {
            "completed_control_step": 1,
            "pulse_start": 1,
            "pulse_end": 5,
            "pulse_count": 5,
            "next_control_step": 2,
        },
        "executed_action": {
            "action_id": "SMOKE_SEED",
            "energy_uJ": 10.0,
            "action_y_um": 0.0,
            "action_z_um": 0.0,
        },
        "action_history": [
            {
                "control_step": 1,
                "action_id": "SMOKE_SEED",
                "energy_uJ": 10.0,
                "action_y_um": 0.0,
                "action_z_um": 0.0,
                "pulse_count": 5,
            }
        ],
    }


def _make_dummy_shock() -> np.ndarray:
    # Deterministic, non-constant values in the training preprocessing range [0, 1.5].
    rng = np.random.default_rng(20261006)
    return rng.uniform(0.0, 1.5, size=(5, 3, 256, 256)).astype(np.float32)


def _candidate_actions(registry: ActionRegistry, next_pulse: int):
    return [
        SimpleNamespace(
            action_id=a.action_id,
            energy_uJ=a.energy_uJ,
            action_y_um=a.action_y_um,
            action_z_um=a.action_z_um,
            next_pulse=next_pulse,
        )
        for a in registry.actions
    ]


def _assert_finite(name: str, x) -> None:
    if isinstance(x, torch.Tensor):
        ok = bool(torch.isfinite(x).all().item())
    else:
        ok = bool(np.isfinite(np.asarray(x)).all())
    if not ok:
        raise RuntimeError(f"{name} contains NaN or Inf")


def run_one_mode(cfg: dict, registry: ActionRegistry, mode: str, device_override: str | None) -> dict:
    if mode not in cfg.get("models", {}):
        raise KeyError(f"Mode is not configured in YAML: {mode}")

    model_cfg = dict(cfg["models"][mode])
    se_ckpt = Path(model_cfg["state_estimator_run"]) / "checkpoints" / "best.pt"
    tr_ckpt = Path(model_cfg["transition_run"]) / "checkpoints" / "best.pt"
    if not se_ckpt.exists():
        raise FileNotFoundError(f"State Estimator checkpoint not found: {se_ckpt}")
    if not tr_ckpt.exists():
        raise FileNotFoundError(f"Transition checkpoint not found: {tr_ckpt}")

    device_name = device_override or cfg.get("device", "auto")
    runtime = MLRuntime(
        ml_repo=Path(cfg["paths"]["ml_repo"]),
        mode=mode,
        model_cfg=model_cfg,
        device_name=device_name,
    )

    request = _make_request(mode)
    shock = _make_dummy_shock() if mode in SHOCK_MODES else None

    with torch.no_grad():
        state_prob, z_shock = runtime.infer_state(request, shock)

    if tuple(state_prob.shape) != (1, 1, 256, 256):
        raise RuntimeError(f"Unexpected state probability shape: {tuple(state_prob.shape)}")
    _assert_finite("state_probability", state_prob)

    if mode in SHOCK_MODES:
        if z_shock is None:
            raise RuntimeError("Shock mode returned z_shock=None")
        if z_shock.ndim != 2 or z_shock.shape[0] != 1:
            raise RuntimeError(f"Unexpected z_shock shape: {tuple(z_shock.shape)}")
        _assert_finite("z_shock", z_shock)
    elif z_shock is not None:
        raise RuntimeError(f"Non-shock mode unexpectedly returned z_shock: {tuple(z_shock.shape)}")

    current_binary = (state_prob >= 0.5).float().cpu()
    actions = _candidate_actions(registry, next_pulse=10)
    preds = runtime.predict_candidates(current_binary, z_shock, actions)

    if tuple(preds.shape) != (27, 256, 256):
        raise RuntimeError(f"Unexpected candidate prediction shape: {tuple(preds.shape)}")
    _assert_finite("candidate_predictions", preds)

    current_np = current_binary.numpy()[0, 0]
    if np.any(preds + 1e-6 < current_np[None, :, :]):
        raise RuntimeError("Transition physics constraint violated: S_next < S_current")
    if float(preds.min()) < -1e-6 or float(preds.max()) > 1.0 + 1e-6:
        raise RuntimeError(
            f"Candidate predictions outside [0,1]: min={preds.min():.6f}, max={preds.max():.6f}"
        )

    result = {
        "mode": mode,
        "device": str(runtime.device),
        "state_input_mode": runtime.state_cfg["model"]["input_mode"],
        "transition_variant": str(runtime.transition_cfg["model"]["variant"]).upper(),
        "state_params": _param_count(runtime.state_model),
        "transition_params": _param_count(runtime.transition_model),
        "state_shape": tuple(state_prob.shape),
        "z_shock_shape": None if z_shock is None else tuple(z_shock.shape),
        "candidate_shape": tuple(preds.shape),
        "candidate_min": float(preds.min()),
        "candidate_max": float(preds.max()),
        "state_checkpoint": str(runtime.state_ckpt),
        "transition_checkpoint": str(runtime.transition_ckpt),
    }

    del runtime, state_prob, z_shock, current_binary, preds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and dummy-forward the four adaptive-control ML pipelines."
    )
    parser.add_argument(
        "--config",
        default="adaptive_runtime_config.yaml",
        help="Path to adaptive runtime YAML.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=ALL_MODES,
        default=ALL_MODES,
        help="Subset of modes to test. Default: all four.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda or cpu. Default: YAML setting.",
    )
    args = parser.parse_args()

    cfg = load_runtime_config(Path(args.config))
    registry_path = Path(cfg["paths"]["action_registry_csv"])
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Action registry not found: {registry_path}\n"
            "Run init_storage.py first for this session root."
        )
    registry = ActionRegistry(registry_path)

    print("=== Adaptive-control ML smoke test ===")
    print(f"config        : {Path(args.config).resolve()}")
    print(f"session_root  : {cfg['storage']['session_root']}")
    print(f"ml_repo       : {cfg['paths']['ml_repo']}")
    print(f"action_registry: {registry_path}")
    print(f"torch         : {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_runtime  : {torch.version.cuda}")
        print(f"gpu           : {torch.cuda.get_device_name(0)}")
    print()

    passed = []
    failed = []
    for mode in args.modes:
        print(f"--- {mode} ---")
        try:
            r = run_one_mode(cfg, registry, mode, args.device)
            print(f"PASS  device={r['device']}")
            print(
                f"      SE={r['state_input_mode']} ({r['state_params']:,} params), "
                f"Transition={r['transition_variant']} ({r['transition_params']:,} params)"
            )
            print(f"      state={r['state_shape']}  z_shock={r['z_shock_shape']}")
            print(
                f"      candidates={r['candidate_shape']}  "
                f"range=[{r['candidate_min']:.6f}, {r['candidate_max']:.6f}]"
            )
            print(f"      SE checkpoint: {r['state_checkpoint']}")
            print(f"      TR checkpoint: {r['transition_checkpoint']}")
            passed.append(mode)
        except Exception as exc:
            print(f"FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(mode)
        print()

    print("=== Summary ===")
    print(f"passed: {len(passed)}/{len(args.modes)}")
    for mode in passed:
        print(f"  PASS {mode}")
    for mode in failed:
        print(f"  FAIL {mode}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
