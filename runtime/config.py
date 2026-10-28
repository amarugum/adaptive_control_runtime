from __future__ import annotations

from pathlib import Path, PureWindowsPath
import re
import math
import yaml


def load_yaml(path: Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be mapping: {path}")
    return obj


def _looks_windows_absolute(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _resolve_user_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve paths relative to the config file, while preserving Windows absolute paths."""
    text = str(value)
    if _looks_windows_absolute(text):
        # On the actual Windows PC this is an absolute Path. Keeping the literal drive
        # form also makes the generated configuration readable when inspected elsewhere.
        return Path(PureWindowsPath(text)) if Path(text).is_absolute() else Path(text)
    p = Path(text).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_runtime_config(path: Path) -> dict:
    """Load the PC2 runtime config and derive persistent vs temporary storage paths.

    ``storage.session_root`` is the date/session-scoped persistent archive root, normally
    on the external SSD. ``storage.transfer_root`` is the stable PC2-internal temporary
    transfer root. Only ``transfer_root / inbox`` is intended to be SMB-shared.
    """
    config_path = Path(path).resolve()
    cfg = load_yaml(config_path)
    base_dir = config_path.parent

    storage = cfg.get("storage") or {}
    session_root_raw = storage.get("session_root")
    if not session_root_raw:
        raise KeyError(
            "Missing storage.session_root in adaptive_runtime_config.yaml. "
            "Example: F:/261006_adaptive_control_data_test"
        )
    transfer_root_raw = storage.get("transfer_root")
    if not transfer_root_raw:
        raise KeyError(
            "Missing storage.transfer_root in adaptive_runtime_config.yaml. "
            "Example: C:/Users/<user>/Documents/azuma/99_adaptive_control_transfer"
        )

    session_root = _resolve_user_path(session_root_raw, base_dir)
    transfer_root = _resolve_user_path(transfer_root_raw, base_dir)

    resources = cfg.get("resources") or {}
    action_set_id = str(resources.get("action_set_id") or "").strip()
    target_set_id = str(resources.get("target_set_id") or "").strip()
    if not action_set_id:
        raise KeyError("Missing resources.action_set_id")
    if not target_set_id:
        raise KeyError("Missing resources.target_set_id")

    paths = dict(cfg.get("paths") or {})
    ml_repo_raw = paths.get("ml_repo", "../laser_processing_ml")
    ml_repo = _resolve_user_path(ml_repo_raw, base_dir)

    # Canonical layout. These values intentionally override legacy path entries in old
    # local YAML files so the runtime has one authoritative storage convention.
    action_set_dir = session_root / "resources" / "action_sets" / action_set_id
    target_set_dir = session_root / "resources" / "target_sets" / target_set_id
    paths.update(
        {
            "ml_repo": str(ml_repo),
            "inbox_root": str(transfer_root / "inbox"),
            "action_registry_csv": str(action_set_dir / "action_registry.csv"),
            "target_root": str(target_set_dir),
            "target_registry_csv": str(target_set_dir / "target_registry.csv"),
            "calibration_root": str(session_root / "resources" / "calibration_sessions"),
            "output_root": str(session_root / "experiments"),
        }
    )

    # Model run directories may also be expressed relative to this YAML.
    models = cfg.get("models") or {}
    for model_cfg in models.values():
        if not isinstance(model_cfg, dict):
            continue
        for key in ("state_estimator_run", "transition_run"):
            if key in model_cfg and model_cfg[key]:
                model_cfg[key] = str(_resolve_user_path(model_cfg[key], base_dir))


    # Safety-critical action-selection policy. V1.8 fails closed when no safe
    # candidate exists; legacy min-over-removal fallback is no longer allowed
    # for the live runtime.
    action_selection = cfg.get("action_selection") or {}
    no_safe_policy = str(action_selection.get("no_safe_action_policy", "stop")).strip().lower()
    if no_safe_policy != "stop":
        raise ValueError(
            "action_selection.no_safe_action_policy must be 'stop' in V1.8; "
            f"got {no_safe_policy!r}"
        )
    action_selection["no_safe_action_policy"] = "stop"
    # Policy v1.0: a deliberate and explicit switch away from L1+Dice.
    # Old YAML files with the legacy scoring parameters must not silently run.
    scoring_method = action_selection.get("scoring_method")
    if scoring_method is None and ("l1_weight" in action_selection or "dice_weight" in action_selection):
        raise ValueError("Legacy L1+Dice config: set action_selection.scoring_method=delta_area_v1 and lambda_over=3.0")
    if scoring_method is not None and scoring_method != "delta_area_v1":
        raise ValueError("action_selection.scoring_method must be 'delta_area_v1'")
    action_selection["scoring_method"] = "delta_area_v1"
    lambda_over = float(action_selection.get("lambda_over", 3.0))
    if not math.isfinite(lambda_over) or lambda_over <= 0:
        raise ValueError("action_selection.lambda_over must be finite and > 0")
    action_selection["lambda_over"] = lambda_over
    threshold = float(action_selection.get("candidate_threshold", 0.5))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("action_selection.candidate_threshold must be in (0, 1]")
    action_selection["candidate_threshold"] = threshold
    cfg["action_selection"] = action_selection

    stop_policy = cfg.get("stop_policy") or {}
    for key, default in (("overall_removal_target", 0.90), ("over_removal_limit", 0.10)):
        value = float(stop_policy.get(key, default))
        if not math.isfinite(value) or value <= 0 or (key == "overall_removal_target" and value > 1):
            raise ValueError(f"stop_policy.{key} must be a positive finite value (completion <= 1)")
        stop_policy[key] = value
    max_steps = int(stop_policy.get("max_control_steps", 20))
    if max_steps < 1:
        raise ValueError("stop_policy.max_control_steps must be >= 1")
    stop_policy["max_control_steps"] = max_steps
    cfg["stop_policy"] = stop_policy

    # Validate optional per-mode GPU inference settings.  Missing values retain
    # conservative backward-compatible defaults inside MLRuntime.
    for mode_name, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        perf = model_cfg.get("performance") or {}
        if perf:
            chunk = int(perf.get("candidate_chunk_size", 9))
            if chunk < 1:
                raise ValueError(
                    f"models.{mode_name}.performance.candidate_chunk_size must be >= 1"
                )
            perf["candidate_chunk_size"] = chunk
            perf["estimator_amp"] = bool(perf.get("estimator_amp", False))
            perf["transition_amp"] = bool(perf.get("transition_amp", False))
            model_cfg["performance"] = perf

    cfg["storage"] = {
        **storage,
        "session_root": str(session_root),
        "transfer_root": str(transfer_root),
    }
    cfg["resources"] = {
        **resources,
        "action_set_id": action_set_id,
        "target_set_id": target_set_id,
    }
    cfg["paths"] = paths
    cfg["_config_path"] = str(config_path)
    return cfg
