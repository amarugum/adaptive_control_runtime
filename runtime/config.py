from __future__ import annotations

from pathlib import Path, PureWindowsPath
import re
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
