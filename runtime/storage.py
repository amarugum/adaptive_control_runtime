from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_inbox_transfer_dir(inbox_root: Path, transfer_rel_dir: str) -> Path:
    """Resolve a request transfer directory and guarantee it remains inside inbox_root.

    PC1 supplies ``transfer_rel_dir`` over HTTP. Even on a trusted dedicated link, the
    runtime must never allow an absolute path or ``..`` traversal to make processing or
    ACK cleanup escape the temporary inbox tree.
    """
    root = Path(inbox_root).resolve()
    rel_text = str(transfer_rel_dir or "").strip()
    if not rel_text:
        raise ValueError("EMPTY_TRANSFER_REL_DIR")

    # Treat both slash styles as separators, regardless of the host OS used for tests.
    normalized = rel_text.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        raise ValueError("INVALID_TRANSFER_REL_DIR")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError("INVALID_TRANSFER_REL_DIR")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ValueError("INVALID_TRANSFER_REL_DIR")

    rel = Path(*parts)
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("TRANSFER_PATH_OUTSIDE_INBOX") from exc
    return candidate


def _copy_if_missing(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if not src.exists():
        return "template_missing"
    shutil.copy2(src, dst)
    return "created"


def initialize_storage(cfg: dict, runtime_root: Path) -> dict[str, Any]:
    """Create persistent session storage plus the PC2-internal temporary inbox."""
    session_root = Path(cfg["storage"]["session_root"])
    transfer_root = Path(cfg["storage"]["transfer_root"])
    action_set_id = cfg["resources"]["action_set_id"]
    target_set_id = cfg["resources"]["target_set_id"]

    inbox = Path(cfg["paths"]["inbox_root"])
    experiments = Path(cfg["paths"]["output_root"])
    calibration_sessions = Path(cfg["paths"]["calibration_root"])
    action_dir = Path(cfg["paths"]["action_registry_csv"]).parent
    target_dir = Path(cfg["paths"]["target_root"])

    for d in (
        session_root,
        transfer_root,
        inbox,
        experiments,
        calibration_sessions,
        action_dir,
        target_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    template_root = Path(runtime_root) / "templates"
    action_template = template_root / "action_sets" / action_set_id / "action_registry.csv"
    target_template_dir = template_root / "target_sets" / target_set_id

    action_dst = action_dir / "action_registry.csv"
    action_status = _copy_if_missing(action_template, action_dst)

    target_registry_dst = target_dir / "target_registry.csv"
    target_registry_status = _copy_if_missing(
        target_template_dir / "target_registry.csv", target_registry_dst
    )

    target_files: dict[str, str] = {}
    if target_template_dir.exists():
        for src in sorted(target_template_dir.glob("*.npy")):
            target_files[src.name] = _copy_if_missing(src, target_dir / src.name)

    return {
        "session_root": str(session_root),
        "transfer_root": str(transfer_root),
        "inbox": str(inbox),
        "experiments": str(experiments),
        "calibration_sessions": str(calibration_sessions),
        "action_registry": {"path": str(action_dst), "status": action_status},
        "target_registry": {
            "path": str(target_registry_dst),
            "status": target_registry_status,
        },
        "target_files": target_files,
    }


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _copy_or_verify(src: Path, dst: Path, conflict_code: str) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(src) != sha256_file(dst):
            raise ValueError(conflict_code)
        return
    shutil.copy2(src, dst)


def _write_or_verify_json(path: Path, obj: Any, conflict_code: str) -> None:
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old = json.dumps(existing, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if old != canonical:
            raise ValueError(conflict_code)
        return
    _atomic_write_json(path, obj)


def ensure_experiment_snapshot(
    *,
    cfg: dict,
    experiment_id: str,
    target_id: str,
    target_file: Path,
    action_registry_file: Path,
    target_registry_file: Path,
    mode: str,
    target_sha256: str,
) -> Path:
    """Freeze resources/policies actually used by an experiment.

    Target files are accumulated by target_id, while action/target registries and the
    experiment-wide runtime policy are immutable once first used.
    """
    snapshot = Path(cfg["paths"]["output_root"]) / experiment_id / "experiment_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)

    _copy_or_verify(
        action_registry_file,
        snapshot / "action_registry.csv",
        "EXPERIMENT_ACTION_REGISTRY_CONFLICT",
    )
    _copy_or_verify(
        target_registry_file,
        snapshot / "target_registry.csv",
        "EXPERIMENT_TARGET_REGISTRY_CONFLICT",
    )
    _copy_or_verify(
        target_file,
        snapshot / "targets" / target_file.name,
        "EXPERIMENT_TARGET_FILE_CONFLICT",
    )

    policy = {
        "action_set_id": cfg["resources"]["action_set_id"],
        "target_set_id": cfg["resources"]["target_set_id"],
        "model_mode": mode,
        "model_config": cfg["models"][mode],
        "action_selection": cfg["action_selection"],
        "stop_policy": cfg["stop_policy"],
        "orca_online": cfg["orca_online"],
    }
    _write_or_verify_json(
        snapshot / "runtime_policy.json",
        policy,
        "EXPERIMENT_RUNTIME_POLICY_CONFLICT",
    )

    targets_manifest_path = snapshot / "targets_manifest.json"
    manifest = {}
    if targets_manifest_path.exists():
        manifest = json.loads(targets_manifest_path.read_text(encoding="utf-8"))
    previous = manifest.get(target_id)
    entry = {"file_name": target_file.name, "sha256": target_sha256}
    if previous is not None and previous != entry:
        raise ValueError("EXPERIMENT_TARGET_HASH_CONFLICT")
    manifest[target_id] = entry
    _atomic_write_json(targets_manifest_path, manifest)
    return snapshot
