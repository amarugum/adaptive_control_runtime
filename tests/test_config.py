from pathlib import Path
import yaml

from runtime.config import load_runtime_config


def test_config_derives_inbox_from_transfer_root(tmp_path: Path):
    cfg_path = tmp_path / "adaptive_runtime_config.yaml"
    session_root = tmp_path / "external" / "session"
    transfer_root = tmp_path / "internal" / "transfer"
    obj = {
        "storage": {
            "session_root": str(session_root),
            "transfer_root": str(transfer_root),
        },
        "resources": {
            "action_set_id": "action27_v1",
            "target_set_id": "target_set_v1",
        },
        "paths": {"ml_repo": "../laser_processing_ml", "inbox_root": "legacy/ignored"},
        "models": {},
    }
    cfg_path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

    cfg = load_runtime_config(cfg_path)
    assert Path(cfg["paths"]["inbox_root"]) == transfer_root / "inbox"
    assert Path(cfg["paths"]["output_root"]) == session_root / "experiments"
    assert Path(cfg["paths"]["calibration_root"]) == session_root / "resources" / "calibration_sessions"
