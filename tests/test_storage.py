from pathlib import Path
import numpy as np
import pytest

from runtime.storage import initialize_storage, resolve_inbox_transfer_dir


def _cfg(session_root: Path, transfer_root: Path) -> dict:
    return {
        "storage": {
            "session_root": str(session_root),
            "transfer_root": str(transfer_root),
        },
        "resources": {"action_set_id": "action27_v1", "target_set_id": "target_set_v1"},
        "paths": {
            "inbox_root": str(transfer_root / "inbox"),
            "output_root": str(session_root / "experiments"),
            "calibration_root": str(session_root / "resources" / "calibration_sessions"),
            "action_registry_csv": str(
                session_root
                / "resources"
                / "action_sets"
                / "action27_v1"
                / "action_registry.csv"
            ),
            "target_root": str(session_root / "resources" / "target_sets" / "target_set_v1"),
            "target_registry_csv": str(
                session_root
                / "resources"
                / "target_sets"
                / "target_set_v1"
                / "target_registry.csv"
            ),
        },
    }


def test_initialize_storage_separates_persistent_and_transfer_roots(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    action_t = runtime_root / "templates" / "action_sets" / "action27_v1"
    target_t = runtime_root / "templates" / "target_sets" / "target_set_v1"
    action_t.mkdir(parents=True)
    target_t.mkdir(parents=True)
    (action_t / "action_registry.csv").write_text(
        "action_id,energy_uJ,action_y_um,action_z_um\n"
        + "\n".join(f"A{i:02d},10,0,0" for i in range(1, 28))
        + "\n",
        encoding="utf-8",
    )
    (target_t / "target_registry.csv").write_text(
        "target_id,file_name,description\nT001,T001.npy,test\n", encoding="utf-8"
    )
    np.save(target_t / "T001.npy", np.ones((256, 256), dtype=np.uint8), allow_pickle=False)

    session_root = tmp_path / "external" / "261007_adaptive_control_data"
    transfer_root = tmp_path / "internal" / "99_adaptive_control_transfer"
    cfg = _cfg(session_root, transfer_root)

    report = initialize_storage(cfg, runtime_root)
    assert Path(report["session_root"]) == session_root
    assert Path(report["transfer_root"]) == transfer_root
    assert Path(report["inbox"]) == transfer_root / "inbox"
    assert Path(report["inbox"]).is_dir()
    assert not (session_root / "inbox").exists()
    assert Path(report["experiments"]).is_dir()
    assert Path(report["action_registry"]["path"]).is_file()
    assert Path(report["target_registry"]["path"]).is_file()
    assert (session_root / "resources" / "target_sets" / "target_set_v1" / "T001.npy").is_file()

    # Re-initialization must not overwrite existing persistent resources.
    report2 = initialize_storage(cfg, runtime_root)
    assert report2["action_registry"]["status"] == "exists"
    assert report2["target_registry"]["status"] == "exists"


def test_resolve_inbox_transfer_dir_accepts_relative_child(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    p = resolve_inbox_transfer_dir(inbox, "exp001/h0001/k01")
    assert p == (inbox / "exp001" / "h0001" / "k01").resolve()


@pytest.mark.parametrize(
    "rel",
    [
        "../outside",
        "x/../../outside",
        r"..\outside",
        r"C:\outside",
        "/absolute/outside",
        r"\\server\share",
        "",
    ],
)
def test_resolve_inbox_transfer_dir_rejects_escape(tmp_path: Path, rel: str):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    with pytest.raises(ValueError):
        resolve_inbox_transfer_dir(inbox, rel)
