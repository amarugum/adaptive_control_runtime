from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from runtime.action_registry import ActionRegistry
from runtime.config import load_runtime_config


def _http_json(method: str, url: str, obj: dict | None = None, timeout_s: float = 120.0) -> tuple[int, dict]:
    data = None if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"status": "error", "raw_body": body}
        return int(e.code), parsed


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_orca_for_request(source: Path, inbox_root: Path, transfer_rel_dir: str) -> Path:
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"ORCA source directory not found: {source}")
    files = sorted([p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])
    if len(files) != 15:
        raise ValueError(f"Expected 15 ORCA TIFF files, found {len(files)} in {source}")

    root = Path(inbox_root) / transfer_rel_dir
    orca_dst = root / "ORCA"
    if root.exists():
        shutil.rmtree(root)
    orca_dst.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, orca_dst / src.name)
    (root / "_READY").write_text("ready\n", encoding="utf-8")
    return root


def _find_seed_action(registry: ActionRegistry, energy_uJ: float, y_um: float, z_um: float):
    matches = [
        a for a in registry.actions
        if abs(a.energy_uJ - energy_uJ) < 1e-9
        and abs(a.action_y_um - y_um) < 1e-9
        and abs(a.action_z_um - z_um) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve seed action ({energy_uJ}, {y_um}, {z_um}); matches={len(matches)}"
        )
    return matches[0]


def _candidate_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _validate_artifacts(out_dir: Path, response: dict) -> list[str]:
    problems: list[str] = []
    for name in ("request.json", "response.json", "decision.json", "state_probability.npy", "state_binary.npy"):
        if not (out_dir / name).is_file():
            problems.append(f"missing {name}")

    if response.get("decision") == "continue":
        for name in ("candidate_probabilities.npy", "candidate_binary.npy", "candidate_scores.csv"):
            if not (out_dir / name).is_file():
                problems.append(f"missing {name}")
        score_csv = out_dir / "candidate_scores.csv"
        if score_csv.is_file() and _candidate_row_count(score_csv) != 27:
            problems.append("candidate_scores.csv does not contain 27 rows")
    return problems


def _build_request(
    *,
    mode: str,
    label: str,
    target_id: str,
    action_registry_sha256: str,
    seed_action,
    calibration_id: str | None,
    transfer_rel_dir: str | None,
    max_control_steps: int,
    previous_start_frame_1based: int | None,
    hole_origin_y_um: float,
    hole_origin_z_um: float,
    actual_y_um: float,
    actual_z_um: float,
) -> dict:
    experiment_id = f"smoke_decision_{label}_{mode}"
    hole_uid = "h0001"
    k = 1
    request_id = f"{experiment_id}_{hole_uid}_k01"
    request = {
        "protocol_version": "1.0",
        "request_id": request_id,
        "experiment": {
            "experiment_id": experiment_id,
            "hole_uid": hole_uid,
            "hole_index": 1,
        },
        "step": {
            "completed_control_step": k,
            "next_control_step": k + 1,
            "pulse_start": 1,
            "pulse_end": 5,
            "pulse_count": 5,
        },
        "model": {"mode": mode},
        "target": {"target_id": target_id},
        "executed_action": {
            "action_id": seed_action.action_id,
            "energy_uJ": float(seed_action.energy_uJ),
            "action_y_um": float(seed_action.action_y_um),
            "action_z_um": float(seed_action.action_z_um),
        },
        "action_history": [
            {
                "control_step": 1,
                "action_id": seed_action.action_id,
                "energy_uJ": float(seed_action.energy_uJ),
                "action_y_um": float(seed_action.action_y_um),
                "action_z_um": float(seed_action.action_z_um),
                "pulse_count": 5,
            }
        ],
        "stage": {
            "hole_origin_y_um": float(hole_origin_y_um),
            "hole_origin_z_um": float(hole_origin_z_um),
            "actual_y_um": float(actual_y_um),
            "actual_z_um": float(actual_z_um),
            "actual_nd_mm": 0.0,
        },
        "control": {
            "max_control_steps": int(max_control_steps),
            "action_registry_sha256": action_registry_sha256,
        },
        "frame_selection_context": {
            "previous_used_start_frame_1based": previous_start_frame_1based
        },
    }
    if mode in {"shock_metadata", "shock_action_history"}:
        if not calibration_id or not transfer_rel_dir:
            raise ValueError("Shock mode requires calibration_id and transfer_rel_dir")
        request["calibration"] = {"session_calibration_id": calibration_id}
        request["acquisition"] = {
            "orca": {
                "transfer_rel_dir": transfer_rel_dir,
                "file_count": 15,
                "expected_file_count": 15,
                "transfer_complete": True,
            },
            "hasu2": {"saved_on_pc1": True},
        }
    else:
        request["calibration"] = {"session_calibration_id": calibration_id or "not_required"}
        request["acquisition"] = {
            "orca": {"required": False, "transfer_complete": False},
            "hasu2": {"saved_on_pc1": True},
        }
    return request


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Full PC2 decision smoke test through the running HTTP server: request -> target -> "
            "State Estimator -> stop policy -> 27 candidates (when continuing) -> action selection -> "
            "response -> idempotent retry -> ACK."
        )
    )
    ap.add_argument("--config", default="adaptive_runtime_config.yaml")
    ap.add_argument("--server-url", default="http://127.0.0.1:8765")
    ap.add_argument("--orca-dir", help="Saved 15-frame ORCA directory; required for shock modes")
    ap.add_argument("--calibration-id", help="Calibration session ID; required for shock modes")
    ap.add_argument("--target-id", default="T001")
    ap.add_argument(
        "--modes",
        default="metadata_only,action_history_only,shock_metadata,shock_action_history",
    )
    ap.add_argument("--label", default="v1_4")
    ap.add_argument("--previous-start-frame-1based", type=int, default=None)
    ap.add_argument("--hole-origin-y-um", type=float, default=0.0)
    ap.add_argument("--hole-origin-z-um", type=float, default=0.0)
    ap.add_argument("--actual-y-um", type=float, default=0.0)
    ap.add_argument("--actual-z-um", type=float, default=0.0)
    ap.add_argument("--no-ack", action="store_true", help="Do not send final ACK; useful when inspecting inbox")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_runtime_config(cfg_path)
    registry = ActionRegistry(Path(cfg["paths"]["action_registry_csv"]))
    min_energy = min(a.energy_uJ for a in registry.actions)
    seed = _find_seed_action(registry, min_energy, 0.0, 0.0)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    allowed = set(cfg["models"])
    unknown = [m for m in modes if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}")
    if any(m in {"shock_metadata", "shock_action_history"} for m in modes):
        if not args.orca_dir:
            raise ValueError("--orca-dir is required when testing shock modes")
        if not args.calibration_id:
            raise ValueError("--calibration-id is required when testing shock modes")
        cal_dir = Path(cfg["paths"]["calibration_root"]) / args.calibration_id
        if not cal_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {cal_dir}")

    # Confirm the running server is the expected session before writing any inbox data.
    status, health = _http_json("GET", args.server_url.rstrip("/") + "/v1/health", timeout_s=15.0)
    if status != 200 or health.get("status") != "ok":
        raise RuntimeError(f"Health check failed: HTTP {status}: {health}")
    expected_root = str(Path(cfg["storage"]["session_root"]))
    if str(health.get("session_root")) != expected_root:
        raise RuntimeError(
            "Running server session_root does not match this config. "
            f"server={health.get('session_root')} config={expected_root}"
        )
    expected_transfer_root = str(Path(cfg["storage"]["transfer_root"]))
    if str(health.get("transfer_root")) != expected_transfer_root:
        raise RuntimeError(
            "Running server transfer_root does not match this config. "
            f"server={health.get('transfer_root')} config={expected_transfer_root}"
        )
    expected_inbox = str(Path(cfg["paths"]["inbox_root"]))
    if str(health.get("inbox_root")) != expected_inbox:
        raise RuntimeError(
            "Running server inbox_root does not match this config. "
            f"server={health.get('inbox_root')} config={expected_inbox}"
        )

    session_root = Path(cfg["storage"]["session_root"])
    diagnostics_root = session_root / "diagnostics" / "decision_smoke" / args.label
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    source_orca = None if args.orca_dir is None else Path(args.orca_dir).resolve()

    print("=== Full decision smoke test ===")
    print(f"config         : {cfg_path}")
    print(f"server         : {args.server_url}")
    print(f"session_root   : {session_root}")
    print(f"target         : {args.target_id}")
    print(f"seed action    : {seed.action_id} ({seed.energy_uJ} uJ, y={seed.action_y_um}, z={seed.action_z_um})")
    if source_orca is not None:
        print(f"ORCA source    : {source_orca}")
    print(f"diagnostics    : {diagnostics_root}")

    passed = 0
    summaries: list[dict] = []
    for mode in modes:
        print(f"\n--- {mode} ---")
        mode_diag = diagnostics_root / mode
        mode_diag.mkdir(parents=True, exist_ok=True)
        transfer_rel_dir = None
        inbox_request_root = None
        if mode in {"shock_metadata", "shock_action_history"}:
            transfer_rel_dir = f"smoke_decision/{args.label}/{mode}"
            inbox_request_root = _copy_orca_for_request(
                source_orca, Path(cfg["paths"]["inbox_root"]), transfer_rel_dir
            )
            print(f"inbox copy     : {inbox_request_root}")
        request_obj = _build_request(
            mode=mode,
            label=args.label,
            target_id=args.target_id,
            action_registry_sha256=registry.sha256,
            seed_action=seed,
            calibration_id=args.calibration_id,
            transfer_rel_dir=transfer_rel_dir,
            max_control_steps=int(cfg["stop_policy"]["max_control_steps"]),
            previous_start_frame_1based=args.previous_start_frame_1based,
            hole_origin_y_um=args.hole_origin_y_um,
            hole_origin_z_um=args.hole_origin_z_um,
            actual_y_um=args.actual_y_um,
            actual_z_um=args.actual_z_um,
        )
        _json_dump(mode_diag / "request_sent.json", request_obj)
        t0 = time.perf_counter()
        code, response = _http_json(
            "POST", args.server_url.rstrip("/") + "/v1/decision", request_obj, timeout_s=180.0
        )
        elapsed = time.perf_counter() - t0
        _json_dump(mode_diag / "response_received.json", response)
        if code != 200 or response.get("status") != "ok":
            print(f"FAIL HTTP {code}: {response}")
            summaries.append({"mode": mode, "pass": False, "reason": f"HTTP {code}", "response": response})
            continue

        experiment_id = request_obj["experiment"]["experiment_id"]
        out_dir = Path(cfg["paths"]["output_root"]) / experiment_id / "h0001" / "k01"
        problems = _validate_artifacts(out_dir, response)

        # Idempotency check: same request must return byte-equivalent JSON content.
        retry_code, retry_response = _http_json(
            "POST", args.server_url.rstrip("/") + "/v1/decision", request_obj, timeout_s=30.0
        )
        idempotent = retry_code == 200 and retry_response == response
        if not idempotent:
            problems.append("idempotent retry returned a different response")

        state_eval = response.get("state_evaluation") or {}
        cand_eval = response.get("candidate_evaluation")
        best = response.get("best_action")
        print(f"decision       : {response.get('decision')}")
        print(f"stop reason    : {response.get('stop_reason')}")
        print(
            "state          : overall={:.6f} over={:.6f} under={:.6f}".format(
                float(state_eval.get("overall_removal_fraction", float("nan"))),
                float(state_eval.get("over_removal_fraction", float("nan"))),
                float(state_eval.get("under_removal_fraction", float("nan"))),
            )
        )
        if response.get("frame_selection") is not None:
            fs = response["frame_selection"]
            print(
                f"frame select   : auto={fs.get('auto_start_frame')} used={fs.get('used_start_frame')} "
                f"source={fs.get('selection_source')} conf={fs.get('confidence'):.6f} "
                f"contrast={fs.get('contrast_score'):.6f}"
            )
        if cand_eval is not None:
            print(
                f"candidates     : safe={cand_eval.get('safe_candidate_count')} "
                f"no_safe={cand_eval.get('no_safe_action')}"
            )
        if best is not None:
            print(
                f"best action    : {best.get('action_id')} E={best.get('energy_uJ')} "
                f"y={best.get('action_y_um')} z={best.get('action_z_um')} "
                f"J={best.get('predicted_cost'):.6f}"
            )
        print(f"response time  : {elapsed:.3f} s")
        print(f"idempotent     : {idempotent}")
        print(f"artifacts dir  : {out_dir}")

        ack_ok = None
        inbox_removed = None
        if not args.no_ack:
            ack_obj = {
                "protocol_version": "1.0",
                "request_id": request_obj["request_id"],
                "acknowledged": True,
                "accepted": True,
            }
            ack_code, ack_resp = _http_json(
                "POST", args.server_url.rstrip("/") + "/v1/ack", ack_obj, timeout_s=15.0
            )
            ack_ok = ack_code == 200 and ack_resp.get("status") == "ok"
            if not ack_ok:
                problems.append(f"ACK failed: HTTP {ack_code}: {ack_resp}")
            if inbox_request_root is not None:
                inbox_removed = not inbox_request_root.exists()
                if not inbox_removed:
                    problems.append("ACK accepted but temporary shock-mode inbox directory still exists")
            print(f"ACK            : {ack_ok}")
            if inbox_removed is not None:
                print(f"inbox removed  : {inbox_removed}")

        ok = not problems
        if ok:
            passed += 1
            print("PASS")
        else:
            print("FAIL")
            for p in problems:
                print(f"  - {p}")
        summary = {
            "mode": mode,
            "pass": ok,
            "http_code": code,
            "decision": response.get("decision"),
            "stop_reason": response.get("stop_reason"),
            "state_evaluation": state_eval,
            "candidate_evaluation": cand_eval,
            "best_action": best,
            "elapsed_s": elapsed,
            "idempotent": idempotent,
            "ack_ok": ack_ok,
            "inbox_removed": inbox_removed,
            "artifacts_dir": str(out_dir),
            "problems": problems,
        }
        summaries.append(summary)
        _json_dump(mode_diag / "summary.json", summary)

    overall = {"passed": passed, "total": len(modes), "modes": summaries}
    _json_dump(diagnostics_root / "summary.json", overall)
    print("\n=== Summary ===")
    print(f"passed: {passed}/{len(modes)}")
    for x in summaries:
        print(f"  {'PASS' if x.get('pass') else 'FAIL'} {x.get('mode')}: decision={x.get('decision')} stop={x.get('stop_reason')}")
    print(f"diagnostics: {diagnostics_root}")
    return 0 if passed == len(modes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
