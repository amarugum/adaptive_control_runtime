# adaptive_control_runtime (PC2)

Place this directory next to `laser_processing_postprocessing` and `laser_processing_ml` under `1_prog/`.

## V1.6 storage model

V1.6 separates **temporary PC1->PC2 transfers** from the **persistent experiment archive**.

### PC2 internal SSD: temporary SMB transfer area

```text
C:/Users/3-019A_ShockExp7/Documents/azuma/99_adaptive_control_transfer/
└─ inbox/                         # SMB-share only this directory
   └─ <request transfer dir>/
      └─ ORCA/                    # temporary copied TIFFs for shock modes
```

`inbox/` is not an experiment archive. For shock modes, PC1 copies ORCA files here, PC2 processes them, and an accepted `/v1/ack` removes the request transfer directory. `metadata_only` and `action_history_only` do not need ORCA transfer.

### External SSD: persistent date/session archive

```text
F:/261006_adaptive_control_data_test/
├─ resources/
│  ├─ action_sets/
│  │  └─ action27_v1/action_registry.csv
│  ├─ target_sets/
│  │  └─ target_set_v1/
│  │     ├─ target_registry.csv
│  │     └─ T001.npy
│  └─ calibration_sessions/
├─ experiments/
│  └─ <experiment_id>/
│     ├─ experiment_snapshot/
│     └─ <hole_uid>/kXX/          # state/candidate/decision artifacts
└─ diagnostics/                   # smoke/integration-test outputs
```

On a new experimental date, normally change only `storage.session_root`. `storage.transfer_root` stays fixed because it points to the PC2 internal SSD transfer area.

Then run:

```bat
python init_storage.py --config adaptive_runtime_config.yaml
```

Initialization creates both roots/directories as needed and copies bundled action/target templates only when the persistent destination does not already exist. Existing action/target resources are never overwritten.

## Runtime behavior

- The server waits for HTTP `/v1/decision`; it does **not** poll `inbox/`.
- `metadata_only` and `action_history_only` do not require ORCA transfer.
- `shock_metadata` and `shock_action_history` read ORCA data from the request-referenced directory under `<transfer_root>/inbox/`.
- PC2 saves state, all 27 candidate predictions, scores, and decisions persistently under `<session_root>/experiments/`.
- After PC1 accepts the response and sends ACK, the corresponding temporary ORCA transfer directory under `inbox/` is deleted; persistent experiment artifacts remain.
- Request-supplied transfer paths are constrained to remain under `inbox/`; absolute paths and `..` traversal are rejected.
- On first use of an experiment ID, the runtime freezes the action registry, target registry/target file, and runtime policy under `experiment_snapshot/`. Conflicting later changes are rejected.

## Config

The main storage/resource settings are:

```yaml
storage:
  session_root: F:/261006_adaptive_control_data_test
  transfer_root: C:/Users/3-019A_ShockExp7/Documents/azuma/99_adaptive_control_transfer

resources:
  action_set_id: action27_v1
  target_set_id: target_set_v1

paths:
  ml_repo: ../laser_processing_ml
```

Derived paths are:

```text
inbox_root       = <transfer_root>/inbox
output_root      = <session_root>/experiments
calibration_root = <session_root>/resources/calibration_sessions
```

Legacy manually specified `inbox_root`, `output_root`, etc. entries are ignored/overridden by the config loader.

## PC2 ML smoke test

After `init_storage.py` has initialized the roots, validate all four configured model pipelines before using real ORCA data:

```bat
python smoke_test_models.py --config adaptive_runtime_config.yaml
```

The test loads each mode one at a time on the configured device, performs a deterministic dummy State Estimator forward, runs all 27 Transition candidates from the external `action_registry.csv`, checks output shapes/NaN/Inf/monotonicity, then releases GPU memory before the next mode. A successful run ends with `passed: 4/4`.

To test one mode only:

```bat
python smoke_test_models.py --config adaptive_runtime_config.yaml --modes shock_metadata
```

## Real ORCA smoke test

With a saved **k01** ORCA acquisition and calibration snapshot, place/copy the calibration under:

`<session_root>/resources/calibration_sessions/<calibration_id>/`

Then run, for example:

```bat
python smoke_test_real_orca.py --config adaptive_runtime_config.yaml --orca-dir "F:/path/to/h0001/k01/ORCA" --calibration-id cal_test01 --label h0001_k01
```

The source raw directory is not modified. Results are written to:

`<session_root>/diagnostics/real_orca_smoke/<label>/`

## Full decision smoke test

Start the runtime server:

```bat
python server.py --config adaptive_runtime_config.yaml
```

Then run:

```bat
python smoke_test_decision.py --config adaptive_runtime_config.yaml --orca-dir "Q:/path/to/h0001/k01/ORCA" --calibration-id cal_test01 --target-id T001 --label test01
```

The test sends real `/v1/decision` requests to the local server. Shock modes copy the 15 saved TIFFs into `<transfer_root>/inbox/`, verify idempotent retry, then send accepted ACK and verify temporary inbox cleanup. Persistent decision artifacts are written under `<session_root>/experiments/`; smoke summaries are written under `<session_root>/diagnostics/decision_smoke/`.

### Diagnostic CONTINUE-branch target

Run:

```bat
python create_smoke_target.py --config adaptive_runtime_config.yaml
```

This creates/registers `T_SMOKE_ALL` for integration testing only. Restart `server.py` afterward because the target registry is loaded when the server starts. Do not use `T_SMOKE_ALL` as a scientific target.

## Version history

See `CHANGELOG.md`.
