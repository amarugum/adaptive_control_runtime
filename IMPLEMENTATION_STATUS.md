# Implementation status

## V1.6 implemented
- PC2 internal-SSD temporary transfer root separated from external-SSD persistent session root.
- `inbox` lives at `<transfer_root>/inbox` and is the only directory intended for SMB sharing.
- Persistent action/target resources, calibration snapshots, experiment snapshots/results, and diagnostics remain under `<session_root>`.
- HTTP runtime, idempotent request handling and accepted-ACK inbox cleanup.
- Inbox transfer path-confinement to prevent processing/deletion outside the configured inbox.
- Four ML modes.
- ORCA-only online path for shock modes.
- State inference, 27-action Transition sweep, candidate NPY/CSV saving.
- Target/over-removal stop rules and no-safe-action fallback.
- Storage initialization command (`init_storage.py`).
- Model, real-ORCA, STOP/CONTINUE decision smoke-test tooling.

## Already validated on PC2 hardware before V1.6
- CUDA / RTX 2080 SUPER environment.
- All four State Estimator + matching Transition pipelines: 4/4 PASS.
- Real saved ORCA k01 -> online postprocessing -> 5x3 shock -> both shock State Estimators -> matching Transition B -> 27 candidates: 2/2 PASS.
- Full HTTP STOP branch in all four modes.
- Full HTTP CONTINUE branch in all four modes using diagnostic `T_SMOKE_ALL`.
- Request idempotency, ACK, and temporary inbox cleanup in local PC2 smoke tests.

## Next validation stage
- Initialize/check the new internal transfer root.
- SMB-share only `C:\\Users\\3-019A_ShockExp7\\Documents\\azuma\\99_adaptive_control_transfer\\inbox` to PC1.
- Validate PC1 -> PC2 SMB + HTTP over the dedicated Ethernet link without laser irradiation.
