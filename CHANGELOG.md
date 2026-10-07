# Changelog

## V1.6
- Split temporary transfer storage from persistent experiment storage.
- Added required `storage.transfer_root`; `inbox_root` is now derived as `<transfer_root>/inbox`.
- Current PC2 transfer root is `C:/Users/3-019A_ShockExp7/Documents/azuma/99_adaptive_control_transfer`.
- Persistent resources, calibration, experiment artifacts, and diagnostics remain under the date/session `storage.session_root` on the external SSD.
- `init_storage.py` now creates/reports both the persistent session root and internal transfer root.
- `/v1/health` now reports `transfer_root` in addition to `session_root` and `inbox_root`.
- Added safe resolution of request `transfer_rel_dir`; absolute paths and parent-directory traversal are rejected before processing or ACK cleanup.
- Added tests for separated storage roots and inbox path-confinement.

## V1.5
- Added `create_smoke_target.py` to create/register a diagnostic all-ones 256x256 target for exercising the full HTTP CONTINUE branch.
- Added unit coverage for the `no_safe_action` fallback policy (`minimum predicted over-removal`, then cost as implemented by selector ordering).
- No production decision logic changed.

## V1.4
- Added `smoke_test_decision.py` for a full PC2 decision-path test through the running HTTP server.
- The test covers all four ML modes, target loading, stop-policy evaluation, 27-candidate scoring/selection when the decision continues, artifact validation, request idempotency, ACK handling, and temporary ORCA inbox cleanup.
- Shock modes copy a saved 15-frame ORCA acquisition into the session inbox to exercise the same request-referenced input path used by closed-loop operation; metadata-only modes do not transfer ORCA.
- Each mode uses a separate diagnostic experiment ID because model mode is intentionally immutable within one experiment snapshot.

## V1.3
- Added `smoke_test_real_orca.py` for a real-data PC2-only integration test.
- The test runs the exact online ORCA processor on one saved 15-frame step, records auto/used start-frame selection and confidence/contrast, generates the real 5-pulse x 3-time shock tensor, and runs both shock ML modes through State Estimator + matching Transition + 27-candidate sweep.
- Diagnostic artifacts are stored under `<session_root>/diagnostics/real_orca_smoke/<label>/` and do not modify the experiment archive or source raw data.

## V1.2
- Added `smoke_test_models.py` for PC2-only validation of all four configured ML pipelines.
- The smoke test verifies checkpoint presence/loading, State Estimator dummy forward, matching Transition A/B loading, 27-action candidate sweep, output shapes, finite values, and the Transition monotonic processing constraint.

## V1.1
- Added date/session-scoped external storage via `storage.session_root`.
- Added `init_storage.py`, resource sets, experiment snapshots, config-path resolution, health storage reporting, and bundled templates.

## V1.0
- Initial PC2 adaptive-control runtime: HTTP request/ACK, four ML modes, ORCA online path, candidate sweep, target scoring, stop/action-selection policy, and artifact saving.

## V1.7 - Target selection and offline control validation tools
- Added `build_target_candidates_from_dataset.py` to extract diverse, actually observed cumulative processed masks from the existing ML dataset as 256x256 binary target candidates.
- Added `screen_target_candidates.py` to re-score saved 27-action Transition predictions against multiple candidate targets without rerunning acquisition or irradiation. It reports safe-action count, no-safe fraction, selected-action cost, predicted target coverage/over-removal, action diversity, extreme-energy selection, and optional nearest-observed IoU.
- Added `register_target_candidate.py` to validate and register one approved non-smoke target without overwriting an existing target.
- Added `validate_stop_policy_offline.py` to exercise the exact runtime STOP priority: over-removal, target-reached, then max-control-steps.
- No production inference or decision logic changed.
- Existing PC2 unit suite: 13 passed; stop-policy branch validation: 4/4 passed.
