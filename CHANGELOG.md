# Changelog

## 2026-10-13 — Policy v1.0 LIVE

- Replaced L1+Dice live Action scoring with `L=3 ΔO−ΔC` on nonnegative predicted increments; all four model modes share it.
- Two-stage STOP: state over 0.10 / coverage 0.90 / max steps; then safe set `predicted next over<0.10`, STOP on no safe action or `min L>=0`.
- Fail-closed candidate handling; per-candidate Δ values, policy metadata, strict config validation and policy health status.
- Retained legacy offline selector for reproducibility; updated live settings and added CPU integration tests.
- PC1 unchanged (not supplied); see `POLICY_V1_20261013.md`.


## V1.12 — overall/over-removal policy sensitivity analysis
- Added `validation/policy_sensitivity_analysis.py`.
- Separates two roles that were previously conflated: continuous `overall`/`over` fractions for scientific model comparison, and hard thresholds used only for runtime STOP/CONTINUE decisions.
- Builds one reusable `.npz` cache containing all 27 candidate costs, overall fractions, and over-removal fractions for both estimated-current and oracle-current paths.
- Sweeps multiple `overall_removal_target` and `over_removal_limit` pairs without rerunning GPU inference for every pair.
- Reports decision agreement, false CONTINUE/false STOP, stop-reason fractions, no-safe rates, selected-action agreement, whether the estimated-selected action is unsafe under the oracle-current branch, and oracle-cost regret when the estimated action remains oracle-safe.
- Outputs summaries overall, by target, and by pulse, plus threshold-independent continuous overall/over estimation summaries and heatmaps.
- Does not automatically choose a “best” threshold pair; historical fixed-plan trajectories were not optimized toward the eight targets, so target-reached frequency is not a valid scalar objective by itself.
- Runtime inference, selector behavior, and the frozen four-model FP32/chunk=6 configuration are unchanged.

# v1.11

- `validation/replay_end_to_end_offline.py` now shows setup-stage progress immediately.
- Long opaque dataset-building calls show an elapsed-time heartbeat, so the console no longer appears frozen while thousands of metadata `.npy` files are loaded.
- Compatible State Estimator sample maps are cached across the four modes (e.g. shared G1/G5 sample construction when checkpoint data settings match), avoiding redundant metadata scans.
- Compatible Transition row maps are cached across modes.
- Setup prints per-mode model loading, state-map/transition-map build or cache hit, common-state intersection, and selected state counts by pulse before GPU replay starts.
- Runtime example/config in this package reflects the frozen FP32 + `candidate_chunk_size=6` setting for all four modes.

# Changelog

## V1.10 — four-model GPU freeze + end-to-end replay
- Expanded GPU benchmarking defaults to all four runtime modes: `metadata_only`, `action_history_only`, `shock_metadata`, and `shock_action_history`.
- Production-intent inference precision is FP32 for both State Estimator and Transition in all four modes. AMP stays disabled after the action-equivalence gate found rare final-action/safe-count changes near decision boundaries.
- GPU benchmark v3 defaults to FP32 only and selects `candidate_chunk_size` per mode under runtime-like B=1 conditions.
- Added `benchmarks/apply_fp32_recommendations.py` to safely apply four-mode chunk recommendations while forcing AMP off and backing up the runtime YAML.
- Expanded end-to-end offline replay defaults to all four modes and uses only held-out physical states common to every mode for fair comparison.
- `metadata_only` and `shock_action_history` ship with temporary conservative chunk=9 until the four-model benchmark is applied; previously benchmarked `action_history_only`/`shock_metadata` remain chunk=12/27.


## V1.8 - GPU runtime settings + fail-closed action selection
- Added per-model runtime GPU settings: `estimator_amp`, `transition_amp`, and `candidate_chunk_size`.
- Applied benchmarked settings for the two primary comparison modes: benchmark-recommended settings are Estimator FP32 + Transition AMP with chunk 12 (parameter-only) and chunk 27 (shock-informed); V1.8 ships with Transition AMP disabled until the action-equivalence gate passes.
- Runtime responses now record the effective inference-performance settings for provenance.
- Changed live no-safe behavior to fail closed: if all 27 candidates exceed the configured over-removal limit, the runtime returns `decision=stop`, `stop_reason=no_safe_action`, and `best_action=null`. No unsafe fallback action is sent to PC1.
- Config loading rejects legacy live `min_over_then_cost` no-safe policy. The selector still computes a diagnostic minimum-over-removal row for offline analysis only.
- Added `validation/validate_amp_action_equivalence.py` to compare AMP vs FP32 under runtime-like B=1 conditions across sampled test states and all prepared reachability targets. PASS requires zero decision mismatches and zero selected-action mismatches.
- Added config tests for the new performance settings and fail-closed no-safe policy.

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

## V1.9 — end-to-end offline replay gate

- Kept the validated production-intent GPU settings in config: Estimator FP32, Transition FP32, chunk 12 (`action_history_only`) / 27 (`shock_metadata`).
- Added `validation/replay_end_to_end_offline.py`.
- Replay uses the actual `MLRuntime` State Estimator → Transition → selector path on held-out physical states.
- Added measured-current oracle comparison to quantify how State Estimator error changes STOP/CONTINUE and selected action.
- Added actual-observed-action next-state evaluation, which is the physically valid end-to-end next-state ground-truth comparison.
- Added known white-band artifact hole exclusion list for offline validation only.
- No scientific PASS/FAIL threshold is hard-coded; the result is reviewed before the real-laser shadow gate.

## Validation add-on (2026-10-11; runtime V1.12 unchanged)
- Added cache-only `validation/continuation_value_sensitivity.py` for lambda=2/3/5 and epsilon threshold diagnostic.
- Added tests and validation documentation of STOP/CONTINUE, oracle-current disagreement, target/pulse and historical virtual first-STOP analysis.
- No live selection/STOP policy, configuration, hardware or trained model changes.


### Offline validation add-on (2026-10-13; no runtime version bump)
- Two-stage STOP sensitivity and physical-coordinate target overlay scripts, tests, and docs. No changes to live inference, selector, policy configuration, or PC1.
