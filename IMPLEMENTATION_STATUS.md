# Implementation status

## Policy v1.0 status — 2026-10-13

- **Live PC2 policy implemented**: λ=3, C_target=.90, O_max=.10, no epsilon. `L<0` is required for continuation after safety filtering.
- Stage 1 and Stage 2 STOP implemented, safe candidate condition is strict `<.10`, hardware authority remains on PC1.
- Unit/integration tests use simulated model results only; PC1 compatibility, Windows GPU model smoke, shadow and human-approved tests remain pending.
- **Older status sections below refer to past versions, and are not current production policy.** See `POLICY_V1_20261013.md`.


## V1.12 current state
- Four-model end-to-end replay has been completed and motivates explicit sensitivity analysis of `overall_removal_target` and `over_removal_limit` before the live policy is frozen.
- `overall` and `over` remain continuous morphology metrics for scientific comparison; hard thresholds are treated as a separate runtime safety/completion policy.
- `validation/policy_sensitivity_analysis.py` performs one fixed GPU candidate-prediction pass, caches all 27-candidate metrics for estimated-current and oracle-current branches, and sweeps threshold pairs CPU-side.
- No automatic threshold recommendation is encoded. The next gate is to review aggregate, target-specific, and pulse-specific sensitivity, then freeze the stop policy before production calibration / Shadow ML.
- Live runtime behavior is unchanged: `no_safe_action -> STOP`, FP32 for all four models, `candidate_chunk_size=6`.

## V1.10 current state
- All four scientific runtime modes are first-class validation targets: `metadata_only`, `action_history_only`, `shock_metadata`, `shock_action_history`.
- All four use FP32 for State Estimator and Transition in production intent.
- Known benchmarked chunks remain 12 (`action_history_only`) and 27 (`shock_metadata`). The other two modes temporarily use chunk 9 until the four-model FP32 benchmark is run.
- Live `no_safe_action` remains fail-closed STOP.
- End-to-end offline replay defaults to all four modes and a common held-out physical-state set.

## Current pre-experiment gate
1. Run the four-model FP32 GPU benchmark.
2. Apply all four recommended chunks with `benchmarks/apply_fp32_recommendations.py`.
3. Re-run the four-model model/decision smoke tests and verify Windows Shared GPU Memory remains near zero.
4. Run a small four-model end-to-end replay (`--max-states-per-pulse 5`).
5. Run the full four-model held-out replay and compare estimator accuracy, actual-action next-state accuracy, and oracle-current policy agreement.
6. Only after reviewing those results, freeze which model set enters the real-laser shadow experiment.

## V1.8 implemented
- PC2 HTTP runtime with idempotent request/ACK handling and separated temporary/persistent storage.
- Four State Estimator / Transition modes remain supported.
- Per-model GPU inference settings are configurable through each model's `performance:` block.
- Benchmarked primary-mode settings are included:
  - `action_history_only`: Estimator AMP=false, candidate chunk=12; benchmark recommends Transition AMP=true, but shipped config keeps it false until equivalence PASS.
  - `shock_metadata`: Estimator AMP=false, candidate chunk=27; benchmark recommends Transition AMP=true, but shipped config keeps it false until equivalence PASS.
- Effective GPU inference settings are written into each decision response for provenance.
- Live no-safe behavior is now fail-closed: zero safe candidates -> STOP (`no_safe_action`) and `best_action=null`.
- Legacy live `min_over_then_cost` policy is rejected at config load. The diagnostic ranking remains available only to offline analysis tools.
- Added runtime-like B=1 AMP-vs-FP32 final-action equivalence validation.
- Added GPU benchmark v2 under `benchmarks/`.

## Previously validated on PC2 hardware
- CUDA / RTX 2080 SUPER environment.
- All four State Estimator + matching Transition pipelines: 4/4 PASS.
- Real saved ORCA k01 -> online postprocessing -> 5x3 shock -> both shock State Estimators -> matching Transition B -> 27 candidates: 2/2 PASS.
- Full HTTP STOP and CONTINUE branches in all four modes.
- Request idempotency, ACK, and temporary inbox cleanup.
- Dedicated Ethernet PC1 -> PC2 bulk transfer after moving PC2 to Intel I219-V.
- Full real-hardware laser-inhibited loop through acquisition, transfer, PC2 decision, ACK, and return-to-waiting.
- Offline target reachability / controllability evaluation for eight ideal targets.
- RTX 2080 SUPER B=1 GPU benchmark for Estimator AMP and Transition AMP/chunk combinations.

## Historical V1.8 gate
The AMP-equivalence gate was completed and produced rare final-action/safe-count changes. Therefore V1.10 keeps AMP disabled for all four scientific modes.

## V1.9 validation gate

Current runtime GPU policy is frozen provisionally as:

- `action_history_only`: estimator AMP=false, transition AMP=false, candidate chunk=12
- `shock_metadata`: estimator AMP=false, transition AMP=false, candidate chunk=27

AMP was not enabled because held-out action-equivalence validation produced rare selected-action changes and safe-count changes near decision boundaries, while the speed benefit was only a few milliseconds.

The next required gate is now implemented as `validation/replay_end_to_end_offline.py`.
Run a small smoke replay first, then the full held-out replay. Review State Estimator accuracy, observed-action next-state accuracy, and estimated-state vs measured-current controller agreement before moving to real-laser shadow operation.

## Validation add-on (2026-10-11; no production freeze)
- `lambda * DeltaO - DeltaC` remains a proposed action-ranking method only.
- `validation/continuation_value_sensitivity.py` evaluates the proposed `max utility > epsilon` continuation gate offline using the 2026-10-08 validation candidate cache.
- Preliminary result: epsilon=0.005 creates few additional STOPs (<1 percentage point for each mode/lambda=2,3,5) and does not improve oracle-current decision match for shock models; epsilon=0 remains a baseline, NOT an approved live setting.
- Hard physical over safety, completion policy, stalled progress safeguards, and PC1 safety authority remain separate and are NOT replaced by this add-on.


## Pending policy validation (2026-10-13)
- Action score candidate: L=λ Δover−Δoverall; λ=2,3,5 not frozen.
- Completed/over STOP thresholds not frozen (exploratory comparison 0.85/0.90/0.95 and 0.05/0.10/0.15/0.20).
- Current/candidate two-stage STOP is OFFLINE DIAGNOSTIC only. Live runtime remains legacy L1+Dice with provisional 0.95/0.05.
- Actual 8 target vs measured-mask overlay requires local target npy and observed target.npy paths; project source ZIP did not contain those masks.
- Human-approved laser experiment is NOT authorized by these analytical scripts. Continue stage-gated safety protocol.
