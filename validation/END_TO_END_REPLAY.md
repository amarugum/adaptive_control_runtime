# End-to-end offline replay

This validation is the next gate after target reachability, four-model FP32 GPU chunk benchmarking, and runtime setting freeze.

It uses the **actual PC2 MLRuntime path** on held-out test states:

1. raw `shock_input.npy` / executed action history → State Estimator
2. `S_hat_k >= 0.5` and, for shock mode, live `z_shock`
3. matching Transition A/B → all 27 candidate next states
4. current-state stop policy + safe-candidate selector + `no_safe_action -> STOP`
5. comparison with an oracle-current replay using the measured current mask

The oracle-current replay uses the **same live `z_shock`** for Transition B. This isolates the effect of current-shape estimation error; it is not a separate oracle shock-latent experiment.

## Default execution

From `adaptive_control_runtime/`:

`python validation\replay_end_to_end_offline.py --runtime-config adaptive_runtime_config.yaml`

Defaults:

- modes: `metadata_only`, `action_history_only`, `shock_metadata`, `shock_action_history`
- split: `test`
- current pulses: 5,10,...,45
- all common held-out states
- all prepared targets in the latest reachability run
- exclude the known 26 white-band artifact holes using `validation/artifact_exclusions_20261008.csv`
- runtime performance settings exactly as configured in `adaptive_runtime_config.yaml`

For a quick smoke run first:

`python validation\replay_end_to_end_offline.py --runtime-config adaptive_runtime_config.yaml --max-states-per-pulse 5`

Then run the full evaluation without `--max-states-per-pulse`.

## Outputs

`<session_root>/offline_evaluation/end_to_end_replay_YYYYMMDD_HHMMSS/`

- `state_estimator_metrics.csv`
- `actual_action_next_state_metrics.csv`
- `control_policy_replay.csv`
- `summary_state_estimator.csv`
- `summary_actual_action_next_state.csv`
- `summary_control_by_target.csv`
- `summary_control_by_pulse.csv`
- `resolved_replay.json`

## Interpretation

`summary_state_estimator.csv` measures the State Estimator against the measured current state.

`summary_actual_action_next_state.csv` evaluates the full State Estimator → Transition chain for the **action that was actually executed in the dataset**, because that is the only next-state counterfactual with physical ground truth.

`summary_control_by_target.csv` compares the live estimated-state controller with an oracle-current controller. Important fields are:

- `decision_match_fraction`
- `selected_action_match_fraction_on_both_continue`
- `top3_set_match_fraction_on_both_continue`
- estimated/oracle `no_safe` and stop fractions
- safe-candidate count difference

There is intentionally no hard scientific PASS/FAIL threshold in the script. The thresholds should be decided after observing the held-out results, rather than chosen post hoc to force a pass.


## Four-model comparison

V1.10は4モデルで同じphysical statesを使うため，state/transition sample mapsの共通集合だけを評価します．これによりmode間の比較で評価stateがずれることを防ぎます．

主比較候補だけに絞らず，4モデルすべてについてState Estimator精度，actual-action next-state精度，oracle-currentとのdecision/action一致を出力します．最終的に本番実験で何モデルを残すかはこの結果を見て決定します．


## v1.11 setup progress

The replay now prints/refreshes setup stages before the state-level tqdm loop. In particular, `build_g1_samples` / `build_g5_samples` can internally load thousands of metadata `.npy` files; these calls now display an elapsed-time heartbeat. Compatible sample maps are cached across modes to avoid repeating the same expensive scan.
