# Offline target reachability / controllability evaluation v2

v1の評価内容は変更せず，長時間処理の進捗を`tqdm`で確認できるようにした版です．

## tqdm progress

通常実行では，`Prepare targets`，`Empirical support`，`parameter_transition_A: 27-action sweep`，`shock_transition_B: 27-action sweep`のprogress barを表示します．Transitionのbarには現在のpulseと処理済みstate数も表示します．27候補を9候補ずつ処理する内部chunkにはbarを出しません．

無効化する場合は`--no-progress`を追加してください．`tqdm`が未導入なら`pip install tqdm`で導入できます．

---

## Where this code belongs

Place the complete `offline_evaluation` folder directly under the PC2 runtime repository:

```
adaptive_control_runtime/
├─ server.py
├─ adaptive_runtime_config.yaml
├─ runtime/
├─ ...
└─ offline_evaluation/
   ├─ evaluate_target_reachability.py
   ├─ README.md
   └─ configs/
      └─ target_reachability.yaml
```

This is intentionally **not** placed under `laser_processing_ml/transition/`.

The transition repository is responsible for training/evaluating the model itself.
This new evaluation asks a controller-level question:

> Are the fixed desired target morphologies geometrically reachable from observed
> process states, and do the 27 actions provide a useful control landscape?

Therefore it belongs with the adaptive-control runtime.

## What v1 evaluates

### 1. 384 → 256 target preparation

The 384x384 `desired_target_mask.npy` files are treated as immutable master targets.

The script creates nearest-neighbor 256x256 copies under the evaluation output:

```
targets_256/
├─ T_TYPICAL.npy
├─ ...
```

It does not overwrite the 384x384 source files.

`target_conversion_manifest.csv` stores source/prepared SHA256 hashes.

### 2. Empirical support

For every ideal target, the script searches observed p50 masks and reports:

- nearest observed IoU
- nearest observed RMSE
- median / mean IoU of the top-K nearest observed states
- number/fraction of observed states above configured IoU levels

This answers whether a desired target is supported by actual observed morphology,
independent of the Transition model.

### 3. Monotonic geometric feasibility

The processed-mask convention is:

- 1 = processed / removed region
- 0 = unprocessed region

The Transition model is monotonic: processed pixels cannot return to material.

For each current observed state and each desired target the script therefore reports:

- current excess processed pixels outside the target
- current missing target pixels
- `monotonic_feasible_exact`
- feasibility under the runtime over-removal tolerance

If the current state already exceeds a target, exact recovery is physically impossible
for this controller.

### 4. 27-action one-step sweep

The script evaluates both control-side Transition variants by default:

- `action_history_only` runtime mode → Transition A
- `shock_metadata` runtime mode → matching Transition B

For every held-out physical current state:

1. encode the current state once
2. evaluate all 27 E/y/z candidate actions
3. keep next-pulse number fixed to the physical transition stage
4. for Transition B, use the matching observed `z_shock`
5. compare every predicted next state with all eight desired targets

The same action-selection policy as the runtime is used:

- among safe candidates: minimum L1 + Dice-loss cost
- if no action is safe: minimum over-removal, then minimum cost

### 5. Reachability metrics

Per current-state / target pair:

- current cost and RMSE
- exact monotonic feasibility
- safe action count
- no-safe-action flag
- selected action
- selected predicted over/under-removal
- selected-vs-current cost/RMSE gain
- minimum RMSE among any action
- minimum RMSE among safe actions

### 6. Controllability metrics

The following describe whether changing the action actually changes the predicted result:

- cost range across 27 actions
- cost standard deviation across 27 actions
- selected-vs-second safe-action cost margin
- removal-area range across 27 actions
- mean pixelwise standard deviation of the 27 predicted next states
- selected-action frequency / energy / y / z distribution

A target can be morphologically reachable but poorly controllable if all 27 actions
produce almost identical predicted next states.

## Important limitation

This v1 intentionally does **not** perform a synthetic multi-step rollout of Transition B.

A hypothetical second step would require the future `z_shock`, which is not available
until the future laser response is actually observed. Holding the old `z_shock` fixed
would make the rollout look precise without being physically justified.

Therefore v1 is:

- empirical support
- monotonic feasibility
- one-step local controllability over many real held-out states

If these results are acceptable, the next evaluation should be an end-to-end replay
using saved shock observations and State Estimator outputs.

## First run: path validation only

From `adaptive_control_runtime`:

```powershell
python offline_evaluation\evaluate_target_reachability.py `
  --config offline_evaluation\configs\target_reachability.yaml `
  --validate-only
```

Expected checks:

- 8 ideal targets found
- 27 actions found
- Transition A checkpoint found
- Transition B checkpoint found
- matching z_shock cache found
- common held-out transition states found

No model inference is run.

## Full run

```powershell
python offline_evaluation\evaluate_target_reachability.py `
  --config offline_evaluation\configs\target_reachability.yaml
```

For a quick GPU smoke test, temporarily set:

```yaml
states:
  max_states_per_pulse: 10
```

After that passes, restore:

```yaml
states:
  max_states_per_pulse: null
```

to evaluate the complete held-out test split.

## Main outputs

Each run creates:

```
F:/261006_adaptive_control_data_test/offline_evaluation/
└─ target_reachability_YYYYMMDD_HHMMSS/
   ├─ validation_report.json
   ├─ resolved_evaluation.json
   ├─ target_conversion_manifest.csv
   ├─ target_contact_sheet_256.png
   ├─ targets_256/
   ├─ empirical_support_summary.csv
   ├─ per_state_target.csv
   ├─ target_model_summary.csv
   ├─ target_model_by_pulse.csv
   ├─ selected_action_frequency.csv
   ├─ target_reachability_controllability_report.csv
   └─ plots/
```

Start interpretation from:

`target_reachability_controllability_report.csv`

and then inspect `target_model_by_pulse.csv` for stage dependence.

## Production session root

Do not create the production experiment root yet.

Continue offline work in:

`F:/261006_adaptive_control_data_test`

When offline target screening, stop policy, and hardware dry-run settings are frozen,
create a new date/experiment-scoped production root and acquire that experiment's
calibration into it.

With the current runtime storage convention, a production root will look like:

```
F:/<production_session>/
├─ resources/
│  ├─ action_sets/
│  │  └─ action27_v1/
│  ├─ target_sets/
│  │  └─ ideal8_v1/
│  └─ calibration_sessions/
└─ experiments/
```

The eight ideal targets are now a versioned resource and should not be edited after
model results are inspected.  The 384x384 master files should be preserved, while
the runtime target set should use the generated 256x256 copies.
