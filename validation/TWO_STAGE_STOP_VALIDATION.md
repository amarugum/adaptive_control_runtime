# Two-stage STOP policy: offline validation (research-only)

## Model

- Current-state Estimator: C_k = |S_k∩T|/|T|, O_k=|S_k\T|/|T|.
- Candidate a predicts C_(k+1)(a), O_(k+1)(a). Define ΔC = max(0,C_next−C_k), ΔO=max(0,O_next−O_k).
- Utility: U(a)=ΔC(a)−λΔO(a). Equivalent cost L(a)=−U(a).

Decision order (priority is explicit):

1. Current-state over `O_k >= O_max`: STOP `current_over_limit`.
2. Current-state completion `C_k >= C_target`: STOP `target_reached` (provided step 1 did not fire).
3. Transition predicts 27 actions; require `O_next(a) < O_max` (strictly `<` to match current runtime). If none, STOP `no_safe_action`.
4. Choose argmax U among safe actions. If max U <= 0, STOP `nonpositive_utility`; otherwise CONTINUE with chosen action.
5. PC1 retains independent physical execution authority, and laser firing is never retried automatically. This script does not alter the live runtime.

Noteworthy: 4 is evaluated **after** eliminating unsafe actions. A globally highest-utility unsafe action must NEVER be proposed.

## Scope / limits

- Input contains all 27 estimated-current and oracle-current Transition predictions cached for 99 validation holes (891 fixed-plan states) and 8 targets across 4 models.
- `oracle-current` substitutes a measured **current** shape into the trained Transition, so predicted next shapes for unexecuted actions are NOT measured ground truth.
- `historical_first_stop` is the first STOP along previously collected fixed-plan states; it is NOT closed-loop rollout.
- This does not calibrate over limits physically nor give a laser-safety guarantee. Especially note systematic Estimator underestimation of over.
- Fixed stop cutoff per step `max_control_steps` not hit in the p5–45 replay and is unaffected by this tool.
- The validation data is for choosing parameters; after freeze, test split must only be evaluated once as a held-out check.

## Example

Run from `adaptive_control_runtime`:

`python validation\combined_stop_policy_validation.py --cache "F:\261006_adaptive_control_data_test\offline_evaluation\policy_sensitivity_20261008_202409.zip" --output "F:\261006_adaptive_control_data_test\offline_evaluation\two_stage_stop_val" --lambdas 2 3 5 --completion-thresholds 0.85 0.90 0.95 --over-limits 0.05 0.10 0.15 0.20`

All operations are read-only with respect to the cache. No GPU or trained weights are loaded.

Outputs: summary_by_model.csv; summary_by_target.csv; summary_by_pulse_target.csv; historical_first_stop.csv; historical_first_stop_summary.csv; analysis_metadata.json.

STOP reasons in each summary are fractions of all state-target cases. `selected_action_oracle_over_limit_rate_among_continue` is conditional on estimated CONTINUE, **not** on all cases. Higher over limit can lower estimated/oracle mismatch simply by relaxing the same criterion on both sides; it is NOT evidence of improved physical safety.
