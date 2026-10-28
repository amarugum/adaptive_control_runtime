# Continuation-value sensitivity (offline-only add-on)

## Purpose

Analyze if an extra minimum-improvement threshold `epsilon` is necessary when using:

- Candidate score `U(a) = Delta C(a) - lambda * Delta O(a)`.
- The best action is `argmax_a U_est(a)` (ties broken by the lowest action index).
- A **diagnostic** STOP occurs when `max_a U_est(a) <= epsilon`.
- STOP has reference utility 0. For `epsilon>0`, we explicitly demand an additional positive predicted improvement.

`C` and `O` are target-area-normalized fractions; action deltas refer to a 5-pulse transition. No image translation or depth partitioning is used.

## Run from the PC2 adaptive_control_runtime directory

```powershell
python validation\continuation_value_sensitivity.py `
  --cache "F:\261006_adaptive_control_data_test\offline_evaluation\policy_sensitivity_20261008_202409.zip" `
  --output "F:\261006_adaptive_control_data_test\offline_evaluation\continuation_value_val"
```

Default: `--expected-split val --lambdas 2 3 5 --epsilons 0 0.001 0.0025 0.005 0.01 0.02 --small-utility-cutoffs 0.001 0.005 0.01`.

`--cache` may be the ZIP containing the policy cache or the extracted `.npz` with matching `policy_cache_metadata.json` alongside it. CPU and numpy only. No inference/GPU, runtime configuration update, or PC1 interaction occurs.

## Outputs

- `continuation_summary_by_model.csv`: the four modes, lambda/epsilon grid, STOP/CONTINUE, agreement, false CONTINUE/STOP, STOP current C/O mean and quartiles, micro-improvement-only STOP fraction, and predicted chosen-action fractions.
- `continuation_summary_by_target.csv`: same diagnostics for each of eight targets.
- `continuation_summary_by_pulse.csv`: same diagnostics for each current pulse (p5 ... p45).
- `small_positive_utility.csv`: frequency of `0 < best utility <= cutoff`, oracle-current proxy for those near-zero *net benefit* cases.
- `small_progress_analysis.csv`: cases in which the selected `Delta C` itself is small despite positive net benefit; also cases where all 27 actions have small `Delta C`. This is different from small net utility.
- `historical_first_stop_cases.csv`: one **virtual first STOP** per hole-target, model, lambda, epsilon; current pulse and STOP C/O values or `none_in_p5_to_p45`.
- `historical_first_stop_summary.csv`: frequency and distribution of virtual first STOP across hole-target series.
- `resolved_continuation_sensitivity.json`: all assumptions and provenance.

## Interpretation safeguards

1. **Different denominators**: STOP and false CONTINUE/STOP fractions are normalized by all independent state-target cases. A selected-action-oracle-nonpositive fraction is conditional on estimated CONTINUE.
2. Estimated/oracle decision agreement compares each side's **optimal action under the same lambda**; it is not a truth label for safety. We separately check the oracle-current utility of the **action estimated-current actually selects**.
3. Oracle-current means replacing current shape with a measured mask and reusing the **same trained Transition** (and relevant shock latent). Counterfactual future actions have no measured outcome. High agreement can reflect a mostly-CONTINUE data distribution.
4. Each physical state is an observation under a **historical fixed plan**. The `first_stop` report finds the first STOP along that observed trajectory; it is NOT a closed-loop rollout. Post-STOP states under an alternative action policy are unknown.
5. **No hard over-removal limit, completion target, max-steps stop, stagnation check, action feasibility check, PC1 safety gate, or estimator uncertainty margin is applied.** No produced CONTINUE is permission to fire a laser.
6. Policy selection must happen on validation only. A test split should be reserved for a single final evaluation after freezing all policy settings.
7. Some floating point monotonicity violations smaller than 1e-5 are clipped to zero; greater violations cause a fail. Keep mask semantics as `1=processed`.

## Criteria for adding epsilon

- Inspect whether a nontrivial proportion of cases has tiny *positive* `U` at epsilon=0.
- Compare whether epsilon reduces bad decisions against the oracle-current model proxy rather than only inflating STOP.
- Check which targets and pulse stages it affects, and whether the extra STOP happens at acceptable current C/O.
- A positive epsilon does **not** certify an engineering safety limit, solve underestimated over-removal, or replace a physically defined safety gate.

This validation script does not edit `runtime/selector.py`, `server.py`, `adaptive_runtime_config.yaml`, or any hardware control code.
