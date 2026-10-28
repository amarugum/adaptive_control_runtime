# Validation 2026-10-08: continuation-value analysis (offline replay)

Results were generated from `policy_sensitivity_20261008_202409.zip` (split=val, 99 holes, 891 physical states, 8 targets, four modes, 27 actions). These are fixed-plan, per-state model-proxy comparisons; NOT closed-loop outcomes.

For the two shock models:

| mode | lambda | epsilon | estimated STOP | est/oracle-current decision match | extra STOP from epsilon |
|---|---:|---:|---:|---:|---:|
| shock_metadata | 2 | 0 | 4.14% | 97.11% | 0.00% |
| shock_metadata | 2 | 0.005 | 4.60% | 96.77% | 0.46% |
| shock_metadata | 3 | 0 | 7.60% | 95.44% | 0.00% |
| shock_metadata | 3 | 0.005 | 8.26% | 95.23% | 0.66% |
| shock_metadata | 3 | 0.01 | 8.84% | 94.91% | 1.23% |
| shock_metadata | 5 | 0 | 13.15% | 92.34% | 0.00% |
| shock_metadata | 5 | 0.005 | 13.71% | 92.05% | 0.56% |
| shock_action_history | 2 | 0 | 4.32% | 97.50% | 0.00% |
| shock_action_history | 3 | 0 | 7.44% | 95.31% | 0.00% |
| shock_action_history | 3 | 0.005 | 8.09% | 95.13% | 0.66% |
| shock_action_history | 3 | 0.01 | 8.68% | 94.84% | 1.25% |
| shock_action_history | 5 | 0 | 12.84% | 91.76% | 0.00% |

At epsilon=0 with lambda=3, the estimated STOP current C/O medians were 0.822/0.123 (`shock_metadata`) and 0.819/0.140 (`shock_action_history`) when pooling independent historical state-target cases. These are *not* the final C/O from controlled holes.

For `shock_metadata` lambda=3 and epsilon=0, STOP probability was 0% at p5--p25, ~0.38% at p30, ~5.93% at p35, ~15.15% at p40, and ~46.97% at p45. Thus observed STOP is predominantly late in the fixed-plan sequence. Target-level STOP varies: at lambda=3, ~2.69% for Rounded-bottom and ~11.78% for Elongated.

Across all four modes and lambda 2/3/5, **the positive-but-small utility interval (0, 0.005] covers only about 0.4--0.9% of state-target cases**, and raising epsilon to 0.005 produces ~0.4--0.9 percentage points additional STOP.

## Decision

These results do **not** demonstrate a need for epsilon>0. A defensible simple baseline is `epsilon=0` during continued offline testing. This is not authorization to switch the live STOP policy. A positive epsilon should only be introduced with a justified minimum benefit threshold or evidence that it helps under noisy/shifted conditions. Larger lambda values themselves increase STOP prevalence and lower the estimated current overall at STOP, so lambda and STOP policy need joint validation.

## Unresolved

- The main safety problem, systematic **over-removal underestimation**, is not addressed by this value-based STOP rule.
- The oracle-current path is not measured next-state ground truth for unexecuted actions.
- A fixed-plan first-STOP time is a **virtual** diagnostic, not a predicted outcome after action adaptations.
- Operational hard safety and completion criteria must be determined before using model-generated actions for automatic laser exposure.
