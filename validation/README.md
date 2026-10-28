# V1.8 pre-experiment validation

## AMP / FP32 final-action equivalence

This test checks the exact question needed before enabling Transition AMP in the live controller:

> Does AMP change the final CONTINUE/STOP decision or selected action relative to FP32?

The test uses runtime-like **B=1** Transition inference, the candidate chunk size configured for each mode, the exact runtime scoring function, and all 8 prepared 256×256 reachability targets.

By default it samples 20 common test states at each current pulse p5, p10, ..., p45, i.e. up to 180 physical states. Each state is tested against all targets.

Run from `adaptive_control_runtime`:

```powershell
python validation\validate_amp_action_equivalence.py `
  --runtime-config adaptive_runtime_config.yaml
```

The script automatically uses the newest:

```text
<session_root>/offline_evaluation/target_reachability_*/targets_256
```

or specify another directory with `--target-root`.

Outputs are written under:

```text
<session_root>/offline_evaluation/amp_action_equivalence_YYYYMMDD_HHMMSS/
```

Main files:

- `validation_report.json`
- `summary.json`
- `summary.csv`
- `per_case_equivalence.csv`

### PASS criterion

PASS is intentionally strict:

- zero AMP/FP32 CONTINUE-vs-STOP mismatches
- zero selected-action mismatches among cases where both precisions CONTINUE

Safe-candidate count, Top-3 set, cost differences, candidate safe-flag differences, and prediction differences are also logged diagnostically.

If the script exits with code 2 / FAIL, keep `transition_amp: false` for the affected mode rather than accepting a few-millisecond speed gain.

## V1.8 no-safe policy

The live runtime now fails closed:

```text
safe_candidate_count == 0
    -> decision = stop
    -> stop_reason = no_safe_action
    -> best_action = null
```

`candidate_scores.csv` and candidate predictions are still saved for diagnosis. The offline selector can still rank the least-over-removal candidate, but the live engine never sends that fallback action to PC1.

### After PASS

The shipped V1.8 config intentionally keeps `transition_amp: false` until this gate passes.
AMP validation is retained for diagnostics, but V1.10 production intent is FP32 for all four modes. Do not enable AMP solely from this validator:

```yaml
models:
  action_history_only:
    performance:
      estimator_amp: false
      transition_amp: true
      candidate_chunk_size: 12

  shock_metadata:
    performance:
      estimator_amp: false
      transition_amp: true
      candidate_chunk_size: 27
```

Then restart `server.py` and run the ordinary PC2 smoke tests again. If the equivalence test FAILs for one mode, leave `transition_amp: false` for that mode; keep its benchmarked candidate chunk size.


## V1.10 four-model replay

`replay_end_to_end_offline.py` now defaults to all four configured modes and evaluates only physical states common to all four pipelines. Run the four-model FP32 GPU benchmark and apply its chunk recommendations before the full replay.


## V1.12 overall / over-removal sensitivity analysis

Run after the four-model end-to-end replay:

`python validation\policy_sensitivity_analysis.py --runtime-config adaptive_runtime_config.yaml --max-states-per-pulse 5`

Then run the full analysis without `--max-states-per-pulse`. The analysis keeps continuous overall/over metrics separate from the hard STOP policy and uses a reusable candidate cache so every threshold pair is compared on identical model predictions. See `POLICY_SENSITIVITY.md`.

## Offline lambda and continuation-value add-ons (no live behavior changes)

- `lambda_action_sensitivity.py`: validation of the action-scoring tradeoff `L=lambda*DeltaO-DeltaC` using existing cache.
- `continuation_value_sensitivity.py`: value-of-continuing diagnostic `max_a (DeltaC-lambda*DeltaO)>epsilon`, estimating STOP/CONTINUE, oracle-current agreement, target/pulse breakdown, small-positive utility and virtual first STOP along historical fixed-plan sequences. See `CONTINUATION_VALUE_SENSITIVITY.md` and `CONTINUATION_VALUE_ANALYSIS_REPORT.md`.
- Do not interpret estimated or oracle-current CONTINUE as live laser firing permission. The main runtime and hard gates are unchanged.


## Two-stage STOP / actual-target overlay validation (offline only)

See [TWO_STAGE_STOP_VALIDATION.md](TWO_STAGE_STOP_VALIDATION.md) and [TARGET_OVERLAY_REVIEW.md](TARGET_OVERLAY_REVIEW.md).

Scripts: `combined_stop_policy_validation.py` (cached estimated/oracle comparison) and `preview_target_completion_overlays.py` (needs actual local mask files). Neither modifies the online runtime/config or laser control.
