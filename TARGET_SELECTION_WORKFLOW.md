# Target selection and offline validation

## 1. Build candidates from observed shapes

Use `build_target_candidates_from_dataset.py`. The default source is the cumulative processed-mask channel (`target.npy`, channel 1) at pulse 50. The script keeps real observed masks, resizes them to the runtime 256x256 grid with nearest-neighbor sampling, and selects a diverse subset by greedy IoU distance.

These files are candidates only and are not added to the active target registry automatically.

## 2. Obtain representative decision states

Run laser-off/real experiments or use existing PC2 experiment artifacts until several `kXX` directories contain both:

- `state_binary.npy`
- `candidate_probabilities.npy`

Transition candidate probabilities are generated from the current state/action conditions and are independent of the target mask; therefore the same saved predictions can be re-scored against many target candidates.

## 3. Screen candidate targets

Use `screen_target_candidates.py` with the candidate directory and a root containing the saved decision artifacts. Optionally supply the ML data root to report nearest-observed IoU as a support metric.

The script does not collapse the result into a hidden score. It reports transparent criteria such as safe candidate count, no-safe fraction, predicted over-removal, cost, and observed-shape support.

## 4. Validate STOP branches

Run `validate_stop_policy_offline.py` to confirm the current configured thresholds and branch priority before real closed-loop irradiation.

## 5. Register one approved target

Use `register_target_candidate.py`. Existing target files/IDs are never overwritten. Restart `server.py` after registration so `TargetStore` reloads the registry.
