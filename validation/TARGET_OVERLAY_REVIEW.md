# Eight-target physical-coordinate overlay review (local masks required)

## Purpose

Compare 8 target masks against actual processed-hole masks in the **unaligned, original crop coordinate frame**. This differs from display-only horizontal centering sometimes used by `ideal_target_designer.py`. We do not shift observed holes to maximize IoU; y-axis laser motion is a real machining variable.

The 8 real `T_*.npy` files and observed hole masks are not contained in the source-code ZIP. You must provide them via paths on PC2. Avoid using the sample `T001.npy`, which is a smoke-test target.

Default target files were recorded under:

`F:\261006_adaptive_control_data_test\offline_evaluation\target_reachability_20261008_110900\targets_256`

## Option 1: target vs three actual observed validation holes

Run from `adaptive_control_runtime`:

`python validation\preview_target_completion_overlays.py --target-root "F:\261006_adaptive_control_data_test\offline_evaluation\target_reachability_20261008_110900\targets_256" --data-root "F:\260910_ml_data" --hole-uids h0001 h0008 h0020 --pulse 50 --output "F:\261006_adaptive_control_data_test\offline_evaluation\target_overlay_review"`

Change `--hole-uids` as needed. All eight targets share the same three example holes, enabling comparable images. Using p50 is appropriate for final geometry illustration, but does not represent stop-at-p25 etc. If `target.npy` is C×H×W, channel 1 is the measured cumulative processed mask (1=removed).

## Option 2: exact three original masks used to design each target

If `ideal_target_designer.py` output directories containing `ideal_target.json` are available, add:

`--designer-root "F:\...\ideal_target_designer_outputs" --use-designer-sources`

The script reads the `sources[].path`, `channel`, `threshold`, and `source_alignment_for_display` keys from each target's JSON. The actual source masks are **never translated during this review**, even if that flag is true. This allows auditing whether the target was geometrically idealized from display-centered sources.

## Output

- `target_observed_strict_overlays.png`: each row contains one target plus up to 3 measured masks. Blue=target only (under), red=processed outside (over), green=intersection.
- `target_observed_metrics.csv`: actual measured C, O, IoU and absolute pixel counts for each target–sample pair.
- `overlay_provenance.json`: paths and any absent masks; includes alignment-flag provenance.

Physical meaning of comparison is limited by calibration/orientation/crop consistency. If a target was centered on the top opening rather than the experiment's physical hole origin, that is a target-definition issue, not a reason to post-align predicted images.

## Completion threshold check

Inspect for example samples in bands C=0.85–0.90, 0.90–0.95, >=0.95, and note whether remaining blue pixels are near the bottom, on the sidewalls, or dispersed. The values alone cannot show if target shape is acceptable. To sample those bands systematically, provide additional measured masks across p5..p50 or a full local dataset; this script deliberately does not manufacture missing regions.
