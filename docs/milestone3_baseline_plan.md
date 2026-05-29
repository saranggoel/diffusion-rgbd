# Milestone 3 Baseline Testing Plan

Deadline: May 29, 2026

## Goal

Produce preliminary quantitative and qualitative results for robust RGB-D semantic segmentation on NYUv2. The Milestone 3 story should answer:

- Does adding depth help over RGB-only segmentation?
- How badly do missing and corrupted modalities hurt each baseline?
- Does the proposed diffusion/consistency design improve robustness beyond standard RGB-D fusion?
- What is working, what is not, and what should we do next?

## Current Repo Status

The local checkout currently contains no project files or commits, and the configured remote has no visible branches or tags. This plan therefore assumes we either need to restore the project code from another location or quickly scaffold the minimum evaluation pipeline before running experiments.

## Minimum Viable Results for Milestone 3

If time is tight, prioritize these results first:

1. RGB-only baseline on NYUv2 validation/test split.
2. Depth-only baseline on the same split.
3. Early-fusion RGB-D baseline using channel concatenation.
4. Missing/corrupted modality evaluation for the above three baselines.
5. One qualitative figure with RGB, depth, corruption, ground truth, and predictions.

This is enough to support a preliminary-results slide even if the full diffusion model is not trained yet.

## Full Baseline Matrix

| Model | Purpose | Milestone Priority |
| --- | --- | --- |
| RGB-only segmentation | Lower bound for no depth available | Must have |
| Depth-only segmentation | Lower bound for no RGB available | Must have |
| Early-fusion RGB-D | Simple channel-concat baseline | Must have |
| Transformer RGB-D fusion / CMX-style | Strong fusion baseline without diffusion | Should have |
| Diffusion without subset/corruption consistency | Ablation isolating consistency losses | Stretch |
| Full model | Diffusion + subset consistency + corruption consistency | Stretch / main method if available |

## Evaluation Conditions

Evaluate every trained model under the same input conditions:

| Condition | Description |
| --- | --- |
| Clean RGB-D | RGB and depth both clean |
| RGB only | Depth missing or zero/masked |
| Depth only | RGB missing or zero/masked |
| Corrupted RGB + clean depth | Tests RGB robustness |
| Clean RGB + corrupted depth | Tests depth robustness |
| Corrupted RGB + corrupted depth | Tests worst-case robustness |

For corruption severity, use at least two levels for Milestone 3:

- Mild: small blur/noise/occlusion or shallow depth holes.
- Severe: stronger blur/noise/occlusion or larger invalid depth regions.

If possible, reserve at least one held-out corruption type that was not used in training.

## Corruption Protocol

RGB corruptions:

- Gaussian blur.
- Gaussian noise.
- Low light / brightness reduction.
- Random occlusion patches.

Depth corruptions:

- Hole injection / invalid-pixel masks.
- Additive depth noise.
- Structured missing bands or patches.
- Reflective/transparent-object style missing regions approximated by masks.

Keep corruption generation deterministic with fixed seeds so results are reproducible.

## Metrics

Primary metric:

- mIoU.

Secondary metrics:

- Pixel accuracy.
- Mean class accuracy.
- mIoU drop from clean RGB-D:
  `drop(condition) = mIoU(clean RGB-D) - mIoU(condition)`.
- Robust average:
  average mIoU across all missing/corrupted settings.

The most useful table for Milestone 3 is:

| Model | Clean RGB-D | RGB only | Depth only | RGB corrupt | Depth corrupt | Both corrupt | Avg Robust | Max Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB-only | TBD | TBD | N/A | TBD | N/A | TBD | TBD | TBD |
| Depth-only | TBD | N/A | TBD | N/A | TBD | TBD | TBD | TBD |
| Early fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Diffusion ablation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full model | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Experiment Execution Plan

1. Restore or scaffold the codebase.
   - Confirm dataset loader, model definitions, train script, eval script, config system, and output directory structure.
   - Confirm NYUv2 path and split files.

2. Lock the evaluation protocol.
   - Fix image resolution, label mapping, ignored label id, batch size, and preprocessing.
   - Implement one shared evaluator used by all baselines.
   - Implement deterministic corruption transforms with severity settings.

3. Train or run available checkpoints.
   - First run RGB-only, depth-only, and early-fusion RGB-D.
   - Use identical optimizer, schedule, crop size, batch size, and training epochs where possible.
   - Log validation mIoU per epoch and save best checkpoints.

4. Evaluate all baselines across conditions.
   - Run one command per model across all conditions.
   - Save a CSV/JSON with per-condition metrics.
   - Save qualitative predictions for a fixed set of examples.

5. Build Milestone 3 figures.
   - Main quantitative table: mIoU by model and condition.
   - Robustness bar plot: average robust mIoU.
   - Drop plot: mIoU drop from clean RGB-D by condition.
   - Qualitative grid: RGB/depth/corruption/ground truth/predictions.

## Slide Plan

1. Preliminary Results
   - Show the main table with mIoU across conditions.
   - Highlight the strongest baseline and the biggest failure case.

2. Baseline Comparisons
   - Compare RGB-only, depth-only, and early-fusion RGB-D first.
   - Add transformer/diffusion rows if available.

3. Analysis and Insights
   - Expected observations:
     - RGB-only should be stronger than depth-only for semantic categories.
     - Clean RGB-D should improve over single-modality baselines.
     - Early fusion may fail sharply when one modality is missing because it expects both channels.
     - Depth corruption should hurt geometry-heavy classes more.
     - RGB corruption should hurt texture/appearance-heavy classes more.
   - Discuss whether the observed results match these expectations.

4. Limitations
   - Current results may be preliminary due to limited training time.
   - Synthetic corruptions approximate sensor failures but do not cover every real sensor artifact.
   - If full diffusion is not ready, current results only establish baseline behavior.
   - NYUv2 is relatively small, so variance across splits/seeds may matter.

5. Next Steps
   - Finish transformer fusion baseline.
   - Train diffusion ablation and full model.
   - Add class-wise IoU analysis.
   - Add held-out corruption types.
   - Run multiple seeds or longer training for final results.

## Same-Day Priority Checklist

- [ ] Locate or restore the actual project code.
- [ ] Confirm NYUv2 data availability.
- [ ] Run a smoke test over a tiny validation subset.
- [ ] Produce metrics for RGB-only, depth-only, and early-fusion RGB-D.
- [ ] Generate qualitative examples.
- [ ] Fill the Milestone 3 result table.
- [ ] Add a slide with limitations and next steps.

## Risk Mitigation

If training from scratch cannot finish today:

- Use a smaller validation subset for smoke-test results and label them clearly as preliminary.
- Train for fewer epochs and report trends, not final performance.
- Use public/pretrained segmentation backbones if already available locally.
- Report qualitative predictions plus metric placeholders for baselines still running.
- Make the presentation explicit: "Current milestone establishes evaluation harness and first baseline behavior; full diffusion comparison is next."
