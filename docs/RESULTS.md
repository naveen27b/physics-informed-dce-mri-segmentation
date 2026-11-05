# Result audit

This file records the recovered evidence behind the public-facing result table. It avoids combining incompatible protocols and distinguishes selected development results from untouched validation.

## Dataset600 internal cross-validation

The strongest complete five-fold artifact is:

```text
nnUNetTrainerResEncM_TopKCE_Parallel__nnUNetResEncUNetPlans__3d_fullres_iso/
  crossval_results_folds_0_1_2_3_4/summary.json
```

Recovered nnU-Net aggregate:

- cases: 1,200 out-of-fold predictions;
- mean Dice: 0.8012963849934753;
- mean IoU: 0.703765062465001;
- mean false-negative voxels per case: 6,763.59;
- mean false-positive voxels per case: 4,409.17.

This is an internal cross-validation estimate, not an external test result.

The separately documented standard Dice+CE 1,000-epoch run reports mean five-fold Dice 0.7940. Its fold means were 0.7782, 0.8159, 0.7767, 0.8047, and 0.7947.

## MAMA-MIA held-out challenge cohort

Recovered aggregate analysis contains 306 cases. Selected configurations include:

| Configuration | Mean Dice | Mean normalized HD | Performance score |
|---|---:|---:|---:|
| Mean ensemble, folds 1+3+4 | 0.78015 | 0.07684 | 0.85165 |
| Mean ensemble, folds 1+3+4 + proximity PP | 0.78100 | 0.07456 | 0.85322 |
| Mean ensemble, folds 1+4 | 0.77641 | 0.08160 | 0.84740 |

Because multiple configurations were compared on this cohort, the best row is a selected result and should not be described as untouched independent validation. The manuscript-designated submission configuration reports Dice 0.7773 and ranking score 0.8597.

## External cohorts

Recovered Dataset600 reports contain:

- YUN: 98 cases, mean Dice 0.8336, median 0.8699, single fold trained for 100 epochs, no YUN fine-tuning;
- IEDH: 31 cases, mean Dice 0.7281, median 0.8258, zero-shot, missing late phase represented by a zero washout channel.

These are artifact-reported results. The source cohorts and original compute host were unavailable for an independent rerun during repository reconstruction.

## Claims intentionally not used

The résumé values “78.98% Dice on 306 cases”, “78.84% on 100 YUN cases”, and “82.00% on 31 IEDH cases” could not all be mapped to one consistent recovered protocol. In particular, 82% closely matches the IEDH median rather than its mean. They are therefore omitted from the headline README pending provenance clarification.
