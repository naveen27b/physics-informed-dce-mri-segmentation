# Model card: Dataset600 MAMAMIAPhysics

## Intended use

Research on multi-phase breast DCE-MRI tumor segmentation, preprocessing robustness, domain shift, and post-processing. The model is not validated for diagnosis, treatment planning, screening, or autonomous clinical use.

## Architecture

- framework: nnU-Net v2;
- configuration: 3D full resolution, 1 mm isotropic target spacing;
- network: ResidualEncoderUNet / ResEncM;
- input: four channels derived from registered DCE phases;
- output: binary tumor segmentation;
- evaluation variants: five-fold ensembles and connected-component post-processing.

## Input channels

1. pre-contrast anatomy after pre-derived percentile scaling;
2. relative enhancement `(post1 - pre) / max(pre, 0.01)`;
3. positive subtraction `post1 - pre`;
4. positive washout slope `(temporal_peak - last_phase) / elapsed_phase_steps`.

## Training data

The recovered experiment used 1,200 MAMA-MIA training cases across DUKE, I-SPY1, I-SPY2, and NACT. Dataset access and redistribution are governed by the original providers; this repository contains no source images or labels.

## Checkpoint status

The local archive contains five `checkpoint_final.pth` files of approximately 815.5 MB each for both the standard 1,000-epoch and TopKCE model variants. They are excluded because:

- ordinary GitHub file limits are unsuitable for multi-gigabyte model releases;
- redistribution terms and institutional ownership must be confirmed;
- a usable release should include hashes, exact nnU-Net version, trainer code, plans, and a tested download mechanism.

A future release should use a dedicated model registry such as Zenodo or Hugging Face, subject to authorization.

## Limitations

- performance varies across institutions, lesion sizes, and acquisition protocols;
- ROI cropping may depend on metadata or a separate breast segmentation step;
- the washout channel assumes temporally ordered, spatially registered phases;
- missing phases change the available kinetic information;
- aggregate external results were recovered from reports but not independently rerun;
- chosen post-processing and ensemble configurations may be development-selected.
