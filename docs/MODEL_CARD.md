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

1. Pre-contrast anatomy after percentile scaling based on the pre-contrast image.
2. Relative enhancement.
3. Positive post-contrast subtraction.
4. Positive washout slope.

## Training data

The recovered experiment used 1,200 MAMA-MIA training cases across DUKE, I-SPY1, I-SPY2, and NACT. MAMA-MIA is a multi-institutional dataset. The repository does not redistribute raw MRI volumes or label files; it includes one attributed, single-slice visualization from DUKE_019 in the README.

- Project and challenge information: [MAMA-MIA website](https://www.ub.edu/mama-mia/).
- Dataset access: follow the [official MAMA-MIA download instructions](https://github.com/LidiaGarrucho/MAMA-MIA#downloading-the-dataset) for Synapse and XNAT access. The dataset is hosted on [Synapse (syn60868042)](https://www.synapse.org/Synapse:syn60868042).
- Dataset citation: Garrucho et al., [MAMA-MIA: A large-scale breast cancer MRI dataset](https://doi.org/10.1038/s41597-025-04707-4), *Scientific Data* (2025).

Please credit the MAMA-MIA dataset and its publication when using these results. Dataset access, permitted use, and redistribution are governed by the dataset providers and source collections; review their current terms and citation requirements before reuse. This repository does not redistribute the data.

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
