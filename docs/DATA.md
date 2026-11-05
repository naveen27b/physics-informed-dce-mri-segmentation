# Data preparation

No raw medical-image volumes or label files are distributed in this repository. The README includes one attributed, single-slice visualization derived from MAMA-MIA case DUKE_019 to show the original DCE-MRI input alongside a Dataset600 prediction. The red mask is the model prediction, not the expert reference label.

MAMA-MIA is distributed under CC BY-NC 4.0 at the level of its most restrictive source collection. The figure's input slice is from the DUKE collection. Cite [Garrucho et al. (MAMA-MIA)](https://doi.org/10.1038/s41597-025-04707-4) and [Saha et al. (Duke-Breast-Cancer-MRI)](https://doi.org/10.7937/TCIA.e3sv-re93), indicate that the displayed image is a single-slice derivative with a model prediction overlay, and follow the [MAMA-MIA access and reuse terms](https://github.com/LidiaGarrucho/MAMA-MIA) and [TCIA data usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

The dataset builder expects one folder per case containing registered NIfTI phases. The pre-contrast phase must use the nnU-Net `_0000.nii.gz` suffix; remaining files are interpreted in sorted order as post-contrast phases.

```text
registered_cases/
  CASE_001/
    CASE_001_0000.nii.gz
    CASE_001_0001.nii.gz
    CASE_001_0002.nii.gz
  CASE_002/
    ...
labels/
  CASE_001.nii.gz
roi_json/
  CASE_001.json
```

The recovered ROI JSON schema stores `primary_lesion.breast_coordinates` with `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max`. Review coordinate conventions carefully: the historical pipeline maps these values into array axes explicitly.

The split CSV has `train_split` and `test_split` columns. Do not generate a random replacement when reproducing reported metrics; use the official subject-level split for the relevant benchmark version.

Before training, confirm:

- all phases are spatially registered and have identical array shapes;
- labels match image geometry;
- phase ordering is correct;
- no subject appears in more than one partition;
- access, consent, and redistribution terms permit the intended use.
