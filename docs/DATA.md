# Data preparation

No medical images are distributed in this repository.

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
