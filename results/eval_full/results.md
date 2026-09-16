# Results for `results/gan_full`

Training set (normal, pneumonia): [1214, 3494]

## Indirect evaluation – downstream classifier on the real test set (mean ± std over seeds)

| Strategy | # train | AUC | Accuracy | Balanced acc. | Macro-F1 | Recall normal | Recall pneumonia |
|---|---|---|---|---|---|---|---|
| baseline | 4708 | 0.972 ± 0.002 | 0.899 ± 0.026 | 0.868 ± 0.037 | 0.885 ± 0.033 | 0.744 ± 0.080 | 0.992 ± 0.008 |
| classical_aug | 4708 | 0.986 ± 0.002 | 0.931 ± 0.020 | 0.910 ± 0.028 | 0.923 ± 0.024 | 0.825 ± 0.058 | 0.995 ± 0.004 |
| oversample | 6988 | 0.978 ± 0.003 | 0.893 ± 0.014 | 0.858 ± 0.019 | 0.878 ± 0.018 | 0.719 ± 0.037 | 0.997 ± 0.000 |
| gan_balance | 6988 | 0.979 ± 0.002 | 0.882 ± 0.012 | 0.843 ± 0.016 | 0.863 ± 0.015 | 0.687 ± 0.032 | 0.999 ± 0.001 |
| gan_balance+classical_aug | 6988 | 0.988 ± 0.001 | 0.937 ± 0.008 | 0.920 ± 0.014 | 0.931 ± 0.009 | 0.852 ± 0.036 | 0.988 ± 0.010 |
| gan_double | 13976 | 0.980 ± 0.001 | 0.887 ± 0.004 | 0.850 ± 0.006 | 0.870 ± 0.005 | 0.699 ± 0.011 | 1.000 ± 0.000 |
| synthetic_only_TSTR | 4708 | 0.957 ± 0.005 | 0.862 ± 0.012 | 0.825 ± 0.014 | 0.842 ± 0.014 | 0.674 ± 0.026 | 0.975 ± 0.003 |

## Direct evaluation – Fréchet distances (lower is better)

| Feature space | Class | Synthetic vs real test | Real train vs real test (reference) |
|---|---|---|---|
| domain_fid | normal | 35.35 | 60.26 |
| domain_fid | pneumonia | 0.73 | 0.59 |
| inception_fid | normal | 83.30 | 37.28 |
| inception_fid | pneumonia | 57.78 | 28.43 |
