# Results for `results/gan_normal10`

Training set (normal, pneumonia): [121, 3494]

## Indirect evaluation – downstream classifier on the real test set (mean ± std over seeds)

| Strategy | # train | AUC | Accuracy | Balanced acc. | Macro-F1 | Recall normal | Recall pneumonia |
|---|---|---|---|---|---|---|---|
| baseline | 3615 | 0.954 ± 0.002 | 0.782 ± 0.025 | 0.709 ± 0.033 | 0.719 ± 0.041 | 0.419 ± 0.066 | 1.000 ± 0.000 |
| classical_aug | 3615 | 0.972 ± 0.005 | 0.821 ± 0.029 | 0.761 ± 0.039 | 0.779 ± 0.043 | 0.523 ± 0.078 | 0.999 ± 0.001 |
| oversample | 6988 | 0.940 ± 0.017 | 0.783 ± 0.039 | 0.711 ± 0.053 | 0.719 ± 0.063 | 0.425 ± 0.109 | 0.997 ± 0.004 |
| gan_balance | 6988 | 0.962 ± 0.003 | 0.767 ± 0.008 | 0.689 ± 0.011 | 0.696 ± 0.014 | 0.379 ± 0.021 | 1.000 ± 0.000 |
| gan_balance+classical_aug | 6988 | 0.984 ± 0.000 | 0.837 ± 0.012 | 0.783 ± 0.016 | 0.803 ± 0.017 | 0.566 ± 0.031 | 1.000 ± 0.000 |
| gan_double | 13976 | 0.959 ± 0.009 | 0.747 ± 0.004 | 0.663 ± 0.005 | 0.662 ± 0.007 | 0.326 ± 0.011 | 1.000 ± 0.000 |
| synthetic_only_TSTR | 3614 | 0.956 ± 0.004 | 0.718 ± 0.008 | 0.676 ± 0.065 | 0.647 ± 0.045 | 0.506 ± 0.347 | 0.846 ± 0.218 |

## Direct evaluation – Fréchet distances (lower is better)

| Feature space | Class | Synthetic vs real test | Real train vs real test (reference) |
|---|---|---|---|
| domain_fid | normal | 24.44 | 66.25 |
| domain_fid | pneumonia | 1.02 | 0.72 |
| inception_fid | normal | 125.20 | 52.32 |
| inception_fid | pneumonia | 52.29 | 28.43 |
