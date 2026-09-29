# PneumoGAN: conditional DCGAN augmentation for imbalanced chest X-rays

A course project on Generative Adversarial Networks, built on two systematic reviews:

- **Akpinar et al. (2024)**, *Synthetic Data Generation via GANs in Healthcare*, IEEE OJEMB. [doi:10.1109/OJEMB.2024.3508472](https://doi.org/10.1109/OJEMB.2024.3508472)
- **Makhlouf et al. (2023)**, *The use of GANs in medical image augmentation*, Neural Comput. & Applic. [doi:10.1007/s00521-023-09100-z](https://doi.org/10.1007/s00521-023-09100-z)

## Why this project, and how it ties to the papers

| Finding in the reviews | How the project uses it |
|---|---|
| cGAN (#1) and DCGAN (#2) are the most used architectures | The model is a **conditional DCGAN** |
| X-ray, chest and lung are among the most studied modality and organs | Data: **PneumoniaMNIST** (5,856 paediatric chest X-rays, 64×64) |
| The main use case is augmenting **imbalanced** classification datasets | The minority class (NORMAL, ~1:2.9) is balanced with synthetic images |
| The most common evaluation is *indirect* (downstream classifier) | Classifier AUC / balanced accuracy, 3 seeds, 7 strategies |
| No standard metrics; FID/SSIM lack context; domain metrics are needed | FID **and** a domain-specific FID, each next to a real-vs-real reference |
| Small datasets hurt GAN quality | A **scarcity experiment** with only 10% of NORMAL images kept |
| Poor reporting of hyperparameters and number of synthetic images | All hyperparameters, image counts and seeds are logged; one script reproduces everything |

## Layout

```
docs/1_GANs_explained.md   theory of GANs + summary of both reviews
presentation/           French slides (PPTX + PDF), built from results/*.json
src/data.py             PneumoniaMNIST loading and scarcity subsampling
src/models.py           conditional DCGAN (G, projection D) + CNN classifier
src/train_gan.py        adversarial training
src/evaluate.py         FID, downstream classifier, TSTR, figures
run_all.sh              reproduces all experiments
results/                sample grids, training logs, metrics (checkpoints not committed)
```

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision medmnist scipy scikit-learn matplotlib
./run_all.sh            # about 1 hour on a laptop RTX 4050
```

The dataset (about 20 MB) downloads automatically into `data/` the first time you run it.

## Main results

Balanced accuracy of the pneumonia classifier on 624 real test X-rays (mean of 3 seeds):

| Training data | Full data (1,214 normal) | Scarce (121 normal) |
|---|---|---|
| Real only | 0.868 | 0.709 |
| + classical augmentation | 0.910 | 0.761 |
| + oversampling | 0.858 | 0.711 |
| + GAN (class balancing) | 0.843 | 0.689 |
| **+ GAN + classical augmentation** | **0.920** | **0.783** |

Full tables: `results/eval_full/results.md` and `results/eval_normal10/results.md`. GAN images help most when combined with classical augmentation. On their own they do not beat simple oversampling. The differences are within about 1–2 standard deviations over 3 seeds.
