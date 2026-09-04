# Generative Adversarial Networks: what they are and how they are used in medicine

Sources:

- **[A]** M. H. Akpinar *et al.*, "Synthetic Data Generation via Generative Adversarial Networks in Healthcare: A Systematic Review of Image- and Signal-Based Studies," *IEEE Open Journal of Engineering in Medicine and Biology*, vol. 6, pp. 183–192, 2025. doi:10.1109/OJEMB.2024.3508472
- **[M]** A. Makhlouf, M. Maayah, N. Abughanam, C. Catal, "The use of generative adversarial networks in medical image augmentation," *Neural Computing and Applications*, 35:24055–24068, 2023. doi:10.1007/s00521-023-09100-z

---

## 1. The problem GANs address

Deep learning needs a lot of labelled data, and medical data is hard to get. Both reviews start from the same three obstacles ([A] §I, [M] §1):

1. **Scarcity.** Acquiring scans is expensive, and patient privacy limits how much data can be shared.
2. **Labelling cost.** Expert annotation is slow, expensive and inconsistent between experts.
3. **Class imbalance.** Diseases are rarer than healthy cases, so classifiers become biased toward the majority class.

Classical augmentation (flips, rotations, crops) only shows the model the same images again with small changes. A generative model learns the *distribution* of the data and can draw **new** plausible samples from it. Those samples can enlarge a dataset, balance its classes, or be shared in place of real patient data (anonymization).

## 2. The core idea: a two-player game

A GAN (Goodfellow *et al.*, 2014) trains two neural networks against each other ([A] Fig. 1, §II-A):

| | **Generator G(z)** | **Discriminator D(x)** |
|---|---|---|
| Input | random noise vector z ~ p_z (e.g. Gaussian) | an image x |
| Output | a synthetic image G(z) | the probability that x is *real* |
| Goal | fool D | tell real images from synthetic ones |
| Analogy | counterfeiter | detective |

```
   z ~ N(0, I) ──►  Generator G  ──► fake image ─┐
                                                 ├──► Discriminator D ──► P(real)
   real image x ─────────────────────────────────┘          │
        ▲                                                   │
        └──────────── gradients flow back to D and to G ────┘
```

### The objective (Eq. 1 in [A])

$$
\min_G \max_D V(G,D) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
$$

- **D maximises V.** It wants D(x) → 1 on real images and D(G(z)) → 0 on fakes. This is ordinary binary cross-entropy.
- **G minimises V.** It wants D(G(z)) → 1, meaning D is fooled.
- Training **alternates**: one gradient step on D, then one on G ([A] §II-A).

**Why this works (intuition).** For a fixed G, the best possible discriminator is D\*(x) = p_data(x) / (p_data(x) + p_g(x)). Substituting D\* back into V shows that G is minimising the **Jensen–Shannon divergence** between the real distribution p_data and its own distribution p_g. The global optimum is p_g = p_data. At that point D outputs 0.5 everywhere because it cannot tell the two apart.

**Practical trick: the non-saturating loss.** Early in training D easily rejects fakes, so log(1 − D(G(z))) saturates and G receives almost no gradient. G therefore maximises log D(G(z)) instead. The fixed point is the same, but the gradients are much stronger. Our project uses this form.

### Known difficulties

- **Mode collapse.** G produces only a few kinds of images that fool D and ignores the rest of the distribution. [M] Table 2 lists this as a weakness of cGAN and WGAN.
- **Training instability.** The losses oscillate instead of converging, and a strong D can starve G of gradient ([M] Table 2: DCGAN "can suffer from instability").
- **No natural stopping criterion.** The adversarial loss does not measure image quality, so separate metrics are needed ([M] §2.2; [A] §V calls for "robust metrics for evaluating GAN training progress").

## 3. The main GAN variants (as surveyed in both reviews)

| Variant | Key idea | Typical medical use | Prevalence |
|---|---|---|---|
| **cGAN** (Mirza & Osindero 2014) | Feed a condition y (e.g. class label) to both G and D: G(z\|y), D(x\|y) | Generate a *chosen* class → balance datasets | **Most used**: 31% of studies in [A]; #1 in [M] |
| **DCGAN** (Radford *et al.* 2016) | All-convolutional G and D, BatchNorm, LeakyReLU, no pooling → stable image GANs | Basic image synthesis/augmentation | #2 in [M] |
| **CycleGAN** (Zhu *et al.* 2017) | Two G/D pairs plus a *cycle-consistency* loss → translate between domains with **unpaired** data | MRI↔CT, stain normalisation | 18% in [A] (#2) |
| **Pix2Pix** (Isola *et al.* 2017) | cGAN conditioned on an input image; needs **paired** data | Image-to-image translation, segmentation masks → image | frequent in [A] |
| **WGAN** (Arjovsky *et al.* 2017) | Replace JS divergence with Wasserstein distance → smoother gradients | Stabilising training | [M] |
| **ACGAN** | D also predicts the class label | Class-conditional augmentation | [M] |
| **PGGAN** (Karras *et al.* 2018) | Grow resolution progressively 4×4 → 1024×1024 | High-resolution scans, mammograms | [A], [M] |
| **StyleGAN** (Karras *et al.* 2019) | Style-based generator, fine control of attributes | High-fidelity retina/histology (best FID 6.8 in [A]) | [A], [M] |
| **InfoGAN** | Maximise mutual information between latent codes and output → interpretable factors | Unsupervised feature learning | [M] |

## 4. What the two systematic reviews found

### [A] Akpinar *et al.* (2024): 72 Q1 journal articles, images **and** signals
- **Clinical areas:** brain (22%), cardiology (18%), cancer (15%), ophthalmology (12%), lung (10%).
- **Modalities:** MRI and CT dominate imaging; ECG dominates signals.
- **Architectures:** cGAN 31%, CycleGAN 18%.
- **Applications:** data augmentation, anonymization, super-resolution, cross-modality synthesis, multi-task learning.
- **Reported quality:** e.g. best PSNR 63.4 / SSIM 0.98 (shallow U-Net cGAN, oncology); FID 6.8 (StyleGAN2, retina). For diagnosis, GAN augmentation gave gains such as AUC 0.90 for glaucoma.
- **Key limitations it identifies:**
  1. No standardized metrics, so studies cannot be compared directly.
  2. Many studies do not report how many synthetic images they generated.
  3. PSNR/SSIM/FID miss clinically relevant artefacts.
  4. Small datasets (<1000 images) reduce SSIM by 15–20%.
  5. Fine pathological detail is hard to preserve.
- **Future directions:** no-reference and domain-specific metrics, hybrid GAN + diffusion/transformer models, transfer learning, and simulation for training clinicians.

### [M] Makhlouf *et al.* (2023): 52 articles (2018–Feb 2022), image **augmentation** specifically
Five research questions, five answers:
- **RQ1 Architectures:** cGAN first, DCGAN second.
- **RQ2 Modalities:** MRI (17), CT (13), X-ray (9), ultrasound, mammography.
- **RQ3 Organs:** brain (15), chest (8), breast, lung.
- **RQ4 Downstream task:** mostly **classification** (37 of 52), then segmentation.
- **RQ5 Evaluation:** the most frequent method is **indirect quantitative**, i.e. measuring whether the downstream classifier improves.
- **Evaluation taxonomy (Table 3):**
  - *Qualitative, direct:* expert scoring (S-score), inter-rater Kappa, visual Turing tests.
  - *Quantitative, direct:* pixel or distribution comparison such as PSNR, SSIM, MSE, FID, IS.
  - *Quantitative, indirect:* the performance of the downstream model (accuracy, AUC, Dice, …).
- **Key limitations it identifies:** many papers do not specify the architecture or hyperparameters, which hurts reproducibility. Few studies combine architectures.

### The common thread
Both reviews agree on three points. First, **conditional, convolutional GANs used for augmenting imbalanced medical classification datasets** are the most common use case. Second, evaluation is inconsistent across studies. Third, reproducibility is often poor. Our project is built around these three points; see `2_project_report.md`.
