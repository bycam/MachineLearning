# Team 6 — ECG Anomaly Detection

## Workshop 5: Generative Adversarial Networks

Report: [group6_workshop5.pdf](group6_workshop5.pdf)

### 1. Overview

A **vanilla GAN** and a **WGAN-GP** are trained on the 8,647 **abnormal** heartbeats of
the 34 training patients (MIT-BIH, 216 samples per beat, scaled to [-1, 1]). The
generated beats are evaluated with a Fréchet distance computed in the feature space of
the Workshop 2 ResNet1D and with domain-specific metrics, and are then used to
augment the training set of the Workshop 2 classifier. Validation and test patients
are never seen by the generators.

| | Vanilla GAN | WGAN-GP |
|---|---|---|
| Generator (shared) | z (64) → Linear → (128, 27) → 3 × ConvTranspose1d (×2 length) → Conv1d → Tanh, 274,897 params | same |
| Discriminator / critic | 3 × strided Conv1d + BatchNorm + LeakyReLU → Linear → Sigmoid (55,489) | same with LayerNorm, no Sigmoid (68,929) |
| Loss | BCE, non-saturating G loss, one-sided label smoothing (0.9) | Wasserstein + gradient penalty (λ = 10) |
| Optimizer | Adam 2e-4, β (0.5, 0.999), 1 D : 1 G step | Adam 1e-4, β (0, 0.9), 5 critic : 1 G step |
| Training | 150 epochs, best epoch 150 | 300 epochs, best epoch 290 |

### 2. Generation quality (5,000 generated beats per model)

| Metric | Real train (half vs. half) | Real validation patients | Vanilla GAN | WGAN-GP |
|---|---:|---:|---:|---:|
| Fréchet distance vs. training abnormal beats ↓ | 0.016 | 1.213 | **0.410** | 0.875 |
| Fréchet distance vs. validation abnormal beats ↓ | 1.127 | — | **1.771** | 3.331 |
| Classified abnormal by W2 ResNet1D (%) | 82.1 | 84.3 | 77.3 | 76.9 |
| Diversity (mean pair distance) | 9.61 | 9.15 | 9.63 | 9.37 |
| Near copies of training beats (%) | 74.2 | 1.1 | 23.6 | 28.7 |
| Class coverage S / V / F (%) | 29.1 / 61.9 / 9.0 | — | 25.7 / 64.4 / 9.9 | 28.2 / 62.6 / 9.2 |

No mode collapse. The vanilla GAN generates closer beats; WGAN-GP has the smoother,
interpretable loss curve (Wasserstein estimate 7.6 → 0.44) but converges more slowly
and shows high-frequency ConvTranspose artifacts.

### 3. Data augmentation (Workshop 2 ResNet1D retrained, test patients, mean ± std over 3 seeds)

| Condition | PR-AUC | ROC AUC | Precision | Recall | F1 | S detected | False alarms on N |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) Real only | 0.839 ± 0.015 | 0.915 | 0.876 | 0.775 | 0.821 ± 0.033 | 13.5% | 1.03% |
| **(b) Real + 4,000 vanilla GAN beats** | **0.852 ± 0.014** | 0.918 | 0.882 | 0.789 | **0.831 ± 0.040** | 14.8% | 1.00% |
| (c) Real + 4,000 WGAN-GP beats | 0.841 ± 0.015 | 0.916 | 0.775 | 0.794 | 0.782 ± 0.049 | 16.0% | 2.16% |
| (d) Real + 4,000 duplicated real beats (control) | 0.837 ± 0.016 | 0.914 | 0.833 | 0.779 | 0.805 ± 0.015 | 12.5% | 1.40% |

Same architecture, hyperparameters, `pos_weight` (2.75), budget (6 epochs) and
patients in every condition; best epoch by validation PR-AUC, threshold tuned on
validation. Vanilla-GAN beats improve PR-AUC in all three seeds, also against the
oversampling control, but the gain is of the size of the seed-to-seed variability.
WGAN-GP beats double the false alarms.

### 4. Cumulative results (test, 13,869 beats, 6 unseen patients)

| Method | ROC AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| W1 Random Forest | 0.917 | 0.770 | 0.707 | 0.692 | 0.700 |
| W2 ResNet1D | 0.908 | 0.827 | 0.814 | 0.768 | 0.790 |
| W3 DQN adaptive threshold | — | — | 0.948 | 0.798 | 0.867 |
| W4 VAE reconstruction error + KL | 0.801 | 0.214 | 0.284 | 0.440 | 0.345 |
| W5 ResNet1D, real only (3 seeds) | 0.915 | 0.839 | 0.876 | 0.775 | 0.821 |
| **W5 ResNet1D + vanilla GAN beats (3 seeds)** | 0.918 | **0.852** | 0.882 | 0.789 | 0.831 |
| W5 ResNet1D + WGAN-GP beats (3 seeds) | 0.916 | 0.841 | 0.775 | 0.794 | 0.782 |

### 5. Structure

```text
workshop_5/
├── src/
│   ├── gan.py                  # Generator, Discriminator, Critic (optional conditional versions)
│   ├── data_gan.py             # data loading, scaling, frozen W2 classifier
│   ├── gan_metrics.py          # Fréchet distance, diversity, nearest-neighbor distance
│   ├── train_gan.py            # vanilla GAN and WGAN-GP training loop
│   ├── generate_samples.py     # synthetic beats + sample grids and overlays
│   ├── evaluate_gan.py         # generation-quality metrics and training comparison
│   └── augment_and_retrain.py  # data augmentation experiment
├── checkpoints/                # gan_vanilla_best.pt, gan_wgan_best.pt, w2_retrained_*.pt
├── data/                       # synthetic_vanilla.npy, synthetic_wgan.npy (5,000 beats each)
├── runs/workshop5/             # TensorBoard logs, history.csv, sample grids every 10 epochs
└── reports/                    # tables (.csv) and figures (.png)
```

### 6. How to reproduce

Same virtual environment as Workshops 2–4 (`pip install -r requirements.txt`). Requires
`workshop_1/data` and the Workshop 2 checkpoint. From `workshop_5`:

```powershell
python src\train_gan.py --model vanilla          # ~150 epochs
python src\train_gan.py --model wgan             # ~300 epochs, ~9 s per epoch on CPU
python src\generate_samples.py --n 5000
python src\evaluate_gan.py
python src\augment_and_retrain.py --epochs 6                           # conditions (a)-(c), 3 seeds
python src\augment_and_retrain.py --epochs 6 --conditions oversample   # control (d)
tensorboard --logdir runs\workshop5
```

Optional conditional GAN: `python src\train_gan.py --model wgan --conditional`.

This is an academic project and is not intended for clinical use.
