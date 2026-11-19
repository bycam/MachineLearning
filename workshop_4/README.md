# Team 6 — ECG Anomaly Detection

## Workshop 4: Autoencoders and Variational Autoencoders

Report: [group6_workshop4.pdf](group6_workshop4.pdf)

### 1. Overview

1D convolutional **autoencoders (AE)** and **variational autoencoders (VAE)** are
trained on the **normal** heartbeats of the training patients only (65,296 beats,
no abnormal labels). A beat that the model cannot reconstruct well is flagged as an
anomaly. Data, normalization and patient splits are the same as in Workshops 1–3.

| | AE | VAE |
|---|---|---|
| Encoder | 3 × Conv1d (stride 2): 216 → 108 → 54 → 27 samples, 16/32/64 channels | same |
| Bottleneck | Linear → z | Linear → μ, log σ²; z = μ + σ·ε (reparameterization) |
| Decoder | Linear → 3 × ConvTranspose1d → Conv1d, linear output | same |
| Loss | MSE | Σ squared error + β · KL, with KL annealing (β ramps 0 → target in 10 epochs) |

### 2. Runs (validation)

| Run | z | β | Val. MSE (normal) | Val. KL | Val. AUC-ROC |
|---|---:|---:|---:|---:|---:|
| `ae_z8` | 8 | — | 0.0724 | — | 0.694 |
| **`ae_z16` → `ae_best.pt`** | 16 | — | 0.0252 | — | **0.708** |
| `ae_z32` | 32 | — | 0.0164 | — | 0.649 |
| **`vae_z16_b0.5` → `vae_best.pt`** | 16 | 0.5 | 0.0523 | 13.8 | **0.754** |
| `vae_z16_b1` | 16 | 1 | 0.0700 | 10.0 | 0.746 |
| `vae_z16_b4` | 16 | 4 | 0.1178 | 4.3 | 0.653 |

Better reconstruction does not mean better anomaly detection: the AE with z = 32
reconstructs best but detects worst.

### 3. Test results (13,869 beats, 6 unseen patients)

| Method | Supervision | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| W1 Random Forest | labels | 0.917 | 0.770 | 0.707 | 0.692 | 0.700 |
| W2 ResNet1D | labels | 0.908 | 0.827 | 0.814 | 0.768 | 0.790 |
| W3 DQN adaptive threshold | labels + reward | — | — | 0.948 | 0.798 | 0.867 |
| W4 AE reconstruction error | none | 0.637 | 0.115 | 0.149 | 0.491 | 0.228 |
| W4 VAE reconstruction error | none | 0.757 | 0.165 | 0.205 | 0.423 | 0.276 |
| **W4 VAE reconstruction error + KL** | none | **0.801** | 0.214 | 0.284 | 0.440 | 0.345 |

Thresholds for precision/recall/F1 are tuned on validation.

### 4. Latent space (silhouette on test latent vectors)

| Representation | normal/abnormal | AAMI class | patient record |
|---|---:|---:|---:|
| AE z (16) | 0.210 | 0.117 | 0.083 |
| VAE μ (16) | 0.192 | 0.163 | 0.138 |
| W2 ResNet1D features (64, supervised) | 0.407 | 0.125 | −0.032 |

The autoencoders organize beats mainly by **patient**; the supervised network
ignores the patient and separates normal from abnormal twice as well.

### 5. Structure

```text
workshop_4/
├── src/
│   ├── autoencoder.py     # ConvAE, ConvVAE, KL, losses
│   ├── data_ae.py         # data loading (Workshop 1 splits, per-beat z-score)
│   ├── train_ae.py        # training script (AE / VAE, KL annealing, TensorBoard)
│   ├── evaluate_ae.py     # model selection, reconstruction, anomaly detection, cumulative table
│   └── analyze_latent.py  # t-SNE, UMAP, silhouette (AE, VAE, frozen W2 encoder)
├── checkpoints/           # ae_best.pt, vae_best.pt and the best epoch of every run
├── runs/workshop4/        # TensorBoard logs, history.csv, curves.png per run
└── reports/               # tables and figures
```

### 6. How to reproduce

Uses the Workshop 3 virtual environment plus `umap-learn`. From `workshop_4`:

```powershell
..\workshop_3\.venv\Scripts\Activate.ps1
python -m pip install umap-learn

python src\train_ae.py --model ae --z-dim 16
python src\train_ae.py --model ae --z-dim 8
python src\train_ae.py --model ae --z-dim 32
python src\train_ae.py --model vae --z-dim 16 --beta 0.5
python src\train_ae.py --model vae --z-dim 16 --beta 1
python src\train_ae.py --model vae --z-dim 16 --beta 4

python src\evaluate_ae.py       # selects ae_best.pt / vae_best.pt on validation, evaluates on test
python src\analyze_latent.py
tensorboard --logdir runs\workshop4
```

Each run takes about 15 minutes on CPU (30 epochs). Seed 42.

This is an academic project and is not intended for clinical use.
