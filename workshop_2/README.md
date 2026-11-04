# Team 6 — ECG Anomaly Detection

## Workshop 2: Deep Learning

### 1. Overview

This workshop trains a deep learning model for the same task as Workshop 1:
binary classification of ECG heartbeats from the MIT-BIH Arrhythmia Database
into **Normal** (AAMI class N) and **Abnormal** (AAMI classes S, V, F).

The model is a **1D residual convolutional network (ResNet1D)**. It reuses the
exact heartbeat windows and the patient-grouped train/validation/test splits of
Workshop 1, so the comparison with the Random Forest baseline is fair.

### 2. Results on the test set

13,869 heartbeats from 6 patients never seen during training (1,139 abnormal, 8.2%).

| Model | Accuracy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Random Forest (Workshop 1) | 95.12% | 70.74% | 69.18% | 69.95% | 76.96% | 91.69% |
| **ResNet1D (Workshop 2)** | **96.65%** | **81.40%** | **76.82%** | **79.04%** | **82.66%** | 90.78% |

The ResNet1D improves precision and recall at the same time: 200 false alarms
instead of 326, and 264 missed abnormal beats instead of 351.

Detection rate per AAMI class on test:

| Class | Beats | Random Forest | ResNet1D |
|---|---:|---:|---:|
| V (ventricular) | 910 | 86.5% | 93.0% |
| S (supraventricular) | 219 | 0.0% | 11.9% |
| F (fusion) | 10 | 10.0% | 30.0% |
| N (false-alarm rate) | 12,730 | 2.56% | 1.57% |

Supraventricular (S) beats remain the main weakness of both models: their shape
is almost normal and they are mainly recognized by arriving early (a short RR
interval), which a single 0.6 s window does not contain.

### 3. Model

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | (1, 216) | |
| Stem: Conv1d k=7 + BN + ReLU | (16, 216) | |
| Residual block 1 | (16, 216) | |
| Residual block 2 (stride 2) | (32, 108) | |
| Residual block 3 (stride 2) | (64, 54) | |
| Residual block 4 (stride 2) | (64, 27) | |
| Global average pooling | (64, 1) | |
| Dropout 0.5 + Linear | 1 logit | |
| **Total** | | **89,809** |

Each residual block is conv (k=5) → BN → ReLU → dropout 0.1 → conv (k=5) → BN,
plus a skip connection (1×1 conv when the shape changes).

### 4. Training protocol

- **Input:** 216-sample windows (0.2 s before and 0.4 s after the R-peak, 360 Hz,
  first channel), z-scored per beat.
- **Augmentation (train only):** amplitude scaling ±10%, time shift ±5 samples,
  Gaussian noise, baseline wander.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight = sqrt(N_normal / N_abnormal) = 2.75`.
- **Optimizer:** AdamW, learning rate 3e-4, weight decay 1e-2.
- **Scheduler:** cosine annealing over 30 epochs.
- **Model selection:** best validation PR-AUC; early stopping with patience 8.
- **Decision threshold:** 0.784, chosen to maximize F1 on the validation set.
- **Seed:** 42 (Python, NumPy and PyTorch).

Two runs were performed (logs in `runs/`):

| Run | Channels | Dropout | LR | Weight decay | pos_weight | Best epoch | Val PR-AUC | Val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet1d` | 32 | 0.3 | 1e-3 | 1e-4 | 7.55 | 1 | 0.905 | 0.861 |
| `resnet1d_v2` (final) | 16 | 0.5 | 3e-4 | 1e-2 | 2.75 | 2 | **0.917** | **0.891** |

Run 1 overfitted to the training patients from epoch 2 on (training loss kept
falling while validation loss rose to 0.72). Run 2 adds regularization and
reduces capacity, which kept the validation loss between 0.16 and 0.24.

### 5. Project structure

```text
workshop_2/
├── src/
│   ├── prepare_data.py    # downloads MIT-BIH and rebuilds the Workshop 1 dataset
│   ├── data.py            # normalization, augmentation, DataLoaders
│   ├── model.py           # ResNet1D architecture
│   ├── train.py           # training loop, TensorBoard logging, checkpointing
│   ├── evaluate.py        # test metrics, comparison with the baseline, error analysis
│   └── test_model.py      # inference on new ECG records
├── checkpoints/
│   ├── model_best.pt      # final model (run 2) + tuned threshold
│   └── model_run1.pt      # run 1, kept for comparison
├── runs/                  # TensorBoard logs, history.csv, learning curves
├── reports/               # metrics, confusion matrices, per-class and per-record tables
├── notebooks/
├── requirements.txt
└── README.md
```

### 6. How to reproduce

From the `workshop_2` directory (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 1. Download MIT-BIH and rebuild the processed dataset (~90 MB)
python src\prepare_data.py

# 2. Train the final model (~25 min on CPU)
python src\train.py --run-name resnet1d_v2 --lr 3e-4 --weight-decay 1e-2 --dropout 0.5 --base-channels 16 --pos-weight-mode sqrt --epochs 30 --patience 8

# 3. Evaluate on validation and test
python src\evaluate.py --split val
python src\evaluate.py

# 4. Predict on a new ECG record
python src\test_model.py --record ..\workshop_1\data\raw\mitdb\106
```

To view the training curves:

```powershell
tensorboard --logdir runs
```

### 7. Limitations

- Only 6–7 patients per validation/test split, so results vary a lot between patients.
- Errors concentrate in a few records: record 108 (very noisy) produces 173 of
  the 200 false alarms; records 201/202 and 124 account for most missed beats.
- Supraventricular beats are rarely detected because the input has no RR-interval
  information.
- Only the first ECG channel is used.

This is an academic project and is not intended for clinical diagnosis.