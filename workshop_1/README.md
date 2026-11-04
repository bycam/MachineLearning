# Team 6 — ECG Anomaly Detection

## Workshop 1: Supervised Machine Learning

### 1. Project Overview

This project focuses on supervised anomaly detection in Electrocardiogram (ECG) signals.

The main objective is to develop and evaluate machine learning classifiers capable of distinguishing normal and abnormal heartbeats using ECG recordings.

The project uses the MIT-BIH Arrhythmia Database, provided by PhysioNet.

### 2. Dataset Description

**Dataset:** MIT-BIH Arrhythmia Database

**Source:** [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

**Version:** 1.0.0

Dataset characteristics:

- 48 ECG recordings.
- 47 subjects.
- Two ECG channels per recording.
- Sampling frequency: 360 Hz.
- Original recordings include ECG signals and heartbeat annotations.

The dataset contains `.dat`, `.hea`, and `.atr` files.

### 3. Problem Definition

The project addresses binary heartbeat classification.

The original heartbeat annotations are grouped according to the AAMI classification:

| AAMI Class | Description |
|---|---|
| N | Normal beats |
| S | Supraventricular ectopic beats |
| V | Ventricular ectopic beats |
| F | Fusion beats |
| Q | Unknown or paced beats |

For the binary classification task, the labels are:

- **0 — Normal:** AAMI class N.
- **1 — Abnormal:** AAMI classes S, V, and F.

Class Q and annotations outside the selected heartbeat categories are excluded.

### 4. Project Structure

```text
workshop_1/
│
├── data/
│   ├── raw/
│   │   └── mitdb/
│   ├── processed/
│   │   ├── X.npy
│   │   ├── metadata.csv
│   │   └── feature_names.json
│   └── splits.json
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── dataset.py
│   └── baselines.py
│
├── models/
│   ├── final_random_forest.joblib
│   └── model_config.json
│
├── reports/
│   ├── dataset_inventory.csv
│   ├── checksums_sha256.csv
│   ├── dataset_summary.json
│   ├── global_heartbeat_distribution.csv
│   ├── eda_summary.csv
│   ├── final_test_metrics.csv
│   └── final_confusion_matrix.csv
│
├── .venv/
├── requirements.txt
├── README.md
└── .gitignore
```

### 5. Installation and Environment Setup

The project uses Python and a virtual environment.

Create and activate the environment from the `workshop_1` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Register the Jupyter kernel:

```powershell
python -m ipykernel install --user --name workshop1 --display-name "Python (Workshop 1)"
```

Select the `Python (Workshop 1)` kernel when opening the notebook.

### 6. Dataset Acquisition

The MIT-BIH Arrhythmia Database can be downloaded from PhysioNet.

From the `workshop_1` directory:

```powershell
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
```

Dataset source:

https://physionet.org/content/mitdb/1.0.0/

### 7. Methodology

The project follows these stages:

**7.1. Exploratory Data Analysis**

- Dataset inventory and integrity checks.
- ECG signal visualization.
- Annotation exploration.
- Heartbeat class distribution analysis.

**7.2. Feature Extraction**

Each annotated heartbeat is represented by a fixed-length ECG window:

- 0.2 seconds before the annotation.
- 0.4 seconds after the annotation.
- 360 Hz sampling frequency.
- 216 samples per heartbeat.
- First ECG channel.

The resulting dataset contains 101,418 heartbeat samples and 216 features per sample.

**7.3. Data Splitting**

The dataset is divided into training, validation, and test sets using `StratifiedGroupKFold`.

Patient groups are kept separate to prevent data leakage.

The 201 and 202 recordings are assigned to the same patient group.

The resulting split sizes are:

| Dataset | Samples |
|---|---:|
| Training | 73,943 |
| Validation | 13,606 |
| Test | 13,869 |

**7.4. Baseline Models**

Three supervised classification algorithms were implemented:

- Logistic Regression.
- Decision Tree.
- Random Forest.

**7.5. Hyperparameter Tuning**

Randomized Search with stratified group cross-validation was used to optimize Random Forest.

The selected parameters were:

```python
{
    "n_estimators": 200,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "max_depth": 10,
    "class_weight": "balanced"
}
```

**7.6. Classification Threshold**

A threshold of 0.6 was selected based on validation F1-score analysis.

### 8. Experimental Results

The final model was a tuned Random Forest classifier.

Its performance on the independent test set was:

| Metric | Test Result |
|---|---:|
| Accuracy | 95.12% |
| Precision | 70.74% |
| Recall | 69.18% |
| F1-score | 69.95% |
| PR-AUC | 76.96% |

The model achieved high overall accuracy, although detecting abnormal heartbeats remains challenging.

The results also show that performance on the test set was lower than on the validation set.

### 9. Limitations

- The dataset contains an imbalanced distribution of heartbeat classes.
- The initial feature representation uses only the first ECG channel.
- The current implementation uses fixed-length signal windows.
- The binary classification task combines different abnormal heartbeat categories.
- Model performance may vary across patients.

This project is an academic machine learning experiment and is not intended for clinical diagnosis.

### 10. Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- WFDB
- Scikit-learn
- Jupyter Notebook

### 11. Reproducibility

The project includes:

- Dataset inventory and local SHA-256 checksums.
- Processed heartbeat features.
- Patient-grouped dataset splits.
- Model configuration.
- Final evaluation metrics.
- Exploratory and experimental notebook.

The raw dataset and virtual environment should not be committed to the Git repository.

### 12. Team

**Team 6**

Machine Learning Course — Workshop 1

**Topic:** Supervised Anomaly Detection in ECG Signals.