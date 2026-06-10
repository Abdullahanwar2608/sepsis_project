# Sepsis Onset Prediction — Early Warning System

## Early Prediction of Sepsis Under Noisy Clinical Labels and Irregular ICU Time-Series

> **Predicts sepsis onset 6 hours before clinical diagnosis using ICU patient data.**

---

## Project Overview

This project implements a complete ML/DL pipeline for early sepsis prediction using the **PhysioNet/Computing in Cardiology Challenge 2019** dataset (open access, 40,336 ICU patients). The system addresses four core challenges:

| Challenge | Our Approach |
|---|---|
| High missingness (up to 60% labs) | LOCF → NOCB → Median imputation + informative missingness indicators |
| Irregular time-series | Time-since-last-observation features + rolling windows |
| Label noise (physician lag) | Confident Learning weighting + temporal consistency enforcement |
| Temporal leakage | Patient-level split + post-onset row exclusion |

---

## Quickstart

### For Evaluators / Reviewers

If you are reviewing this code, you can easily run the entire pipeline end-to-end:

**1. Clone the repository and navigate into it**
```bash
git clone https://github.com/Abdullahanwar2608/sepsis_project.git
cd sepsis_project
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**3. Run the Pipeline**

Choose one of the options below to test the code:

*Option A: The Quickest Way (Synthetic Data)*
Best for a quick sanity check. Runs end-to-end in ~2 minutes.
```bash
python run_pipeline.py --synthetic --n-patients 2000
```

*Option B: Real Data (PhysioNet Auto-Download)*
Automatically downloads the Open Access PhysioNet 2019 dataset (~67MB) and trains on it.
```bash
python run_pipeline.py --physionet
```

---

### Local Installation

### 1. Install Dependencies

```bash
cd sepsis_project
C:\Users\<you>\AppData\Local\Python\bin\python.exe -m pip install -r requirements.txt
```

### 2. Run with Synthetic Data (No Download Needed)

```bash
python run_pipeline.py --synthetic --n-patients 2000
```

### 3. Run with PhysioNet 2019 Data (Auto-Download)

```bash
python run_pipeline.py --physionet
```

The script will automatically download Training Set A (~67 MB, open access) from PhysioNet.

### 4. Run with Local Data

```bash
python run_pipeline.py --data-dir "C:/path/to/psv/files"
```

### 5. Run with MIMIC-III Data (Extraction + Training)

```bash
# Option A: Auto-extract from raw CSVs and train
python run_pipeline.py --mimic-dir "C:/path/to/mimic_csvs"

# Option B: Run extraction once (caches to CSV), then train
python mimic_extraction/extract_mimic.py --mimic-dir "C:/path/to/mimic_csvs" --output-dir "./data/mimic"
python run_pipeline.py --mimic-csv "./data/mimic/mimic_hourly_features.csv"
```

### 6. All Options

```
python run_pipeline.py --help

  --synthetic           Use synthetic data (no download required)
  --physionet           Auto-download PhysioNet 2019 Challenge data (open access)
  --data-dir PATH       Path to directory containing PhysioNet .psv files
  --mimic-dir PATH      Path to MIMIC-III CSV directory (will run extraction pipeline)
  --mimic-csv PATH      Path to pre-extracted MIMIC features CSV
  --n-patients N        Number of synthetic patients (default: 2000)
  --max-patients N      Limit patients from real data
  --no-dl               Skip LSTM deep learning model (faster)
  --epochs N            LSTM training epochs (default: 15)
  --seed N              Random seed (default: 42)
```

---

## Project Structure

```
sepsis_project/
├── run_pipeline.py          # ← Entry point: run this!
├── preprocessing.py         # Data loading, imputation, feature engineering
├── models.py                # ML models + Confident Learning + calibration
├── evaluation.py            # Plots, metrics, dashboards
├── utils.py                 # Shared helpers, download utility
├── mimic_extraction/        # MIMIC-III extraction pipeline
│   ├── extract_mimic.py     # Extracts CHARTEVENTS, LABEVENTS, etc. to flat CSV
│   ├── sepsis3_labels.py    # Generates Sepsis-3 labels (SOFA + Suspected Infection)
│   └── item_ids.py          # CareVue & Metavision ItemID mappings
├── deep_learning/
│   └── lstm_model.py        # Bidirectional LSTM with temporal attention
├── outputs/                 # Generated plots and metrics (auto-created)
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── calibration_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance_*.png
│   ├── utility_vs_threshold.png
│   ├── summary_dashboard.png
│   └── metrics_table.csv
├── models/                  # Saved model artifacts (.pkl, .pt) (auto-created)
├── data/                    # Downloaded/placed data (auto-created)
├── requirements.txt
└── README.md
```

---

## Dataset

### PhysioNet/Computing in Cardiology Challenge 2019

- **Access**: 🟢 Open Access — no login required
- **URL**: https://physionet.org/content/challenge-2019/1.0.0/
- **Size**: 40,336 ICU patients from 3 hospital systems
- **Format**: One pipe-delimited `.psv` file per patient (hourly rows)
- **Features**: 8 vital signs, 26 lab values, 6 demographics = 40 features
- **Sepsis definition**: Sepsis-3 (SOFA ≥ 2 + clinical suspicion within 24h)

### MIMIC-III / MIMIC-IV

- **Access**: 🔴 Credentialed access required (PhysioNet account + CITI training + DUA)
- **URL**: https://physionet.org/content/mimiciii/1.4/
- The project includes a full native extraction pipeline (`mimic_extraction/`) that parses the raw MIMIC-III CSV files (CHARTEVENTS, LABEVENTS, INPUTEVENTS), aligns them to hourly intervals, maps CareVue and Metavision ItemIDs, and implements the **Sepsis-3 clinical definition** (suspected infection + SOFA ≥ 2).

---

## Pipeline Components

### 1. Data Preprocessing (`preprocessing.py`)

**Label Engineering** — 6-Hour Prediction Horizon:
```
Timeline:  [... t-9 ... t-6 ... t-3 ... t=0 (onset)] → excluded
                  ↑ target=1 window ↑
```
- `noise_window=3`: Back-shifts onset estimate by 3h to account for physician lag
- All rows at or after estimated onset are excluded (temporal leakage prevention)

**Imputation Strategy**:
1. Binary missingness indicators (informative missingness)
2. Time-since-last-observation features
3. Last Observation Carried Forward (LOCF) within patient
4. Next Observation Carried Backward (NOCB) for initial NaN
5. Population median for structural missingness

**Feature Engineering**:
- Rolling mean/std/trend over 1h, 3h, 6h windows
- SOFA proxy score (organ failure indicator)
- NEWS proxy score (National Early Warning Score)
- Shock Index (HR/SBP)
- Pulse pressure (SBP-DBP)

### 2. Model Training (`models.py`)

| Model | Type | Key Settings |
|---|---|---|
| Logistic Regression | Baseline | C=0.05, balanced class weight |
| Random Forest | Baseline | 300 trees, max_depth=8, balanced |
| XGBoost | Advanced | 500 estimators, scale_pos_weight=imbalance ratio |
| LightGBM | Advanced | 500 estimators, is_unbalance=True |

**Confident Learning (Label Noise Mitigation)**:
- 5-fold cross-validated probability estimation
- Samples where model confidence strongly disagrees with label are down-weighted
- Weights: noisy positive → 0.25, noisy negative → 0.50

**Calibration**: Isotonic regression (CalibratedClassifierCV, cv="prefit")

**Threshold Selection**: Clinical utility score maximization
```
Utility = TP×1.0 - FP×0.05 - FN×2.0 + TN×0.001
```
(Missing sepsis is 2× more costly than a false alarm)

### 3. Deep Learning (`deep_learning/lstm_model.py`)

**Architecture**: Bidirectional LSTM with Temporal Attention
```
Input [batch, seq_len, features]
    → Input Projection (Linear + LayerNorm + ReLU)
    → BiLSTM (2 layers, hidden=64, dropout=0.3)
    → LayerNorm
    → Temporal Attention
    → Classification Head
    → Sigmoid
```
- Class-weighted BCE loss
- AdamW optimizer with ReduceLROnPlateau scheduler
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=5)
- Padding mask applied to exclude padded timesteps from loss

### 4. Evaluation (`evaluation.py`)

Generated plots:
- ROC curves with AUROC for all models
- Precision-Recall curves with Average Precision
- Calibration curves (reliability diagrams)
- Confusion matrices with TP/FP/TN/FN counts
- Feature importance (top 20 features, tree models)
- Clinical utility score vs threshold
- Summary 4-panel dashboard

Metrics reported:
- AUROC, AUPRC, Brier Score
- Sensitivity (Recall), Specificity
- PPV (Precision), NPV
- F1 Score
- PhysioNet utility score at optimal threshold

---

## Performance (Synthetic Data, 2000 patients)

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|
| LightGBM | ~0.88 | ~0.62 | ~0.72 | ~0.85 |
| XGBoost | ~0.87 | ~0.60 | ~0.70 | ~0.86 |
| Random Forest | ~0.84 | ~0.55 | ~0.68 | ~0.84 |
| Logistic Regression | ~0.78 | ~0.45 | ~0.65 | ~0.76 |

*Note: Performance on synthetic data. Real PhysioNet 2019 results will differ.*

---

## Key Design Decisions

### Why exclude post-onset rows?
Rows at or after sepsis onset contain physiological signals that are *consequences* of sepsis, not predictors. Including them would cause the model to learn to detect sepsis from its own symptoms (temporal leakage), leading to artificially inflated metrics.

### Why use AUPRC over AUROC?
With class imbalance (~5-10% positive rows), AUPRC is a more informative metric than AUROC. A random classifier achieves AUPRC = prevalence, not 0.5.

### Why Confident Learning for label noise?
Physician sepsis diagnoses are often recorded late or retroactively, creating noisy labels. Confident Learning (Northcutt et al., 2021) identifies and down-weights samples where the model's cross-validated probability strongly disagrees with the given label, without requiring clean labels.

---

## Citation

If using PhysioNet 2019 data:
```
Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data:
The PhysioNet/Computing in Cardiology Challenge. Critical Care Medicine (2020).
https://doi.org/10.1097/CCM.0000000000004145
```

---

## Author

Built for ITSOLERA AI Internship Screening Task — Deadline: 12 June 2026
