"""
Sepsis Onset Prediction - Data Preprocessing Pipeline
=======================================================
Handles:
- PhysioNet Sepsis Challenge 2019 PSV files (open access)
- MIMIC-III/IV (CSV format after credentialed access + extraction)
- Irregular time-series with up to 60% missingness
- Temporal leakage prevention
- Label noise handling
- Informative missingness features
- Time-since-last-observation features
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import OUTPUTS_DIR, MODELS_DIR, DATA_DIR, ensure_dirs, set_seed

ensure_dirs()
set_seed(42)


# ─────────────────────────────────────────────────
# CONSTANTS — PhysioNet 2019 Feature Schema
# ─────────────────────────────────────────────────

VITAL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"
]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
    "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"
]

DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

FEATURE_COLS = VITAL_COLS + LAB_COLS + DEMO_COLS
LABEL_COL = "SepsisLabel"
PREDICTION_HORIZON = 6   # hours before clinical diagnosis
LABEL_NOISE_WINDOW = 3   # smoothing window for noisy labels


# ─────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────

def load_physionet_psv(data_dir: str, max_patients: Optional[int] = None) -> pd.DataFrame:
    """
    Load PhysioNet 2019 PSV files.

    Args:
        data_dir: Directory containing .psv files (searched recursively)
        max_patients: Limit number of files for quick testing
    """
    files = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.psv"))

    if not files:
        raise FileNotFoundError(f"No .psv files found in {data_dir}")

    if max_patients:
        files = files[:max_patients]

    print(f"  Loading {len(files)} patient PSV files...")

    dfs = []
    for f in files:
        try:
            patient_id = Path(f).stem
            df = pd.read_csv(f, sep="|")
            df["patient_id"] = patient_id
            df["hour"] = np.arange(len(df))
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure all expected columns exist
    for col in FEATURE_COLS + [LABEL_COL]:
        if col not in combined.columns:
            combined[col] = np.nan

    print(f"  Loaded {len(combined):,} rows | {combined['patient_id'].nunique():,} patients")
    return combined


def load_mimic_csv(
    features_path: str,
    max_patients: Optional[int] = None
) -> pd.DataFrame:
    """
    Load MIMIC-III data extracted by mimic_extraction/extract_mimic.py.

    The extractor produces a single flat CSV (mimic_hourly_features.csv)
    with columns matching the PhysioNet 2019 schema:
        patient_id, hour, HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2,
        [26 lab columns], Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS,
        SepsisLabel (Sepsis-3 definition)

    Args:
        features_path: Path to mimic_hourly_features.csv
        max_patients:  Optional limit on number of patients
    """
    print(f"  Loading MIMIC features from: {features_path}")
    df = pd.read_csv(features_path, low_memory=False)

    if max_patients:
        pids = df["patient_id"].unique()[:max_patients]
        df = df[df["patient_id"].isin(pids)].copy()

    # Ensure all expected feature columns exist
    for col in FEATURE_COLS + [LABEL_COL]:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure hour is integer
    df["hour"] = df["hour"].astype(int)

    # Fill SepsisLabel (Sepsis-3 label from extractor, or 0)
    if "SepsisLabel" in df.columns:
        df[LABEL_COL] = df["SepsisLabel"].fillna(0).astype(int)
    else:
        df[LABEL_COL] = 0

    print(f"  Loaded {len(df):,} rows | {df['patient_id'].nunique():,} patients")
    sepsis_prev = df.groupby("patient_id")[LABEL_COL].max().mean()
    print(f"  Sepsis-3 prevalence (patient-level): {sepsis_prev:.1%}")
    return df


def load_mimic_from_raw(
    mimic_dir: str,
    output_dir: str = None,
    max_stays: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract features directly from MIMIC-III CSV directory.
    Calls the mimic_extraction pipeline and returns the result.

    Args:
        mimic_dir:  Path to directory containing MIMIC-III CSV files
        output_dir: Where to cache extracted CSV (default: ./data/mimic)
        max_stays:  Limit for quick testing

    Requires: MIMIC-III access (CITI training + signed DUA + PhysioNet credentials)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from mimic_extraction.extract_mimic import extract_mimic_features

    out = output_dir or str(DATA_DIR / "mimic")
    df = extract_mimic_features(
        mimic_dir=mimic_dir,
        output_dir=out,
        max_stays=max_stays,
        generate_labels=True
    )
    return df


def generate_synthetic_dataset(n_patients: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a clinically-realistic synthetic dataset.

    Key design decisions:
    - ~28% sepsis prevalence (matches PhysioNet 2019 distribution)
    - Vitals and labs trend toward abnormal values near sepsis onset
    - ~60% lab missingness, ~15% vital missingness
    - Correlated physiological changes (e.g., HR ↑ when MAP ↓)
    """
    np.random.seed(seed)
    rows = []

    for pid in range(n_patients):
        stay_len = np.random.randint(24, 120)
        is_sepsis = np.random.random() < 0.28

        if is_sepsis:
            onset_hour = np.random.randint(12, max(13, stay_len - 6))
        else:
            onset_hour = None

        # Patient-level baseline variation
        base_hr = np.random.normal(70, 8)
        base_sbp = np.random.normal(120, 12)
        age = np.random.randint(40, 85)

        for h in range(stay_len):
            # Physiological deterioration trajectory
            if is_sepsis and onset_hour is not None:
                hours_to_onset = onset_hour - h
                if hours_to_onset > 12:
                    progress = 0.0
                elif hours_to_onset >= 0:
                    progress = (12 - hours_to_onset) / 12.0  # 0→1 as onset approaches
                else:
                    progress = 1.0  # post-onset
            else:
                progress = 0.0

            progress = min(max(progress, 0), 1.0)

            row = {
                "patient_id": f"p{pid:06d}",
                "hour": h,
                # Vitals — correlated deterioration
                "HR": np.random.normal(base_hr + 35 * progress, 8),
                "O2Sat": np.random.normal(98 - 6 * progress, 1.5),
                "Temp": np.random.normal(37.0 + 1.8 * progress, 0.4),
                "SBP": np.random.normal(base_sbp - 25 * progress, 12),
                "MAP": np.random.normal(90 - 18 * progress, 8),
                "DBP": np.random.normal(75 - 12 * progress, 6),
                "Resp": np.random.normal(16 + 10 * progress, 2.5),
                "EtCO2": np.nan,
                # Key labs
                "Lactate": np.random.normal(1.2 + 3.5 * progress, 0.4),
                "WBC": np.random.normal(8 + 8 * progress, 2),
                "Creatinine": np.random.normal(1.0 + 2.0 * progress, 0.25),
                "Glucose": np.random.normal(110 + 45 * progress, 18),
                "Platelets": np.random.normal(220 - 90 * progress, 35),
                "Bilirubin_total": np.random.normal(0.8 + 1.2 * progress, 0.25),
                "HCO3": np.random.normal(24 - 6 * progress, 2),
                "pH": np.random.normal(7.40 - 0.10 * progress, 0.03),
                "PaCO2": np.random.normal(40 - 5 * progress, 3),
                "BUN": np.random.normal(15 + 20 * progress, 4),
                "Hct": np.random.normal(38 - 5 * progress, 3),
                "Hgb": np.random.normal(12.5 - 1.5 * progress, 1),
                "PTT": np.random.normal(30 + 10 * progress, 4),
                "Fibrinogen": np.random.normal(300 - 100 * progress, 40),
                # Demographics (constant per patient)
                "Age": age,
                "Gender": np.random.randint(0, 2),
                "Unit1": np.random.randint(0, 2),
                "Unit2": np.random.randint(0, 2),
                "HospAdmTime": np.random.uniform(-24, 0),
                "ICULOS": h,
                # Label: 1 only during/after onset
                "SepsisLabel": int(
                    is_sepsis and onset_hour is not None and h >= onset_hour
                ),
            }

            # Fill remaining lab cols with NaN
            for col in LAB_COLS:
                if col not in row:
                    row[col] = np.nan

            # Simulate missingness: ~60% labs, ~15% vitals
            for col in LAB_COLS:
                if np.random.random() < 0.60:
                    row[col] = np.nan
            for col in VITAL_COLS:
                if col != "EtCO2" and np.random.random() < 0.15:
                    row[col] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    sepsis_prev = df.groupby("patient_id")["SepsisLabel"].max().mean()
    print(f"  Synthetic: {len(df):,} rows | {df['patient_id'].nunique():,} patients | "
          f"Sepsis prevalence: {sepsis_prev:.1%}")
    return df


# ─────────────────────────────────────────────────
# 2. LABEL ENGINEERING — 6-HOUR AHEAD TARGET
# ─────────────────────────────────────────────────

def create_prediction_labels(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
    noise_window: int = LABEL_NOISE_WINDOW
) -> pd.DataFrame:
    """
    Create prediction target: 1 if sepsis onset occurs within next `horizon` hours.

    Design decisions:
    - Target=1 for rows in window [onset - horizon - noise_window, onset - noise_window)
    - Post-onset rows are EXCLUDED to prevent temporal leakage
    - noise_window accounts for physician labelling lag (labels often recorded late)

    Args:
        df: Patient time-series with SepsisLabel column
        horizon: Hours before onset to start predicting (default 6)
        noise_window: Hours to subtract from first positive label to estimate true onset
    """
    df = df.sort_values(["patient_id", "hour"]).copy()
    df["target"] = 0
    df["exclude"] = False

    total_excluded = 0
    total_positive = 0

    for pid, group in df.groupby("patient_id"):
        idx = group.index
        labels = group["SepsisLabel"].values
        hours = group["hour"].values

        pos_hours = hours[labels == 1]

        if len(pos_hours) == 0:
            continue  # Non-sepsis patient

        # Clinical diagnosis time (first positive label)
        diagnosis_hour = pos_hours[0]

        # Estimate true onset: back-shift by noise_window to account for
        # physician documentation lag (labels often entered retroactively)
        true_onset_estimate = max(0, diagnosis_hour - noise_window)

        # Prediction window: target=1 for hours [true_onset - horizon, true_onset)
        pred_start = max(0, true_onset_estimate - horizon)
        pred_end = true_onset_estimate

        for i, h in zip(idx, hours):
            if pred_start <= h < pred_end:
                df.at[i, "target"] = 1
                total_positive += 1
            elif h >= true_onset_estimate:
                df.at[i, "exclude"] = True
                total_excluded += 1

    print(f"  Post-onset rows removed (leakage prevention): {total_excluded:,}")
    df = df[~df["exclude"]].copy()
    print(f"  Positive target rows: {total_positive:,} ({total_positive/len(df):.3f} prevalence)")

    return df.drop(columns=["exclude", "SepsisLabel"], errors="ignore")


# ─────────────────────────────────────────────────
# 3. MISSING VALUE HANDLING
# ─────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-stage imputation strategy:

    1. Add binary missingness indicators for each lab BEFORE imputation
       (informative missingness: labs only drawn when clinically indicated)
    2. Add time-since-last-observation for vitals
       (captures measurement frequency — a key clinical signal)
    3. LOCF (Last Observation Carried Forward) within patient
    4. NOCB (Next Observation Carried Backward) for initial NaN
    5. Population median for remaining structural missingness
    """
    df = df.sort_values(["patient_id", "hour"]).copy()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # ── Step 1: Missingness indicators for labs ──────────────────────────
    for col in LAB_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(np.int8)

    # ── Step 2: Time-since-last-observation for vitals ───────────────────
    for col in VITAL_COLS:
        if col in df.columns:
            def time_since_obs(x):
                last_seen = np.full(len(x), np.nan)
                t = -1
                for i, val in enumerate(x.values):
                    if not np.isnan(val):
                        t = 0
                    elif t >= 0:
                        t += 1
                    last_seen[i] = t
                return pd.Series(last_seen, index=x.index)

            df[f"{col}_time_since"] = (
                df.groupby("patient_id")[col].transform(time_since_obs)
            )

    # ── Step 3 & 4: LOCF then NOCB within each patient ──────────────────
    df[feature_cols] = (
        df.groupby("patient_id")[feature_cols]
        .transform(lambda x: x.ffill().bfill())
    )

    # ── Step 5: Population median for remaining structural missingness ───
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    # Final safety fill: if entire column is NaN, median is also NaN → fill with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Fill time-since columns that are still NaN (no prior obs) with 999
    time_since_cols = [c for c in df.columns if c.endswith("_time_since")]
    df[time_since_cols] = df[time_since_cols].fillna(999)

    remaining = df[feature_cols].isna().sum().sum()
    print(f"  Remaining NaN after imputation: {remaining}")
    return df


# ─────────────────────────────────────────────────
# 4. TEMPORAL FEATURE ENGINEERING
# ─────────────────────────────────────────────────

def engineer_temporal_features(
    df: pd.DataFrame,
    windows: List[int] = [1, 3, 6]
) -> pd.DataFrame:
    """
    Rolling statistics over 1h, 3h, 6h windows — PAST ONLY (no look-ahead).

    Features per vital:
    - rolling mean (trend smoothing)
    - rolling std (variability — high variability = instability)
    - trend: current - value w hours ago (direction of change)

    Composite clinical scores:
    - SOFA proxy (organ failure score)
    - NEWS proxy (National Early Warning Score)
    - Shock index (HR / SBP)
    """
    df = df.sort_values(["patient_id", "hour"]).copy()
    vital_feature_cols = [c for c in VITAL_COLS if c in df.columns]

    new_cols = {}

    for col in vital_feature_cols:
        for w in windows:
            grp = df.groupby("patient_id")[col]
            new_cols[f"{col}_mean_{w}h"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{col}_std_{w}h"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            new_cols[f"{col}_trend_{w}h"] = grp.transform(
                lambda x: x - x.shift(w).fillna(x)
            )

    # ── SOFA proxy (simplified) ──────────────────────────────────────────
    # Captures multi-organ dysfunction (the clinical definition of sepsis)
    sofa_components = {}
    if "Creatinine" in df.columns:
        sofa_components["renal"] = (df["Creatinine"] > 1.2).astype(int)
    if "Platelets" in df.columns:
        sofa_components["coag"] = (df["Platelets"] < 150).astype(int)
    if "Bilirubin_total" in df.columns:
        sofa_components["liver"] = (df["Bilirubin_total"] > 1.2).astype(int)
    if "MAP" in df.columns:
        sofa_components["cardio"] = (df["MAP"] < 70).astype(int)
    if sofa_components:
        new_cols["sofa_proxy"] = sum(sofa_components.values())

    # ── NEWS score proxy ─────────────────────────────────────────────────
    # UK National Early Warning Score for acute deterioration
    news = pd.Series(0, index=df.index)
    if "Resp" in df.columns:
        news += (df["Resp"] >= 25).astype(int) * 3
        news += ((df["Resp"] >= 21) & (df["Resp"] < 25)).astype(int) * 2
        news += ((df["Resp"] >= 9) & (df["Resp"] <= 11)).astype(int) * 1
        news += (df["Resp"] <= 8).astype(int) * 3
    if "O2Sat" in df.columns:
        news += (df["O2Sat"] <= 91).astype(int) * 3
        news += ((df["O2Sat"] >= 92) & (df["O2Sat"] <= 93)).astype(int) * 2
        news += ((df["O2Sat"] >= 94) & (df["O2Sat"] <= 95)).astype(int) * 1
    if "SBP" in df.columns:
        news += (df["SBP"] <= 90).astype(int) * 3
        news += ((df["SBP"] >= 91) & (df["SBP"] <= 100)).astype(int) * 2
        news += ((df["SBP"] >= 101) & (df["SBP"] <= 110)).astype(int) * 1
        news += (df["SBP"] >= 220).astype(int) * 3
    if "HR" in df.columns:
        news += (df["HR"] <= 40).astype(int) * 3
        news += ((df["HR"] >= 41) & (df["HR"] <= 50)).astype(int) * 1
        news += ((df["HR"] >= 91) & (df["HR"] <= 110)).astype(int) * 1
        news += ((df["HR"] >= 111) & (df["HR"] <= 130)).astype(int) * 2
        news += (df["HR"] > 130).astype(int) * 3
    if "Temp" in df.columns:
        news += (df["Temp"] <= 35.0).astype(int) * 3
        news += ((df["Temp"] >= 35.1) & (df["Temp"] <= 36.0)).astype(int) * 1
        news += ((df["Temp"] >= 38.1) & (df["Temp"] <= 39.0)).astype(int) * 1
        news += (df["Temp"] > 39.0).astype(int) * 2
    new_cols["news_proxy"] = news

    # ── Shock Index (HR / SBP) ───────────────────────────────────────────
    if "HR" in df.columns and "SBP" in df.columns:
        new_cols["shock_index"] = df["HR"] / (df["SBP"] + 1e-6)

    # ── Pulse pressure (SBP - DBP) ───────────────────────────────────────
    if "SBP" in df.columns and "DBP" in df.columns:
        new_cols["pulse_pressure"] = df["SBP"] - df["DBP"]

    engineered = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, engineered], axis=1)

    print(f"  Added {len(new_cols)} temporal features | Total columns: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────────
# 5. LABEL NOISE MITIGATION
# ─────────────────────────────────────────────────

def detect_label_noise(
    df: pd.DataFrame,
    method: str = "temporal_consistency"
) -> pd.DataFrame:
    """
    Label noise mitigation strategies.

    Methods:
    - 'temporal_consistency': Enforce monotonicity — once positive, stays positive.
      Handles the case where physicians correct diagnoses retroactively.
    - 'smoothing': Soft labels [0.1, 0.9] instead of hard [0, 1].
      Used as an option during training.

    Note: Confident Learning (cleanlab-style) is applied during model training
    after an initial probability estimate is available.
    """
    df = df.copy()

    if method == "temporal_consistency":
        # Enforce monotonicity: once target=1, it stays 1.
        # Use a vectorized cummax approach (no groupby.apply needed).
        df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)

        # cummax within patient groups: once we see a 1, all subsequent rows become 1
        df["target"] = (
            df.groupby("patient_id")["target"]
            .transform("cummax")
        )

        n_patients = df["patient_id"].nunique()
        print(f"  Temporal consistency enforced on {n_patients:,} patients")

    elif method == "smoothing":
        df["target_original"] = df["target"].copy()
        df["target_smooth"] = df["target"].apply(lambda x: 0.9 if x == 1 else 0.1)
        print("  Label smoothing applied: hard labels → [0.1, 0.9]")

    return df


# ─────────────────────────────────────────────────
# 6. TRAIN/VAL/TEST SPLIT — Patient-Level Stratified
# ─────────────────────────────────────────────────

def patient_stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at PATIENT level (not row level) to prevent data leakage.
    Stratified by sepsis status to maintain class balance across splits.

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    patient_labels = (
        df.groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "has_sepsis"})
    )

    sepsis_pids = patient_labels[patient_labels["has_sepsis"] == 1]["patient_id"].values
    non_sepsis_pids = patient_labels[patient_labels["has_sepsis"] == 0]["patient_id"].values

    np.random.shuffle(sepsis_pids)
    np.random.shuffle(non_sepsis_pids)

    def split_pids(pids):
        n = len(pids)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        # train gets the remainder
        train = pids[n_val + n_test:]
        val = pids[:n_val]
        test = pids[n_val:n_val + n_test]
        return train, val, test

    train_sep, val_sep, test_sep = split_pids(sepsis_pids)
    train_non, val_non, test_non = split_pids(non_sepsis_pids)

    train_pids = np.concatenate([train_sep, train_non])
    val_pids = np.concatenate([val_sep, val_non])
    test_pids = np.concatenate([test_sep, test_non])

    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()

    print(f"  Train: {len(train_pids):,} patients ({len(train_df):,} rows) | "
          f"target={train_df['target'].mean():.3f}")
    print(f"  Val:   {len(val_pids):,} patients ({len(val_df):,} rows) | "
          f"target={val_df['target'].mean():.3f}")
    print(f"  Test:  {len(test_pids):,} patients ({len(test_df):,} rows) | "
          f"target={test_df['target'].mean():.3f}")

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature columns (excludes metadata and targets)."""
    exclude = {
        "patient_id", "hour", "target",
        "target_original", "target_smooth", "SepsisLabel"
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────

def run_preprocessing_pipeline(
    data_dir: Optional[str] = None,
    use_synthetic: bool = True,
    n_synthetic: int = 2000,
    max_patients: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Full preprocessing pipeline.

    Returns:
        train_df, val_df, test_df, feature_cols
    """
    print("=" * 60)
    print("SEPSIS PREDICTION — PREPROCESSING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────
    if use_synthetic or data_dir is None:
        print(f"\n[1/5] Generating synthetic dataset ({n_synthetic} patients)...")
        df = generate_synthetic_dataset(n_synthetic)
    else:
        print(f"\n[1/5] Loading PhysioNet PSV data from: {data_dir}")
        df = load_physionet_psv(data_dir, max_patients=max_patients)

    # ── Step 2: Label engineering ────────────────────────────────────────
    print(f"\n[2/5] Engineering labels (horizon={PREDICTION_HORIZON}h, "
          f"noise_window={LABEL_NOISE_WINDOW}h)...")
    df = create_prediction_labels(df)

    # ── Step 3: Imputation ───────────────────────────────────────────────
    print("\n[3/5] Handling missing values (LOCF + NOCB + median)...")
    df = handle_missing_values(df)

    # ── Step 4: Feature engineering ──────────────────────────────────────
    print("\n[4/5] Engineering temporal features...")
    df = engineer_temporal_features(df, windows=[1, 3, 6])

    # ── Step 5: Label noise mitigation ───────────────────────────────────
    print("\n[5/5] Applying label noise mitigation (temporal consistency)...")
    df = detect_label_noise(df, method="temporal_consistency")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[Split] Patient-level stratified split (70/15/15)...")
    train_df, val_df, test_df = patient_stratified_split(df)

    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal features: {len(feature_cols)}")
    print("=" * 60)

    return train_df, val_df, test_df, feature_cols


if __name__ == "__main__":
    train_df, val_df, test_df, feature_cols = run_preprocessing_pipeline(
        use_synthetic=True, n_synthetic=500
    )
    print("\nSample features:", feature_cols[:10])
    print("Train target distribution:\n", train_df["target"].value_counts())
