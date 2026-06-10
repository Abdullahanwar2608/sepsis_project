"""
MIMIC-III Feature Extraction Pipeline
======================================
Extracts hourly time-series features from MIMIC-III CSV tables and
produces a single flat CSV compatible with the sepsis prediction pipeline.

This script reads directly from the MIMIC-III CSV files (no database needed).

Input:  MIMIC-III CSV directory (26 tables)
Output: mimic_hourly_features.csv  — one row per patient-ICU-hour

Key tables used:
    PATIENTS       → demographics (age, gender, DOB)
    ADMISSIONS     → hospital admission/discharge times, mortality
    ICUSTAYS       → ICU stay start/end, unit type
    CHARTEVENTS    → vital signs (~1/hour, CareVue + Metavision ITEMIDs)
    LABEVENTS      → lab results (linked via D_LABITEMS)
    OUTPUTEVENTS   → urine output (for SOFA renal)
    INPUTEVENTS_MV → medications including vasopressors/antibiotics (Metavision)
    INPUTEVENTS_CV → medications including vasopressors/antibiotics (CareVue)

Usage:
    python extract_mimic.py --mimic-dir /path/to/mimic_csvs --output-dir ./data

    # Quick test with only 500 ICU stays
    python extract_mimic.py --mimic-dir /path/to/mimic --output-dir ./data --max-stays 500
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from item_ids import (
    VITAL_ITEMIDS, LAB_ITEMIDS, FIO2_CHART_ITEMIDS,
    VASOPRESSOR_ITEMIDS, VALUE_BOUNDS, FEATURE_TO_COLUMN
)


# ─────────────────────────────────────────────────
# HELPER — flatten ItemID dict to list
# ─────────────────────────────────────────────────

def all_vital_itemids() -> List[int]:
    ids = []
    for v in VITAL_ITEMIDS.values():
        ids.extend(v)
    ids.extend(FIO2_CHART_ITEMIDS)
    return list(set(ids))


def all_lab_itemids() -> List[int]:
    ids = []
    for v in LAB_ITEMIDS.values():
        ids.extend(v)
    return list(set(ids))


def build_itemid_to_feature() -> Dict[int, str]:
    """Build reverse mapping: itemid → feature name."""
    mapping = {}
    for feat, ids in VITAL_ITEMIDS.items():
        for iid in ids:
            mapping[iid] = feat
    for iid in FIO2_CHART_ITEMIDS:
        mapping[iid] = "FiO2"
    for feat, ids in LAB_ITEMIDS.items():
        for iid in ids:
            mapping[iid] = feat
    return mapping


# ─────────────────────────────────────────────────
# 1. LOAD ICU STAYS METADATA
# ─────────────────────────────────────────────────

def load_icu_metadata(mimic_dir: str, max_stays: Optional[int] = None) -> pd.DataFrame:
    """
    Load and join ICUSTAYS + PATIENTS + ADMISSIONS.

    Returns DataFrame with one row per ICU stay:
        patient_id, ICUSTAY_ID, HADM_ID, SUBJECT_ID,
        INTIME, OUTTIME, icu_los_hours,
        Age, Gender, hospital_expire_flag, DOD
    """
    mimic_path = Path(mimic_dir)
    print("  Loading ICUSTAYS...")
    icu = pd.read_csv(
        mimic_path / "ICUSTAYS.csv",
        parse_dates=["INTIME", "OUTTIME"]
    )
    print(f"    {len(icu):,} ICU stays")

    # Limit for testing
    if max_stays:
        icu = icu.head(max_stays)

    print("  Loading PATIENTS...")
    patients = pd.read_csv(
        mimic_path / "PATIENTS.csv",
        parse_dates=["DOB", "DOD"]
    )

    print("  Loading ADMISSIONS...")
    admissions = pd.read_csv(
        mimic_path / "ADMISSIONS.csv",
        parse_dates=["ADMITTIME", "DISCHTIME"],
        usecols=["HADM_ID", "SUBJECT_ID", "ADMITTIME", "DISCHTIME",
                 "HOSPITAL_EXPIRE_FLAG", "ADMISSION_TYPE",
                 "DIAGNOSIS", "EDREGTIME", "EDOUTTIME"]
    )

    # Join
    df = icu.merge(patients[["SUBJECT_ID", "DOB", "DOD", "GENDER"]],
                   on="SUBJECT_ID", how="left")
    df = df.merge(admissions, on=["HADM_ID", "SUBJECT_ID"], how="left")

    # Compute ICU LOS (hours)
    df["icu_los_hours"] = (df["OUTTIME"] - df["INTIME"]).dt.total_seconds() / 3600

    # Compute age at ICU admission
    # Note: MIMIC shifts patients >89 by 300 years (DOB will be ~1800s)
    df["Age"] = (df["INTIME"] - df["DOB"]).dt.days / 365.25
    # Cap at 89 for de-identified >89 patients
    df["Age"] = df["Age"].clip(upper=89)

    # Gender encoding
    df["Gender"] = (df["GENDER"] == "M").astype(int)

    # ICU unit type encoding
    unit_map = {
        "MICU": 0, "CCU": 1, "CSRU": 2,
        "SICU": 3, "TSICU": 4, "NICU": 5
    }
    df["Unit1"] = df["FIRST_CAREUNIT"].map(unit_map).fillna(0).astype(int)
    df["Unit2"] = df["LAST_CAREUNIT"].map(unit_map).fillna(0).astype(int)

    # Hospital admission time offset (hours before ICU admission)
    df["HospAdmTime"] = (
        (df["INTIME"] - df["ADMITTIME"]).dt.total_seconds() / 3600
    ).clip(-72, 0)

    # Patient ID in our pipeline format
    df["patient_id"] = df["ICUSTAY_ID"].apply(lambda x: f"mimic_{int(x)}")

    print(f"  ICU stays after join: {len(df):,}")
    print(f"  Unique patients: {df['SUBJECT_ID'].nunique():,}")

    return df


# ─────────────────────────────────────────────────
# 2. EXTRACT CHARTEVENTS (VITAL SIGNS)
# ─────────────────────────────────────────────────

def extract_chartevents(
    mimic_dir: str,
    icustay_ids: List[int],
    icustay_metadata: pd.DataFrame,
    chunksize: int = 1_000_000
) -> pd.DataFrame:
    """
    Extract vital signs from CHARTEVENTS, binned to ICU hours.

    CHARTEVENTS is the largest table (~330M rows, ~33 GB uncompressed).
    We process in chunks and filter to relevant ICU stays + item IDs.
    """
    mimic_path = Path(mimic_dir)
    chart_path = mimic_path / "CHARTEVENTS.csv"

    if not chart_path.exists():
        print("  CHARTEVENTS.csv not found — skipping vital signs")
        return pd.DataFrame()

    target_ids = set(all_vital_itemids())
    target_stays = set(icustay_ids)
    itemid_to_feat = build_itemid_to_feature()

    # Build intime lookup for hour calculation
    intime_lookup = icustay_metadata.set_index("ICUSTAY_ID")["INTIME"]

    print(f"  Extracting CHARTEVENTS (this may take several minutes)...")
    print(f"  Filtering to {len(target_ids)} ItemIDs × {len(target_stays):,} ICU stays")

    all_chunks = []
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(
        chart_path,
        usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM", "ERROR"],
        parse_dates=["CHARTTIME"],
        chunksize=chunksize,
        low_memory=False
    ):
        total_rows += len(chunk)

        # Filter
        chunk = chunk[
            (chunk["ICUSTAY_ID"].isin(target_stays)) &
            (chunk["ITEMID"].isin(target_ids)) &
            (chunk["ERROR"].fillna(0) == 0) &
            (chunk["VALUENUM"].notna())
        ].copy()

        if chunk.empty:
            continue

        # Map ITEMID → feature name
        chunk["feature"] = chunk["ITEMID"].map(itemid_to_feat)

        # Convert Fahrenheit → Celsius
        temp_f_mask = chunk["ITEMID"].isin(VITAL_ITEMIDS.get("Temp_F", []))
        chunk.loc[temp_f_mask, "VALUENUM"] = (
            chunk.loc[temp_f_mask, "VALUENUM"] - 32
        ) * 5 / 9
        chunk.loc[temp_f_mask, "feature"] = "Temp_C"

        # Compute hour offset from ICU admission
        chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(int)
        chunk["intime"] = chunk["ICUSTAY_ID"].map(intime_lookup)
        chunk = chunk.dropna(subset=["intime"])
        chunk["hour"] = (
            (chunk["CHARTTIME"] - chunk["intime"]).dt.total_seconds() / 3600
        ).round().astype(int)

        # Keep only within 0–168h (first week of ICU)
        chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 168)]

        kept_rows += len(chunk)
        all_chunks.append(chunk[["ICUSTAY_ID", "hour", "feature", "VALUENUM"]])

        if total_rows % 5_000_000 == 0:
            print(f"    Processed {total_rows/1e6:.0f}M rows, kept {kept_rows:,}")

    print(f"  Total rows processed: {total_rows:,} | Kept: {kept_rows:,}")

    if not all_chunks:
        return pd.DataFrame()

    chart_df = pd.concat(all_chunks, ignore_index=True)

    # Aggregate: median per (stay, hour, feature) — robust to outliers
    pivot = (
        chart_df
        .groupby(["ICUSTAY_ID", "hour", "feature"])["VALUENUM"]
        .median()
        .reset_index()
        .pivot_table(index=["ICUSTAY_ID", "hour"], columns="feature",
                     values="VALUENUM", aggfunc="median")
        .reset_index()
    )
    pivot.columns.name = None

    # Rename Temp_C → Temp
    if "Temp_C" in pivot.columns:
        pivot.rename(columns={"Temp_C": "Temp"}, inplace=True)

    # Add patient_id
    icustay_to_pid = icustay_metadata.set_index("ICUSTAY_ID")["patient_id"]
    pivot["patient_id"] = pivot["ICUSTAY_ID"].map(icustay_to_pid)

    print(f"  Vitals extracted: {len(pivot):,} patient-hour rows")
    return pivot


# ─────────────────────────────────────────────────
# 3. EXTRACT LABEVENTS
# ─────────────────────────────────────────────────

def extract_labevents(
    mimic_dir: str,
    hadm_ids: List[int],
    icustay_metadata: pd.DataFrame,
    chunksize: int = 500_000
) -> pd.DataFrame:
    """
    Extract lab values from LABEVENTS, aligned to ICU hours.

    LABEVENTS does not have ICUSTAY_ID — we join via HADM_ID
    and then align to the ICU stay's time window.
    """
    mimic_path = Path(mimic_dir)
    lab_path = mimic_path / "LABEVENTS.csv"

    if not lab_path.exists():
        print("  LABEVENTS.csv not found — skipping labs")
        return pd.DataFrame()

    target_ids = set(all_lab_itemids())
    target_hadms = set(hadm_ids)
    itemid_to_feat = build_itemid_to_feature()

    # Build hadm → icu stay info lookup
    hadm_lookup = icustay_metadata.set_index("HADM_ID")[[
        "ICUSTAY_ID", "patient_id", "INTIME", "OUTTIME"
    ]].drop_duplicates()

    print(f"  Extracting LABEVENTS...")

    all_chunks = []
    for chunk in pd.read_csv(
        lab_path,
        usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "FLAG"],
        parse_dates=["CHARTTIME"],
        chunksize=chunksize,
        low_memory=False
    ):
        chunk = chunk[
            (chunk["HADM_ID"].isin(target_hadms)) &
            (chunk["ITEMID"].isin(target_ids)) &
            (chunk["VALUENUM"].notna())
        ].copy()

        if chunk.empty:
            continue

        # Join ICU stay info
        chunk = chunk.merge(
            hadm_lookup.reset_index(),
            on="HADM_ID", how="inner"
        )

        # Filter to within ICU stay time window
        chunk = chunk[
            (chunk["CHARTTIME"] >= chunk["INTIME"]) &
            (chunk["CHARTTIME"] <= chunk["OUTTIME"])
        ]

        # Hour offset
        chunk["hour"] = (
            (chunk["CHARTTIME"] - chunk["INTIME"]).dt.total_seconds() / 3600
        ).round().astype(int)

        chunk["feature"] = chunk["ITEMID"].map(itemid_to_feat)

        all_chunks.append(chunk[["ICUSTAY_ID", "patient_id", "hour",
                                  "feature", "VALUENUM"]])

    if not all_chunks:
        print("  No lab data extracted")
        return pd.DataFrame()

    lab_df = pd.concat(all_chunks, ignore_index=True)

    # Pivot to wide format
    pivot = (
        lab_df
        .groupby(["ICUSTAY_ID", "patient_id", "hour", "feature"])["VALUENUM"]
        .median()
        .reset_index()
        .pivot_table(index=["ICUSTAY_ID", "patient_id", "hour"],
                     columns="feature", values="VALUENUM", aggfunc="median")
        .reset_index()
    )
    pivot.columns.name = None

    print(f"  Labs extracted: {len(pivot):,} patient-hour rows")
    return pivot


# ─────────────────────────────────────────────────
# 4. EXTRACT VASOPRESSOR FLAGS
# ─────────────────────────────────────────────────

def extract_vasopressors(
    mimic_dir: str,
    icustay_ids: List[int],
    icustay_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Extract hourly vasopressor flag (True/False) per ICU stay."""
    mimic_path = Path(mimic_dir)
    all_vaso_ids = []
    for ids in VASOPRESSOR_ITEMIDS.values():
        all_vaso_ids.extend(ids)
    all_vaso_ids = set(all_vaso_ids)

    target_stays = set(icustay_ids)
    intime_lookup = icustay_metadata.set_index("ICUSTAY_ID")["INTIME"]

    rows = []
    for source in ["INPUTEVENTS_MV", "INPUTEVENTS_CV"]:
        path = mimic_path / f"{source}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path,
                             usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME"],
                             parse_dates=["STARTTIME", "ENDTIME"])
            df = df[
                df["ICUSTAY_ID"].isin(target_stays) &
                df["ITEMID"].isin(all_vaso_ids)
            ]
            rows.append(df)
        except Exception as e:
            print(f"  Warning: {source} vasopressor extraction failed: {e}")

    if not rows:
        return pd.DataFrame()

    vaso_df = pd.concat(rows, ignore_index=True)

    # Expand drug administrations to hourly flags
    hourly_flags = []
    for _, row in vaso_df.iterrows():
        stay_id = int(row["ICUSTAY_ID"])
        intime = intime_lookup.get(stay_id)
        if pd.isna(intime):
            continue
        start_h = max(0, int((row["STARTTIME"] - intime).total_seconds() / 3600))
        end_h   = min(168, int((row["ENDTIME"]   - intime).total_seconds() / 3600))
        for h in range(start_h, end_h + 1):
            hourly_flags.append({"ICUSTAY_ID": stay_id, "hour": h,
                                  "on_vasopressor": True})

    if not hourly_flags:
        return pd.DataFrame()

    return (
        pd.DataFrame(hourly_flags)
        .groupby(["ICUSTAY_ID", "hour"])["on_vasopressor"]
        .any()
        .reset_index()
    )


# ─────────────────────────────────────────────────
# 5. BUILD HOURLY SCAFFOLD + MERGE ALL FEATURES
# ─────────────────────────────────────────────────

def build_hourly_scaffold(icustay_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Create a complete hour-by-hour scaffold for each ICU stay.
    One row per (patient_id, hour) from hour 0 to min(LOS, 168h).
    """
    rows = []
    for _, stay in icustay_metadata.iterrows():
        max_hour = min(int(stay["icu_los_hours"]), 168)
        for h in range(max_hour + 1):
            rows.append({
                "patient_id": stay["patient_id"],
                "ICUSTAY_ID": int(stay["ICUSTAY_ID"]),
                "HADM_ID":    int(stay["HADM_ID"]),
                "hour":       h,
                "Age":        stay["Age"],
                "Gender":     stay["Gender"],
                "Unit1":      stay["Unit1"],
                "Unit2":      stay["Unit2"],
                "HospAdmTime": stay["HospAdmTime"],
                "ICULOS":     h,
                "hospital_expire_flag": int(stay.get("HOSPITAL_EXPIRE_FLAG", 0)),
            })
    df = pd.DataFrame(rows)
    print(f"  Hourly scaffold: {len(df):,} rows for {df['patient_id'].nunique():,} stays")
    return df


def apply_plausibility_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values to physiologically plausible ranges."""
    for col, (lo, hi) in VALUE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ─────────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────────

def extract_mimic_features(
    mimic_dir: str,
    output_dir: str,
    max_stays: Optional[int] = None,
    generate_labels: bool = True
) -> pd.DataFrame:
    """
    Full MIMIC-III extraction pipeline.

    Steps:
        1. Load ICU stay metadata (ICUSTAYS + PATIENTS + ADMISSIONS)
        2. Build hourly scaffold
        3. Extract vital signs (CHARTEVENTS)
        4. Extract lab values (LABEVENTS)
        5. Extract vasopressor flags (INPUTEVENTS)
        6. Merge all features onto scaffold
        7. Clip physiological outliers
        8. Optionally generate Sepsis-3 labels
        9. Save to CSV

    Returns:
        Complete feature DataFrame (compatible with preprocessing.py)
    """
    print("\n" + "=" * 60)
    print("MIMIC-III FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Metadata ──────────────────────────────────────────────
    print("\n[1/6] Loading ICU stay metadata...")
    icu_meta = load_icu_metadata(mimic_dir, max_stays=max_stays)
    icustay_ids = icu_meta["ICUSTAY_ID"].tolist()
    hadm_ids = icu_meta["HADM_ID"].tolist()

    # ── Step 2: Scaffold ──────────────────────────────────────────────
    print("\n[2/6] Building hourly scaffold...")
    scaffold = build_hourly_scaffold(icu_meta)

    # ── Step 3: Vitals ────────────────────────────────────────────────
    print("\n[3/6] Extracting vital signs (CHARTEVENTS)...")
    vitals_df = extract_chartevents(mimic_dir, icustay_ids, icu_meta)

    # ── Step 4: Labs ──────────────────────────────────────────────────
    print("\n[4/6] Extracting laboratory values (LABEVENTS)...")
    labs_df = extract_labevents(mimic_dir, hadm_ids, icu_meta)

    # ── Step 5: Vasopressors ──────────────────────────────────────────
    print("\n[5/6] Extracting vasopressor flags...")
    vaso_df = extract_vasopressors(mimic_dir, icustay_ids, icu_meta)

    # ── Step 6: Merge all features ────────────────────────────────────
    print("\n[6/6] Merging all features onto scaffold...")
    df = scaffold.copy()

    if not vitals_df.empty:
        vitals_cols = [c for c in vitals_df.columns
                       if c not in ("ICUSTAY_ID",)]
        df = df.merge(
            vitals_df[["ICUSTAY_ID", "hour"] + [c for c in vitals_cols
                                                  if c not in ("patient_id",)]],
            on=["ICUSTAY_ID", "hour"], how="left"
        )

    if not labs_df.empty:
        lab_cols = [c for c in labs_df.columns
                    if c not in ("ICUSTAY_ID", "patient_id", "hour")]
        df = df.merge(
            labs_df[["ICUSTAY_ID", "hour"] + lab_cols],
            on=["ICUSTAY_ID", "hour"], how="left"
        )

    if not vaso_df.empty:
        df = df.merge(
            vaso_df, on=["ICUSTAY_ID", "hour"], how="left"
        )
        df["on_vasopressor"] = df["on_vasopressor"].fillna(False)

    # Apply plausibility bounds
    df = apply_plausibility_bounds(df)

    # ── Sepsis-3 labels ───────────────────────────────────────────────
    if generate_labels:
        print("\n[Labels] Generating Sepsis-3 labels...")
        sys.path.insert(0, str(Path(__file__).parent))
        from sepsis3_labels import detect_suspected_infection, generate_sepsis3_labels

        icu_meta_loaded = pd.read_csv(
            Path(mimic_dir) / "ICUSTAYS.csv",
            parse_dates=["INTIME", "OUTTIME"]
        )
        si_df = detect_suspected_infection(mimic_dir, icu_meta_loaded)
        df = generate_sepsis3_labels(df, si_df)
    else:
        df["SepsisLabel"] = 0

    # ── Save ──────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "mimic_hourly_features.csv")
    df.to_csv(out_path, index=False)

    n_patients = df["patient_id"].nunique()
    sepsis_prev = df.groupby("patient_id")["SepsisLabel"].max().mean()

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Rows:             {len(df):,}")
    print(f"  Patients:         {n_patients:,}")
    print(f"  Columns:          {len(df.columns)}")
    print(f"  Sepsis prevalence:{sepsis_prev:.1%}")
    print(f"  Saved to:         {out_path}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-III features for sepsis prediction"
    )
    parser.add_argument(
        "--mimic-dir", required=True,
        help="Path to directory containing MIMIC-III CSV files"
    )
    parser.add_argument(
        "--output-dir", default="./data/mimic",
        help="Output directory for extracted CSVs"
    )
    parser.add_argument(
        "--max-stays", type=int, default=None,
        help="Limit to first N ICU stays (for testing)"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Skip Sepsis-3 label generation"
    )
    args = parser.parse_args()

    df = extract_mimic_features(
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        max_stays=args.max_stays,
        generate_labels=not args.no_labels
    )
