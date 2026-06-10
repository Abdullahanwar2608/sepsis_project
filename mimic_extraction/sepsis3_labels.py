"""
MIMIC-III Sepsis-3 Label Generation
=====================================
Implements the Sepsis-3 consensus definition (Singer et al., JAMA 2016):

    Sepsis = Suspected Infection + Acute Organ Dysfunction (SOFA ≥ 2)

Suspected Infection is defined as:
    Blood culture ordered AND antibiotic administered within ±1 day
    (Seymour et al., NEJM 2016 / Rhee et al., JAMA 2017)

SOFA Score components:
    1. Respiratory:     PaO2/FiO2 ratio
    2. Coagulation:     Platelets (×10³/μL)
    3. Liver:           Bilirubin total (mg/dL)
    4. Cardiovascular:  MAP or vasopressor dose
    5. CNS:             Glasgow Coma Scale (GCS)
    6. Renal:           Creatinine (mg/dL) or Urine output (mL/24h)

Usage:
    Run standalone (after extract_mimic.py has produced the CSV files):
        python sepsis3_labels.py --mimic-dir /path/to/mimic_csvs --output-dir ./data

    Or import and call:
        from mimic_extraction.sepsis3_labels import generate_sepsis3_labels
        labels_df = generate_sepsis3_labels(mimic_dir)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from item_ids import (
    VASOPRESSOR_ITEMIDS, ANTIBIOTIC_ITEMIDS, URINE_OUTPUT_ITEMIDS,
    VALUE_BOUNDS
)


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def sofa_respiratory(pao2_fio2: float) -> int:
    """PaO2/FiO2 ratio → SOFA respiratory score (0-4)."""
    if pd.isna(pao2_fio2):
        return 0
    if pao2_fio2 >= 400: return 0
    if pao2_fio2 >= 300: return 1
    if pao2_fio2 >= 200: return 2
    if pao2_fio2 >= 100: return 3
    return 4

def sofa_coagulation(platelets: float) -> int:
    """Platelets ×10³/μL → SOFA coagulation score (0-4)."""
    if pd.isna(platelets):
        return 0
    if platelets >= 150: return 0
    if platelets >= 100: return 1
    if platelets >= 50:  return 2
    if platelets >= 20:  return 3
    return 4

def sofa_liver(bilirubin: float) -> int:
    """Total bilirubin mg/dL → SOFA liver score (0-4)."""
    if pd.isna(bilirubin):
        return 0
    if bilirubin < 1.2:  return 0
    if bilirubin < 2.0:  return 1
    if bilirubin < 6.0:  return 2
    if bilirubin < 12.0: return 3
    return 4

def sofa_cardiovascular(map_val: float, on_vasopressor: bool) -> int:
    """MAP + vasopressor → SOFA cardiovascular score (0-4)."""
    if on_vasopressor:
        return 3  # simplified: any vasopressor = score ≥ 3
    if pd.isna(map_val):
        return 0
    if map_val >= 70: return 0
    return 1

def sofa_cns(gcs: float) -> int:
    """GCS total → SOFA CNS score (0-4)."""
    if pd.isna(gcs):
        return 0
    if gcs >= 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6:  return 3
    return 4

def sofa_renal(creatinine: float, urine_24h: Optional[float] = None) -> int:
    """Creatinine mg/dL (+ urine output) → SOFA renal score (0-4)."""
    score = 0
    if not pd.isna(creatinine):
        if creatinine >= 5.0:   score = 4
        elif creatinine >= 3.5: score = 3
        elif creatinine >= 2.0: score = 2
        elif creatinine >= 1.2: score = 1
    if urine_24h is not None and not pd.isna(urine_24h):
        if urine_24h < 200:   score = max(score, 4)
        elif urine_24h < 500: score = max(score, 3)
    return score


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def compute_hourly_sofa(
    vitals_labs_df: pd.DataFrame,
    vasopressor_df: Optional[pd.DataFrame] = None,
    urine_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute SOFA score for each patient-hour.

    Args:
        vitals_labs_df: DataFrame with columns:
            patient_id, hour, MAP, Platelets, Bilirubin_total,
            Creatinine, PaO2, FiO2, GCS_Eye, GCS_Verbal, GCS_Motor
        vasopressor_df: Optional. patient_id, hour, on_vasopressor (bool)
        urine_df: Optional. patient_id, hour, urine_24h (rolling 24h total)

    Returns:
        DataFrame with patient_id, hour, sofa_total, sofa_{component}
    """
    df = vitals_labs_df.copy()

    # ── GCS total ──────────────────────────────────────────────────────
    gcs_cols = [c for c in ["GCS_Eye", "GCS_Verbal", "GCS_Motor"] if c in df.columns]
    if gcs_cols:
        df["GCS_total"] = df[gcs_cols].sum(axis=1)
        df["GCS_total"] = df["GCS_total"].where(
            df[gcs_cols].notna().all(axis=1), other=np.nan
        )
    else:
        df["GCS_total"] = np.nan

    # ── PaO2/FiO2 ratio ───────────────────────────────────────────────
    if "PaO2" in df.columns and "FiO2" in df.columns:
        df["FiO2_norm"] = df["FiO2"].apply(
            lambda x: x / 100.0 if (not pd.isna(x) and x > 1.0) else x
        )
        df["PF_ratio"] = df["PaO2"] / df["FiO2_norm"].replace(0, np.nan)
    else:
        df["PF_ratio"] = np.nan

    # ── Vasopressor flag (join if provided) ───────────────────────────
    if vasopressor_df is not None and not vasopressor_df.empty:
        df = df.merge(
            vasopressor_df[["patient_id", "hour", "on_vasopressor"]],
            on=["patient_id", "hour"], how="left"
        )
        df["on_vasopressor"] = df["on_vasopressor"].fillna(False)
    else:
        df["on_vasopressor"] = False

    # ── Urine output (join if provided) ───────────────────────────────
    if urine_df is not None and not urine_df.empty:
        df = df.merge(
            urine_df[["patient_id", "hour", "urine_24h"]],
            on=["patient_id", "hour"], how="left"
        )
    else:
        df["urine_24h"] = np.nan

    # ── Compute SOFA components ────────────────────────────────────────
    def row_sofa(row):
        resp  = sofa_respiratory(row.get("PF_ratio", np.nan))
        coag  = sofa_coagulation(row.get("Platelets", np.nan))
        liver = sofa_liver(row.get("Bilirubin_total", np.nan))
        cardio= sofa_cardiovascular(row.get("MAP", np.nan), row.get("on_vasopressor", False))
        cns   = sofa_cns(row.get("GCS_total", np.nan))
        renal = sofa_renal(row.get("Creatinine", np.nan), row.get("urine_24h", np.nan))
        total = resp + coag + liver + cardio + cns + renal
        return pd.Series({
            "sofa_respiratory": resp,
            "sofa_coagulation": coag,
            "sofa_liver":       liver,
            "sofa_cardiovascular": cardio,
            "sofa_cns":         cns,
            "sofa_renal":       renal,
            "sofa_total":       total,
        })

    sofa_cols = df.apply(row_sofa, axis=1)
    result = pd.concat([
        df[["patient_id", "hour"]].reset_index(drop=True),
        sofa_cols.reset_index(drop=True)
    ], axis=1)

    return result


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def detect_suspected_infection(
    mimic_dir: str,
    icustays_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect suspected infection = antibiotic + blood culture within ±1 day.

    Returns DataFrame: patient_id, suspected_infection_hour (ICU hours offset)
    """
    print("  [Sepsis-3] Detecting suspected infection...")

    mimic_path = Path(mimic_dir)

    # ── Load MICROBIOLOGYEVENTS (blood cultures) ────────────────────────
    micro_path = mimic_path / "MICROBIOLOGYEVENTS.csv"
    if not micro_path.exists():
        print("    MICROBIOLOGYEVENTS.csv not found — skipping culture criterion")
        cultures_df = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "CHARTTIME"])
    else:
        cultures_df = pd.read_csv(micro_path, usecols=[
            "SUBJECT_ID", "HADM_ID", "SPEC_TYPE_DESC", "CHARTTIME"
        ], parse_dates=["CHARTTIME"])
        blood_cultures = cultures_df[
            cultures_df["SPEC_TYPE_DESC"].str.contains(
                "blood|BLOOD", case=False, na=False
            )
        ].copy()
        print(f"    Blood cultures: {len(blood_cultures):,} rows")

    # ── Load INPUTEVENTS for antibiotics ───────────────────────────────
    antibiotic_rows = []
    for source in ["INPUTEVENTS_MV", "INPUTEVENTS_CV"]:
        path = mimic_path / f"{source}.csv"
        if not path.exists():
            continue
        cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME",
                "ENDTIME", "ITEMID"]
        try:
            df = pd.read_csv(path, usecols=cols,
                             parse_dates=["STARTTIME", "ENDTIME"])
            df = df[df["ITEMID"].isin(ANTIBIOTIC_ITEMIDS)]
            antibiotic_rows.append(df)
            print(f"    {source}: {len(df):,} antibiotic rows")
        except Exception as e:
            print(f"    Warning: {source} failed: {e}")

    if not antibiotic_rows:
        print("    No antibiotic data found")
        return pd.DataFrame(columns=["patient_id", "suspected_infection_hour"])

    antibiotics_df = pd.concat(antibiotic_rows, ignore_index=True)

    # ── Match to ICU stays ─────────────────────────────────────────────
    results = []
    icu_map = icustays_df.set_index("ICUSTAY_ID")

    for hadm_id, abx_group in antibiotics_df.groupby("HADM_ID"):
        icu_rows = icustays_df[icustays_df["HADM_ID"] == hadm_id]
        if icu_rows.empty:
            continue

        icu_row = icu_rows.iloc[0]
        icu_intime = pd.to_datetime(icu_row["INTIME"])
        patient_id = f"mimic_{int(icu_row['ICUSTAY_ID'])}"

        first_abx_time = abx_group["STARTTIME"].min()
        if pd.isna(first_abx_time):
            continue

        if len(blood_cultures) > 0:
            pt_cultures = blood_cultures[blood_cultures["HADM_ID"] == hadm_id]
            if len(pt_cultures) > 0:
                culture_times = pd.to_datetime(pt_cultures["CHARTTIME"])
                close_culture = culture_times[
                    abs((culture_times - first_abx_time).dt.total_seconds()) <= 86400
                ]
                has_culture = len(close_culture) > 0
            else:
                has_culture = False
        else:
            has_culture = True

        if has_culture:
            offset_hours = (first_abx_time - icu_intime).total_seconds() / 3600
            results.append({
                "patient_id": patient_id,
                "suspected_infection_hour": max(0, offset_hours)
            })

    si_df = pd.DataFrame(results)
    print(f"    Suspected infection identified: {len(si_df):,} ICU stays")
    return si_df


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def generate_sepsis3_labels(
    vitals_labs_df: pd.DataFrame,
    suspected_infection_df: pd.DataFrame,
    sofa_increase_threshold: int = 2,
    label_horizon_hours: int = 6,
) -> pd.DataFrame:
    """
    Generate Sepsis-3 labels for each patient-hour.

    Sepsis onset = first hour where:
        1. Suspected infection is present (antibiotic + culture)
        2. SOFA score increased by ≥ threshold from baseline

    Args:
        vitals_labs_df: Full patient-hour DataFrame with lab/vital columns
        suspected_infection_df: patient_id, suspected_infection_hour
        sofa_increase_threshold: SOFA increase ≥ this = organ dysfunction
        label_horizon_hours: Hours before onset to mark as "pre-sepsis"

    Returns:
        vitals_labs_df with added 'SepsisLabel' column (0/1)
    """
    print("  [Sepsis-3] Computing SOFA scores...")

    sofa_df = compute_hourly_sofa(vitals_labs_df)

    df = vitals_labs_df.merge(
        sofa_df.drop(columns=["sofa_respiratory","sofa_coagulation",
                               "sofa_liver","sofa_cardiovascular",
                               "sofa_cns","sofa_renal"], errors="ignore"),
        on=["patient_id", "hour"], how="left"
    )

    # ── Compute SOFA increase from admission baseline ──────────────────
    df = df.sort_values(["patient_id", "hour"])

    def compute_sofa_increase(group):
        sofa = group["sofa_total"].fillna(0)
        baseline_window = sofa[group["hour"] <= 24]
        baseline = baseline_window.min() if len(baseline_window) > 0 else 0
        group = group.copy()
        group["sofa_increase"] = sofa - baseline
        return group

    df = df.groupby("patient_id", group_keys=False).apply(compute_sofa_increase)

    # ── Apply Sepsis-3 criteria ────────────────────────────────────────
    print("  [Sepsis-3] Applying Sepsis-3 criteria...")
    si_lookup = suspected_infection_df.set_index("patient_id")["suspected_infection_hour"]

    df["SepsisLabel"] = 0
    df["sepsis_onset_hour"] = np.nan

    sepsis_count = 0
    for pid, group in df.groupby("patient_id"):
        if pid not in si_lookup.index:
            continue  # No suspected infection → cannot be sepsis-3

        si_hour = si_lookup[pid]
        organ_dysfunction = group[
            (group["sofa_increase"] >= sofa_increase_threshold) &
            (group["hour"] >= max(0, si_hour - 24)) &
            (group["hour"] <= si_hour + 24)
        ]

        if organ_dysfunction.empty:
            continue

        onset_hour = organ_dysfunction["hour"].min()
        df.loc[
            (df["patient_id"] == pid) & (df["hour"] >= onset_hour),
            "SepsisLabel"
        ] = 1
        df.loc[df["patient_id"] == pid, "sepsis_onset_hour"] = onset_hour
        sepsis_count += 1

    n_patients = df["patient_id"].nunique()
    prev = df.groupby("patient_id")["SepsisLabel"].max().mean()
    print(f"  [Sepsis-3] Sepsis patients: {sepsis_count:,} / {n_patients:,} "
          f"({prev:.1%} prevalence)")

    return df


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Sepsis-3 labels from MIMIC-III CSV files"
    )
    parser.add_argument("--mimic-dir", required=True,
                        help="Directory containing MIMIC-III CSV files")
    parser.add_argument("--vitals-labs", required=True,
                        help="Path to extracted vitals/labs CSV (from extract_mimic.py)")
    parser.add_argument("--output-dir", default="./data",
                        help="Where to save labeled output CSV")
    args = parser.parse_args()

    print("Loading vitals/labs...")
    vl_df = pd.read_csv(args.vitals_labs)

    print("Loading ICU stays...")
    icustays_df = pd.read_csv(
        os.path.join(args.mimic_dir, "ICUSTAYS.csv"),
        parse_dates=["INTIME", "OUTTIME"]
    )

    si_df = detect_suspected_infection(args.mimic_dir, icustays_df)

    labeled_df = generate_sepsis3_labels(vl_df, si_df)

    out_path = os.path.join(args.output_dir, "mimic_sepsis3_labeled.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    labeled_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {labeled_df.shape}")
    print(labeled_df["SepsisLabel"].value_counts())
