"""
MIMIC-III ItemID Mappings for Sepsis Feature Extraction
========================================================
Maps clinical concepts to their MIMIC-III ITEMIDs.

Sources:
- CHARTEVENTS: bedside monitoring (~1 per hour)
  * CareVue ITEMIDs: < 220000
  * Metavision ITEMIDs: >= 220000
- LABEVENTS: laboratory results (linked via D_LABITEMS)

References:
- PhysioNet MIMIC-III v1.4 documentation
- MIT-LCP mimic-code GitHub repository
"""

# ─────────────────────────────────────────────────
# VITAL SIGN ItemIDs  (from CHARTEVENTS)
# ─────────────────────────────────────────────────

VITAL_ITEMIDS = {

    # Heart Rate (bpm)
    "HR": [211, 220045],

    # Oxygen Saturation SpO2 (%)
    "O2Sat": [646, 220277],

    # Temperature Celsius — prefer C; convert F if C missing
    "Temp_C": [676, 223762],
    "Temp_F": [678, 223761],   # will be converted: (F-32)*5/9

    # Systolic Blood Pressure (mmHg)
    "SBP": [51, 220179, 225309],

    # Diastolic Blood Pressure (mmHg)
    "DBP": [8368, 220180, 225310],

    # Mean Arterial Pressure (mmHg)
    "MAP": [52, 220052, 225312, 224322],

    # Respiratory Rate (breaths/min)
    "Resp": [615, 618, 220210, 224690],

    # End-tidal CO2 (mmHg) — rarely documented
    "EtCO2": [1817, 228640],

    # GCS components (for SOFA CNS score)
    "GCS_Eye":    [184, 220739],
    "GCS_Verbal": [723, 223900],
    "GCS_Motor":  [454, 223901],
}

# ─────────────────────────────────────────────────
# LABORATORY ItemIDs  (from LABEVENTS via D_LABITEMS)
# ─────────────────────────────────────────────────

LAB_ITEMIDS = {

    # Arterial Blood Gas
    "BaseExcess": [50802],
    "HCO3":       [50882],        # Bicarbonate
    "FiO2":       [50816],        # Fraction Inspired O2 (also in CHARTEVENTS: 223835)
    "pH":         [50820],
    "PaCO2":      [50818],        # Arterial pCO2
    "PaO2":       [50821],        # Arterial pO2 (for PF ratio)
    "SaO2":       [50817],        # Arterial O2 Saturation

    # Liver
    "AST":              [50878],  # Aspartate aminotransferase
    "Alkalinephos":     [50863],  # Alkaline phosphatase
    "Bilirubin_total":  [50885],
    "Bilirubin_direct": [50883],

    # Renal / Electrolytes
    "BUN":        [51006],        # Blood Urea Nitrogen
    "Calcium":    [50893],
    "Chloride":   [50902],
    "Creatinine": [50912],
    "Magnesium":  [50960],
    "Phosphate":  [50970],
    "Potassium":  [50971],
    "Sodium":     [50983],

    # Metabolic
    "Glucose":  [50931, 50809],   # 50931=serum, 50809=whole blood
    "Lactate":  [50813],

    # Hematology
    "Hct":        [51221],        # Hematocrit
    "Hgb":        [51222],        # Hemoglobin
    "Platelets":  [51265],
    "PTT":        [51275],        # Partial thromboplastin time
    "Fibrinogen": [51214],
    "WBC":        [51301],

    # Cardiac
    "TroponinI": [51002],
}

# FiO2 is also documented in CHARTEVENTS for ventilated patients
FIO2_CHART_ITEMIDS = [3420, 190, 223835, 3422]

# ─────────────────────────────────────────────────
# VASOPRESSOR / ANTIBIOTIC ItemIDs  (INPUTEVENTS_MV / INPUTEVENTS_CV)
# Used for Sepsis-3 suspected infection criterion
# ─────────────────────────────────────────────────

VASOPRESSOR_ITEMIDS = {
    # Norepinephrine
    "norepinephrine": [30047, 30112, 221906],
    # Epinephrine
    "epinephrine":    [30044, 30119, 30309, 221289],
    # Dopamine
    "dopamine":       [30043, 30307, 221662],
    # Dobutamine
    "dobutamine":     [30042, 30306, 221653],
    # Vasopressin
    "vasopressin":    [30051, 222315],
    # Phenylephrine
    "phenylephrine":  [30127, 30128, 221749],
}

ANTIBIOTIC_ITEMIDS = [
    # Vancomycin
    225798, 225849,
    # Piperacillin-Tazobactam
    225893, 225894,
    # Meropenem
    225909,
    # Ceftriaxone
    225855,
    # Metronidazole
    225884,
    # Ciprofloxacin
    225860,
    # Levofloxacin
    225887,
    # Ampicillin
    225843,
    # Cefazolin
    225850,
    # Gentamicin
    225879,
    # Azithromycin
    225846,
    # Clindamycin
    225861,
    # Trimethoprim-Sulfamethoxazole
    225906,
    # Linezolid
    225888,
    # Daptomycin
    225863,
    # Amikacin
    225840,
    # Oxacillin
    225891,
    # Nafcillin
    225890,
    # Colistin
    225862,
]

# ─────────────────────────────────────────────────
# URINE OUTPUT (OUTPUTEVENTS) — for SOFA renal
# ─────────────────────────────────────────────────

URINE_OUTPUT_ITEMIDS = [
    40055,   # Foley
    43175,   # Urine
    40069,   # Urine out Foley
    40094,   # Urine out Void
    40715,   # Urine out Leg Bag
    40473,   # Urine out Suprapubic
    40085,   # Urine out Incontinent
    40057,   # Urine out Straight Cath
    40056,   # GU Irrigant/Urine Volume Out
    40405,   # Urine Out 5oz Cup
    40428,   # Urine out Straight Cath
    40651,   # Urine out Void
    226559,  # Foley (Metavision)
    226560,  # Void (Metavision)
    226561,  # Condom Cath (Metavision)
    226584,  # Ileoconduit (Metavision)
    226563,  # Suprapubic (Metavision)
    226564,  # R Nephrostomy (Metavision)
    226565,  # L Nephrostomy (Metavision)
    226567,  # Straight Cath (Metavision)
    226557,  # R Ureteral Stent (Metavision)
    226558,  # L Ureteral Stent (Metavision)
]

# ─────────────────────────────────────────────────
# MAPPING FROM FEATURE NAME → COLUMN NAME
# (used to standardize output to our pipeline format)
# ─────────────────────────────────────────────────

FEATURE_TO_COLUMN = {
    "HR":               "HR",
    "O2Sat":            "O2Sat",
    "Temp_C":           "Temp",   # merged into Temp
    "Temp_F":           "Temp",   # converted then merged
    "SBP":              "SBP",
    "DBP":              "DBP",
    "MAP":              "MAP",
    "Resp":             "Resp",
    "EtCO2":            "EtCO2",
    "BaseExcess":       "BaseExcess",
    "HCO3":             "HCO3",
    "FiO2":             "FiO2",
    "pH":               "pH",
    "PaCO2":            "PaCO2",
    "SaO2":             "SaO2",
    "AST":              "AST",
    "BUN":              "BUN",
    "Alkalinephos":     "Alkalinephos",
    "Calcium":          "Calcium",
    "Chloride":         "Chloride",
    "Creatinine":       "Creatinine",
    "Bilirubin_direct": "Bilirubin_direct",
    "Glucose":          "Glucose",
    "Lactate":          "Lactate",
    "Magnesium":        "Magnesium",
    "Phosphate":        "Phosphate",
    "Potassium":        "Potassium",
    "Bilirubin_total":  "Bilirubin_total",
    "TroponinI":        "TroponinI",
    "Hct":              "Hct",
    "Hgb":              "Hgb",
    "PTT":              "PTT",
    "WBC":              "WBC",
    "Fibrinogen":       "Fibrinogen",
    "Platelets":        "Platelets",
}

# ─────────────────────────────────────────────────
# PLAUSIBILITY BOUNDS — clip physiologically impossible values
# ─────────────────────────────────────────────────

VALUE_BOUNDS = {
    "HR":             (0, 300),
    "O2Sat":          (0, 100),
    "Temp":           (25, 45),    # Celsius
    "SBP":            (0, 375),
    "DBP":            (0, 375),
    "MAP":            (0, 375),
    "Resp":           (0, 80),
    "FiO2":           (0.21, 1.0),
    "pH":             (6.5, 8.0),
    "PaCO2":          (0, 200),
    "PaO2":           (0, 800),
    "SaO2":           (0, 100),
    "Lactate":        (0, 30),
    "Creatinine":     (0, 30),
    "WBC":            (0, 1000),
    "Platelets":      (0, 10000),
    "Bilirubin_total":(0, 150),
    "Glucose":        (0, 2200),
    "HCO3":           (0, 66),
    "Hct":            (0, 100),
    "Hgb":            (0, 30),
    "BUN":            (0, 300),
    "Potassium":      (0, 15),
    "Sodium":         (0, 200),
    "Calcium":        (0, 20),
    "Magnesium":      (0, 15),
    "Phosphate":      (0, 25),
    "PTT":            (0, 500),
}
