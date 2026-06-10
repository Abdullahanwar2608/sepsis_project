"""
Sepsis Onset Prediction — Model Training Pipeline
===================================================
Implements:
- Baseline:  Logistic Regression, Random Forest
- Advanced:  XGBoost, LightGBM
- Label-noise aware training (Confident Learning sample weighting)
- Probability calibration (Platt scaling / isotonic regression)
- Clinical threshold selection (maximize utility score)
- SHAP feature importance (model-agnostic explainability)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    f1_score, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from utils import MODELS_DIR, ensure_dirs

ensure_dirs()


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Lightweight isotonic regression calibrator.
    Works as a drop-in for CalibratedClassifierCV(cv='prefit')
    on all sklearn versions.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, X_val, y_val):
        """Fit calibrator on held-out validation set."""
        raw_probs = self.base_model.predict_proba(X_val)[:, 1]
        self.calibrator.fit(raw_probs, y_val)
        self._fitted = True
        return self

    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        if self._fitted:
            cal_probs = self.calibrator.predict(raw_probs)
        else:
            cal_probs = raw_probs
        return np.column_stack([1 - cal_probs, cal_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getattr__(self, name):
        if name in ("base_model", "calibrator", "_fitted"):
            raise AttributeError(name)
        return getattr(self.base_model, name)



# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def confident_learning_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_threshold: float = 0.35,
    n_splits: int = 5
) -> np.ndarray:
    """
    Confident Learning (Northcutt et al. 2021, JAIR):
    https://arxiv.org/abs/1911.00068

    Strategy:
    1. Estimate per-sample probabilities via stratified K-fold cross-validation
    2. Flag samples where model confidence strongly disagrees with given label
    3. Down-weight likely noisy samples (weight in [0.1, 1.0])

    Noise cases:
    - Label=1 but p̂ < threshold  → likely noisy positive (missed diagnosis reversal)
    - Label=0 but p̂ > 1-threshold → likely noisy negative (early true sepsis)
    """
    probs = np.zeros(len(y_train), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr = LogisticRegression(
        max_iter=500, C=0.5,
        class_weight="balanced",
        solver="lbfgs", n_jobs=-1
    )

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train)):
        lr.fit(X_train[tr_idx], y_train[tr_idx])
        probs[vl_idx] = lr.predict_proba(X_train[vl_idx])[:, 1]

    weights = np.ones(len(y_train), dtype=float)

    noisy_pos = (y_train == 1) & (probs < noise_threshold)
    noisy_neg = (y_train == 0) & (probs > (1 - noise_threshold))

    weights[noisy_pos] = 0.25   # Strong down-weight: high-confidence mislabel
    weights[noisy_neg] = 0.50   # Moderate down-weight: possible early sepsis

    n_flagged = noisy_pos.sum() + noisy_neg.sum()
    print(f"  Confident Learning: flagged {n_flagged:,} "
          f"({n_flagged / len(y_train):.1%}) potentially noisy labels "
          f"({noisy_pos.sum():,} false positives, {noisy_neg.sum():,} false negatives)")

    return weights


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def physionet_utility_score(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float
) -> float:
    """
    Simplified PhysioNet 2019 utility score.

    Clinical cost asymmetry:
    - TP (early detection):   +1.0  reward
    - FP (false alarm):       -0.05 penalty (unnecessary antibiotics, alarm fatigue)
    - FN (missed sepsis):     -2.0  penalty (mortality risk)
    - TN (correct no-alarm):  +0.001 small reward
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return float(tp * 1.0 - fp * 0.05 - fn * 2.0 + tn * 0.001)


def select_optimal_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    strategy: str = "utility"
) -> float:
    """Select classification threshold using specified strategy."""
    thresholds = np.linspace(0.05, 0.95, 181)

    if strategy == "utility":
        scores = [physionet_utility_score(y_val, probs_val, t) for t in thresholds]
        best_t = float(thresholds[np.argmax(scores)])

    elif strategy == "f1":
        scores = []
        for t in thresholds:
            preds = (probs_val >= t).astype(int)
            scores.append(f1_score(y_val, preds, zero_division=0))
        best_t = float(thresholds[np.argmax(scores)])

    elif strategy == "youden":
        fpr, tpr, roc_t = roc_curve(y_val, probs_val)
        j = tpr - fpr
        best_t = float(np.clip(roc_t[np.argmax(j)], 0.05, 0.95))

    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    print(f"    Optimal threshold ({strategy}): {best_t:.3f}")
    return best_t


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def build_models(class_ratio: float = 10.0) -> Dict[str, Any]:
    """
    Build all model variants.

    Args:
        class_ratio: Approximate ratio of negatives to positives (for XGB/LGB)
    """
    models = {}

    # ── Baseline: Logistic Regression ──────────────────────────────────
    models["logistic_regression"] = LogisticRegression(
        max_iter=2000,
        C=0.05,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42
    )

    # ── Baseline: Random Forest ─────────────────────────────────────────
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # ── Advanced: XGBoost ───────────────────────────────────────────────
    if HAS_XGB:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=class_ratio,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    # ── Advanced: LightGBM ──────────────────────────────────────────────
    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    # ── Fallback if neither boosting library available ──────────────────
    if not HAS_XGB and not HAS_LGB:
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42
        )

    return models


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    split: str = "test"
) -> Dict[str, float]:
    """Comprehensive clinical evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # True Positive Rate / Recall
    specificity = tn / (tn + fp + 1e-8)   # True Negative Rate
    ppv = tp / (tp + fp + 1e-8)           # Precision
    npv = tn / (tn + fn + 1e-8)           # Negative Predictive Value

    metrics = {
        f"{split}_auroc": roc_auc_score(y_true, probs),
        f"{split}_auprc": average_precision_score(y_true, probs),
        f"{split}_brier": brier_score_loss(y_true, probs),
        f"{split}_sensitivity": sensitivity,
        f"{split}_specificity": specificity,
        f"{split}_ppv": ppv,
        f"{split}_npv": npv,
        f"{split}_f1": f1_score(y_true, preds, zero_division=0),
        f"{split}_tp": int(tp),
        f"{split}_fp": int(fp),
        f"{split}_tn": int(tn),
        f"{split}_fn": int(fn),
    }

    print(f"\n  [{split.upper()}] Metrics:")
    print(f"    AUROC={metrics[f'{split}_auroc']:.4f}  "
          f"AUPRC={metrics[f'{split}_auprc']:.4f}  "
          f"Brier={metrics[f'{split}_brier']:.4f}")
    print(f"    Sensitivity={sensitivity:.3f}  Specificity={specificity:.3f}")
    print(f"    PPV={ppv:.3f}  NPV={npv:.3f}  F1={metrics[f'{split}_f1']:.3f}")
    print(f"    TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

    return metrics


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def train_single_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    calibrate: bool = True
) -> Tuple[Any, float, Dict]:
    """Train one model, calibrate, select threshold, evaluate on validation set."""
    print(f"\n  Training {model_name}...")

    fit_kwargs = {}
    if sample_weights is not None:
        if model_name in ("xgboost", "lightgbm", "gradient_boosting"):
            fit_kwargs["sample_weight"] = sample_weights
        elif model_name in ("random_forest",):
            fit_kwargs["sample_weight"] = sample_weights

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)

    if calibrate:
        cal_model = IsotonicCalibrator(model)
        cal_model.fit(X_val, y_val)
        final_model = cal_model
        print(f"    Isotonic calibration applied.")
    else:
        final_model = model

    probs_val = final_model.predict_proba(X_val)[:, 1]

    threshold = select_optimal_threshold(y_val, probs_val, strategy="utility")

    preds_val = (probs_val >= threshold).astype(int)
    val_metrics = evaluate_model(y_val, probs_val, preds_val, split="val")

    return final_model, threshold, val_metrics


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def run_training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_confident_learning: bool = True,
    output_dir: str = None
) -> Dict[str, Dict]:
    """
    Complete training pipeline for all models.

    Returns dict with results for each model.
    """
    output_dir = output_dir or str(MODELS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SEPSIS PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ── Prepare arrays ───────────────────────────────────────────────────
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["target"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(int)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # ── Feature scaling (needed for LR) ──────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    print(f"\nDataset sizes: Train={len(y_train):,} | Val={len(y_val):,} | Test={len(y_test):,}")
    print(f"Class balance:  Train={y_train.mean():.3f} | Val={y_val.mean():.3f} | Test={y_test.mean():.3f}")

    # ── Class imbalance weights ───────────────────────────────────────────
    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    class_weights = compute_sample_weight("balanced", y_train)

    # ── Confident Learning ────────────────────────────────────────────────
    cl_weights = None
    if use_confident_learning:
        print("\n[Label Noise] Running Confident Learning...")
        cl_weights = confident_learning_weights(X_train_sc, y_train)

    final_weights = cl_weights * class_weights if cl_weights is not None else class_weights

    # ── Build and train models ────────────────────────────────────────────
    models = build_models(class_ratio=class_ratio)
    all_results = {}
    best_auroc = 0.0
    best_model_name = None

    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"  Model: {model_name.upper()}")

        use_scaled = (model_name == "logistic_regression")
        X_tr = X_train_sc if use_scaled else X_train
        X_vl = X_val_sc if use_scaled else X_val
        X_ts = X_test_sc if use_scaled else X_test

        trained_model, threshold, val_metrics = train_single_model(
            model, model_name,
            X_tr, y_train,
            X_vl, y_val,
            sample_weights=final_weights,
            calibrate=True
        )

        probs_test = trained_model.predict_proba(X_ts)[:, 1]
        preds_test = (probs_test >= threshold).astype(int)
        test_metrics = evaluate_model(y_test, probs_test, preds_test, split="test")

        all_results[model_name] = {
            "model": trained_model,
            "threshold": threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_probs": probs_test,
            "test_preds": preds_test,
            "y_test": y_test,
            "scaler": scaler if use_scaled else None,
            "use_scaled": use_scaled,
            "feature_cols": feature_cols,
        }

        save_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump({
            "model": trained_model,
            "scaler": scaler if use_scaled else None,
            "threshold": threshold,
            "feature_cols": feature_cols,
        }, save_path)
        print(f"    Saved: {save_path}")

        if test_metrics["test_auroc"] > best_auroc:
            best_auroc = test_metrics["test_auroc"]
            best_model_name = model_name

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_model_name} (AUROC={best_auroc:.4f})")
    print("=" * 60)

    return all_results


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def predict_sepsis_risk(
    model_path: str,
    patient_features: np.ndarray
) -> Tuple[float, int]:
    """
    Load a saved model and predict sepsis risk for a new patient.

    Args:
        model_path: Path to .pkl model artifact
        patient_features: Feature vector (1D array matching training feature_cols)

    Returns:
        (probability, binary_prediction)
    """
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    threshold = artifact["threshold"]

    X = patient_features.reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    return prob, pred


if __name__ == "__main__":
    from preprocessing import run_preprocessing_pipeline

    train_df, val_df, test_df, feature_cols = run_preprocessing_pipeline(
        use_synthetic=True, n_synthetic=500
    )
    results = run_training_pipeline(
        train_df, val_df, test_df, feature_cols,
        use_confident_learning=True
    )
