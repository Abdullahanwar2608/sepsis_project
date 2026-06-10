"""
Sepsis Onset Prediction — Evaluation & Visualization
======================================================
Generates:
- ROC curve comparison across models
- Precision-Recall curves
- Calibration curves (reliability diagrams)
- Confusion matrices
- Feature importance (tree models + permutation)
- SHAP summary plots
- PhysioNet utility score vs threshold
- Metrics summary table (CSV + print)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, roc_auc_score,
    average_precision_score
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

from utils import OUTPUTS_DIR, ensure_dirs

ensure_dirs()

# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

MODEL_COLORS = {
    "logistic_regression":  "#4FC3F7",
    "random_forest":        "#81C784",
    "xgboost":              "#FF7043",
    "lightgbm":             "#CE93D8",
    "gradient_boosting":    "#FFB74D",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


def _model_label(name: str) -> str:
    return name.replace("_", " ").title()


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#888888")


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")

    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_metrics"]["test_auroc"]):
        y, p = r["y_test"], r["test_probs"]
        fpr, tpr, _ = roc_curve(y, p)
        auroc = roc_auc_score(y, p)
        ax.plot(fpr, tpr,
                label=f"{_model_label(name)}  (AUC = {auroc:.3f})",
                color=_color(name), linewidth=2.2)

    ax.fill_between([0, 1], [0, 1], alpha=0.06, color="gray")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Sepsis 6h Early Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = save_path or str(OUTPUTS_DIR / "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_pr_curves(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#F8F9FA")

    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_metrics"]["test_auprc"]):
        y, p = r["y_test"], r["test_probs"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = auc(rec, prec)
        ax.plot(rec, prec,
                label=f"{_model_label(name)}  (AP = {ap:.3f})",
                color=_color(name), linewidth=2.2)

    baseline = list(results.values())[0]["y_test"].mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.7, linewidth=1,
               label=f"Baseline prevalence = {baseline:.3f}")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Sepsis 6h Early Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = save_path or str(OUTPUTS_DIR / "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_calibration_curves(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#F8F9FA")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.5,
            label="Perfect calibration")

    for name, r in results.items():
        y, p = r["y_test"], r["test_probs"]
        try:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos,
                    "o-", markersize=5,
                    label=_model_label(name),
                    color=_color(name), linewidth=2)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Empirical)", fontsize=12)
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    path = save_path or str(OUTPUTS_DIR / "calibration_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_confusion_matrices(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        y, preds = r["y_test"], r["test_preds"]
        cm = confusion_matrix(y, preds, labels=[0, 1])

        cm_norm = cm.astype(float) / cm.sum()
        sns.heatmap(
            cm_norm, ax=ax, cmap="Blues",
            annot=False, cbar=False,
            linewidths=0.5, linecolor="white"
        )

        thresh = cm.max() / 2.0
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > thresh else "#333333"
                ax.text(j + 0.5, i + 0.35, f"{labels[i][j]}",
                        ha="center", va="center", color=color,
                        fontsize=11, fontweight="bold")
                ax.text(j + 0.5, i + 0.65, f"{cm[i, j]:,}",
                        ha="center", va="center", color=color, fontsize=13)

        ax.set_title(_model_label(name), fontsize=11, fontweight="bold", pad=10)
        ax.set_xticklabels(["Pred: 0\n(No Sepsis)", "Pred: 1\n(Sepsis)"], fontsize=8.5)
        ax.set_yticklabels(["True: 0\n(No Sepsis)", "True: 1\n(Sepsis)"],
                           fontsize=8.5, va="center")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = save_path or str(OUTPUTS_DIR / "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_feature_importance(
    results: Dict[str, Dict],
    feature_cols: List[str],
    top_n: int = 20,
    save_path: str = None
) -> Optional[str]:
    """Plot feature importance for tree-based models."""
    saved_path = None

    for name, r in results.items():
        model = r["model"]

        base = model
        if hasattr(model, "calibrated_classifiers_"):
            try:
                base = model.calibrated_classifiers_[0].estimator
            except (IndexError, AttributeError):
                pass

        if not hasattr(base, "feature_importances_"):
            continue

        importances = base.feature_importances_
        if len(importances) != len(feature_cols):
            continue

        feat_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.4)))
        ax.set_facecolor("#F8F9FA")

        cmap = plt.cm.get_cmap("RdYlGn")
        colors = [cmap(i / len(feat_df)) for i in range(len(feat_df))]

        bars = ax.barh(feat_df["feature"], feat_df["importance"],
                       color=colors, edgecolor="white", height=0.7)

        for bar, val in zip(bars, feat_df["importance"]):
            ax.text(bar.get_width() + feat_df["importance"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
        ax.set_title(f"Top {top_n} Features — {_model_label(name)}",
                     fontsize=13, fontweight="bold")
        ax.set_xlim([0, feat_df["importance"].max() * 1.15])

        plt.tight_layout()
        p = save_path or str(OUTPUTS_DIR / f"feature_importance_{name}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")
        saved_path = p

    return saved_path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_utility_vs_threshold(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    """Show how PhysioNet utility score varies with threshold."""
    from models import physionet_utility_score

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#F8F9FA")

    thresholds = np.linspace(0.05, 0.95, 181)

    for name, r in results.items():
        y, p = r["y_test"], r["test_probs"]
        scores = [physionet_utility_score(y, p, t) for t in thresholds]
        opt_t = r["threshold"]
        opt_s = physionet_utility_score(y, p, opt_t)

        ax.plot(thresholds, scores, color=_color(name),
                linewidth=2, label=_model_label(name))
        ax.axvline(opt_t, color=_color(name),
                   linestyle=":", alpha=0.6, linewidth=1.5)
        ax.scatter([opt_t], [opt_s], color=_color(name),
                   s=60, zorder=5)

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Clinical Utility Score", fontsize=12)
    ax.set_title("Utility Score vs Threshold\n(FN penalty = 2.0, FP penalty = 0.05)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    path = save_path or str(OUTPUTS_DIR / "utility_vs_threshold.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def generate_metrics_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Compile all test metrics into a formatted comparison table."""
    rows = []
    for name, r in results.items():
        m = r["test_metrics"]
        rows.append({
            "Model":        _model_label(name),
            "AUROC":        f"{m['test_auroc']:.4f}",
            "AUPRC":        f"{m['test_auprc']:.4f}",
            "Brier Score":  f"{m['test_brier']:.4f}",
            "Sensitivity":  f"{m['test_sensitivity']:.3f}",
            "Specificity":  f"{m['test_specificity']:.3f}",
            "PPV":          f"{m['test_ppv']:.3f}",
            "NPV":          f"{m['test_npv']:.3f}",
            "F1":           f"{m['test_f1']:.3f}",
            "Threshold":    f"{r['threshold']:.3f}",
            "TP":           m["test_tp"],
            "FP":           m["test_fp"],
            "TN":           m["test_tn"],
            "FN":           m["test_fn"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("AUROC", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def plot_summary_dashboard(
    results: Dict[str, Dict],
    feature_cols: List[str],
    save_path: str = None
) -> str:
    """4-panel summary dashboard: ROC, PR, Calibration, Feature Importance."""
    best_tree = None
    best_auroc = 0
    for name, r in results.items():
        if name != "logistic_regression":
            if r["test_metrics"]["test_auroc"] > best_auroc:
                best_auroc = r["test_metrics"]["test_auroc"]
                best_tree = name

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr  = fig.add_subplot(gs[0, 1])
    ax_cal = fig.add_subplot(gs[1, 0])
    ax_fi  = fig.add_subplot(gs[1, 1])

    # ── ROC ──────────────────────────────────────────────────────────────
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_metrics"]["test_auroc"]):
        y, p = r["y_test"], r["test_probs"]
        fpr, tpr, _ = roc_curve(y, p)
        auroc = roc_auc_score(y, p)
        ax_roc.plot(fpr, tpr, label=f"{_model_label(name)} ({auroc:.3f})",
                    color=_color(name), lw=2)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title("ROC Curves", fontweight="bold")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_facecolor("#F8F9FA")

    # ── PR ───────────────────────────────────────────────────────────────
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_metrics"]["test_auprc"]):
        y, p = r["y_test"], r["test_probs"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = auc(rec, prec)
        ax_pr.plot(rec, prec, label=f"{_model_label(name)} ({ap:.3f})",
                   color=_color(name), lw=2)
    baseline = list(results.values())[0]["y_test"].mean()
    ax_pr.axhline(baseline, color="gray", linestyle="--", alpha=0.5)
    ax_pr.set_title("Precision-Recall Curves", fontweight="bold")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.set_facecolor("#F8F9FA")

    # ── Calibration ───────────────────────────────────────────────────────
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for name, r in results.items():
        y, p = r["y_test"], r["test_probs"]
        try:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
            ax_cal.plot(mean_pred, frac_pos, "o-", markersize=4,
                        label=_model_label(name), color=_color(name), lw=2)
        except Exception:
            pass
    ax_cal.set_title("Calibration Curves", fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Prob")
    ax_cal.set_ylabel("Fraction Positives")
    ax_cal.legend(fontsize=8)
    ax_cal.set_facecolor("#F8F9FA")

    # ── Feature Importance ────────────────────────────────────────────────
    if best_tree:
        r = results[best_tree]
        model = r["model"]
        base = model
        if hasattr(model, "calibrated_classifiers_"):
            try:
                base = model.calibrated_classifiers_[0].estimator
            except Exception:
                pass
        if hasattr(base, "feature_importances_"):
            imp = base.feature_importances_
            if len(imp) == len(feature_cols):
                top_n = 15
                feat_df = pd.DataFrame({
                    "f": feature_cols, "i": imp
                }).sort_values("i", ascending=True).tail(top_n)
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
                ax_fi.barh(feat_df["f"], feat_df["i"], color=colors, height=0.7)
                ax_fi.set_title(f"Top {top_n} Features — {_model_label(best_tree)}",
                                fontweight="bold")
                ax_fi.set_xlabel("Importance (Gini)")
                ax_fi.set_facecolor("#F8F9FA")

    fig.suptitle("Sepsis 6h Early Prediction — Model Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    path = save_path or str(OUTPUTS_DIR / "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

def run_evaluation_pipeline(
    results: Dict[str, Dict],
    feature_cols: List[str]
) -> Tuple:
    """Generate all evaluation plots and metrics table."""
    print("\n" + "=" * 60)
    print("EVALUATION — GENERATING PLOTS & METRICS")
    print("=" * 60)

    plot_paths = {}

    print("\n  Plotting ROC curves...")
    plot_paths["roc"] = plot_roc_curves(results)

    print("  Plotting PR curves...")
    plot_paths["pr"] = plot_pr_curves(results)

    print("  Plotting calibration curves...")
    plot_paths["cal"] = plot_calibration_curves(results)

    print("  Plotting confusion matrices...")
    plot_paths["cm"] = plot_confusion_matrices(results)

    print("  Plotting feature importance...")
    fi_path = plot_feature_importance(results, feature_cols, top_n=20)
    if fi_path:
        plot_paths["fi"] = fi_path

    print("  Plotting utility score vs threshold...")
    plot_paths["utility"] = plot_utility_vs_threshold(results)

    print("  Generating summary dashboard...")
    plot_paths["dashboard"] = plot_summary_dashboard(results, feature_cols)

    metrics_table = generate_metrics_table(results)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS:")
    print(metrics_table.to_string(index=False))
    print("=" * 60)

    csv_path = str(OUTPUTS_DIR / "metrics_table.csv")
    metrics_table.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved: {csv_path}")

    return plot_paths, metrics_table


from typing import Tuple


if __name__ == "__main__":
    from preprocessing import run_preprocessing_pipeline
    from models import run_training_pipeline

    train_df, val_df, test_df, feature_cols = run_preprocessing_pipeline(
        use_synthetic=True, n_synthetic=500
    )
    results = run_training_pipeline(
        train_df, val_df, test_df, feature_cols
    )
    plot_paths, metrics_table = run_evaluation_pipeline(results, feature_cols)
