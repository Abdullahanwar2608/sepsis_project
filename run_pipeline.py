"""
Sepsis Onset Prediction — End-to-End Pipeline Runner
=====================================================

Usage:
    python run_pipeline.py --synthetic --n-patients 500

    # PhysioNet 2019 data (auto-download if not present)
    python run_pipeline.py --physionet

    python run_pipeline.py --data-dir "C:/path/to/psv/files"

    python run_pipeline.py --synthetic --no-dl

    python run_pipeline.py --physionet --max-patients 5000
"""

import argparse
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import setup_logger, ensure_dirs, OUTPUTS_DIR, MODELS_DIR, Timer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sepsis 6h Early Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no download required)"
    )
    data_group.add_argument(
        "--physionet", action="store_true",
        help="Auto-download PhysioNet 2019 Challenge data (open access)"
    )
    data_group.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to directory containing PhysioNet .psv files"
    )
    data_group.add_argument(
        "--mimic-dir", type=str, default=None,
        help="Path to MIMIC-III CSV directory (requires credentialed access). "
             "Will run full extraction pipeline (CHARTEVENTS + LABEVENTS + Sepsis-3 labels)."
    )
    data_group.add_argument(
        "--mimic-csv", type=str, default=None,
        help="Path to pre-extracted MIMIC features CSV (mimic_hourly_features.csv). "
             "Use this if you've already run extract_mimic.py."
    )
    parser.add_argument(
        "--n-patients", type=int, default=2000,
        help="Number of synthetic patients (default: 2000)"
    )
    parser.add_argument(
        "--max-patients", type=int, default=None,
        help="Maximum patients to load from real data"
    )
    parser.add_argument(
        "--no-dl", action="store_true",
        help="Skip deep learning (LSTM) model"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="LSTM training epochs (default: 15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    logger = setup_logger(
        "sepsis",
        log_file=str(OUTPUTS_DIR / "pipeline.log")
    )

    logger.info("=" * 60)
    logger.info("SEPSIS ONSET PREDICTION PIPELINE")
    logger.info("=" * 60)

    # ── Determine data source ─────────────────────────────────────────────
    use_synthetic = args.synthetic or (
        not args.physionet
        and args.data_dir is None
        and getattr(args, "mimic_dir", None) is None
        and getattr(args, "mimic_csv", None) is None
    )
    data_dir = args.data_dir
    mimic_dir = getattr(args, "mimic_dir", None)
    mimic_csv = getattr(args, "mimic_csv", None)

    if args.physionet and not data_dir:
        logger.info("Downloading PhysioNet 2019 Challenge data...")
        from utils import download_physionet_2019, DATA_DIR
        data_dir = download_physionet_2019(str(DATA_DIR / "physionet2019"))
        if data_dir is None:
            logger.warning("Download failed — switching to synthetic data")
            use_synthetic = True

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    logger.info("\nSTEP 1: Preprocessing")
    with Timer() as t:
        from preprocessing import (
            run_preprocessing_pipeline,
            load_mimic_csv, load_mimic_from_raw
        )

        if mimic_dir:
            logger.info(f"MIMIC-III raw extraction from: {mimic_dir}")
            from utils import DATA_DIR
            raw_df = load_mimic_from_raw(
                mimic_dir=mimic_dir,
                output_dir=str(DATA_DIR / "mimic"),
                max_stays=args.max_patients
            )
            from preprocessing import (
                create_prediction_labels, handle_missing_values,
                engineer_temporal_features, detect_label_noise,
                patient_stratified_split, get_feature_columns
            )
            print("\n[2/5] Engineering labels...")
            raw_df = create_prediction_labels(raw_df)
            print("\n[3/5] Handling missing values...")
            raw_df = handle_missing_values(raw_df)
            print("\n[4/5] Engineering temporal features...")
            raw_df = engineer_temporal_features(raw_df)
            print("\n[5/5] Label noise mitigation...")
            raw_df = detect_label_noise(raw_df)
            print("\n[Split] Patient-level split...")
            train_df, val_df, test_df = patient_stratified_split(raw_df)
            feature_cols = get_feature_columns(train_df)

        elif mimic_csv:
            logger.info(f"Loading pre-extracted MIMIC CSV: {mimic_csv}")
            raw_df = load_mimic_csv(mimic_csv, max_patients=args.max_patients)
            from preprocessing import (
                create_prediction_labels, handle_missing_values,
                engineer_temporal_features, detect_label_noise,
                patient_stratified_split, get_feature_columns
            )
            raw_df = create_prediction_labels(raw_df)
            raw_df = handle_missing_values(raw_df)
            raw_df = engineer_temporal_features(raw_df)
            raw_df = detect_label_noise(raw_df)
            train_df, val_df, test_df = patient_stratified_split(raw_df)
            feature_cols = get_feature_columns(train_df)

        else:
            # PhysioNet PSV or synthetic
            train_df, val_df, test_df, feature_cols = run_preprocessing_pipeline(
                data_dir=data_dir,
                use_synthetic=use_synthetic,
                n_synthetic=args.n_patients,
                max_patients=args.max_patients
            )

    logger.info(f"Preprocessing complete in {t}")

    # ── Step 2: ML Model Training ─────────────────────────────────────────
    logger.info("\nSTEP 2: Model Training (ML)")
    with Timer() as t:
        from models import run_training_pipeline
        results = run_training_pipeline(
            train_df, val_df, test_df, feature_cols,
            use_confident_learning=True
        )
    logger.info(f"ML training complete in {t}")

    # ── Step 3: Deep Learning (optional) ─────────────────────────────────
    if not args.no_dl:
        logger.info("\nSTEP 3: Deep Learning (Bidirectional LSTM)")
        try:
            from deep_learning.lstm_model import (
                train_lstm, predict_lstm, HAS_TORCH
            )
            if HAS_TORCH:
                with Timer() as t:
                    lstm_model, best_auroc, history = train_lstm(
                        train_df, val_df, feature_cols,
                        epochs=args.epochs,
                        batch_size=32,
                        max_len=72
                    )
                logger.info(f"LSTM training complete in {t}")

                if lstm_model is not None:
                    from sklearn.metrics import (
                        roc_auc_score, average_precision_score,
                        brier_score_loss
                    )
                    from models import evaluate_model, select_optimal_threshold
                    import torch

                    test_probs, test_labels = predict_lstm(
                        lstm_model, test_df, feature_cols, max_len=72
                    )

                    if len(test_labels) > 0 and test_labels.sum() > 0:
                        threshold = select_optimal_threshold(
                            test_labels, test_probs, strategy="utility"
                        )
                        test_preds = (test_probs >= threshold).astype(int)
                        test_metrics = evaluate_model(
                            test_labels, test_probs, test_preds, split="test"
                        )

                        results["lstm"] = {
                            "model": lstm_model,
                            "threshold": threshold,
                            "val_metrics": {"val_auroc": best_auroc},
                            "test_metrics": test_metrics,
                            "test_probs": test_probs,
                            "test_preds": test_preds,
                            "y_test": test_labels,
                            "scaler": None,
                            "use_scaled": False,
                            "feature_cols": feature_cols
                        }
                        from evaluation import MODEL_COLORS
                        MODEL_COLORS["lstm"] = "#AB47BC"
            else:
                logger.warning("PyTorch not installed — LSTM skipped")
                logger.warning("Install with: pip install torch")
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            logger.warning("Continuing with ML models only")
    else:
        logger.info("\nSTEP 3: Deep Learning skipped (--no-dl)")

    # ── Step 4: Evaluation ────────────────────────────────────────────────
    logger.info("\nSTEP 4: Evaluation & Visualization")
    with Timer() as t:
        from evaluation import run_evaluation_pipeline
        plot_paths, metrics_table = run_evaluation_pipeline(results, feature_cols)
    logger.info(f"Evaluation complete in {t}")

    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {OUTPUTS_DIR}")
    logger.info(f"Models saved to:  {MODELS_DIR}")
    logger.info("\nGenerated files:")
    for key, path in plot_paths.items():
        logger.info(f"  [{key:12s}] {path}")
    logger.info(f"  [metrics_csv ] {OUTPUTS_DIR / 'metrics_table.csv'}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(metrics_table.to_string(index=False))
    print("=" * 60)

    return results, metrics_table


if __name__ == "__main__":
    results, metrics_table = main()
