"""
Sepsis Onset Prediction — Bidirectional LSTM with Missingness Masking
======================================================================
Architecture:
- Input: [batch, time_steps, features + missingness_mask]
- Time-delta embedding (irregular interval encoding)
- Bidirectional LSTM with dropout + LayerNorm
- Attention pooling over time dimension
- Binary classification head with sigmoid

Handles:
- Variable-length sequences (padded to max_len, masked)
- Irregular time intervals (time-delta feature)
- High missingness (mask channel as extra input)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available — LSTM model will be skipped.")
    DEVICE = None

import sklearn.base
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import MODELS_DIR, ensure_dirs

ensure_dirs()


# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────

if HAS_TORCH:
    class SepsisSequenceDataset(Dataset):
        """
        Dataset for patient-level time-series.

        Each sample is a full ICU stay (variable length).
        The prediction target is the LAST row's label
        (i.e., predict sepsis risk at the current timestep
        given all past observations).

        Features are padded to max_len; a padding mask is returned.
        """

        def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            max_len: int = 72
        ):
            self.max_len = max_len
            self.feature_cols = feature_cols
            self.samples = []

            for pid, group in df.groupby("patient_id"):
                group = group.sort_values("hour")
                X = group[feature_cols].values.astype(np.float32)
                y = group["target"].values.astype(np.float32)

                T = min(len(X), max_len)
                X_trunc = X[-T:]
                y_trunc = y[-T:]

                pad_len = max_len - T
                X_pad = np.vstack([
                    np.zeros((pad_len, X.shape[1]), dtype=np.float32),
                    X_trunc
                ])
                y_pad = np.concatenate([
                    np.zeros(pad_len, dtype=np.float32),
                    y_trunc
                ])
                mask = np.array(
                    [0] * pad_len + [1] * T, dtype=np.float32
                )  # 1 = real, 0 = padding

                self.samples.append((X_pad, y_pad, mask))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            X, y, mask = self.samples[idx]
            return (
                torch.tensor(X),
                torch.tensor(y),
                torch.tensor(mask)
            )


    # ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────

    class TemporalAttention(nn.Module):
        """Soft attention over time dimension."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.attn = nn.Linear(hidden_size, 1)

        def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor):
            """
            Args:
                lstm_out: [batch, seq_len, hidden]
                mask:     [batch, seq_len] — 1=real, 0=padding
            Returns:
                context:  [batch, hidden]
            """
            scores = self.attn(lstm_out).squeeze(-1)         # [batch, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)     # mask padding
            weights = torch.softmax(scores, dim=1)           # [batch, seq_len]
            context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # [batch, hidden]
            return context, weights


    # ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────

    class SepsisLSTM(nn.Module):
        """
        Bidirectional LSTM for sepsis onset prediction.

        Input features are augmented with:
        - Missingness mask channel (informative missingness)
        - Time-delta channel (hours since previous observation)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.directions = 2 if bidirectional else 1

            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )

            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )

            lstm_out_size = hidden_size * self.directions

            self.layer_norm = nn.LayerNorm(lstm_out_size)

            self.attention = TemporalAttention(lstm_out_size)

            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )

            self.timestep_classifier = nn.Sequential(
                nn.Linear(lstm_out_size, 1)
            )

            self._init_weights()

        def _init_weights(self):
            for name, param in self.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            return_sequence: bool = True
        ):
            """
            Args:
                x:     [batch, seq_len, input_size]
                mask:  [batch, seq_len] — 1=real, 0=padding
                return_sequence: if True, return per-timestep predictions

            Returns:
                logits: [batch, seq_len] if return_sequence else [batch]
            """
            h = self.input_proj(x)   # [batch, seq_len, hidden]

            lstm_out, _ = self.lstm(h)   # [batch, seq_len, hidden*dirs]
            lstm_out = self.layer_norm(lstm_out)

            if return_sequence:
                logits = self.timestep_classifier(lstm_out).squeeze(-1)
                return logits  # [batch, seq_len]
            else:
                context, _ = self.attention(lstm_out, mask)
                logits = self.classifier(context).squeeze(-1)
                return logits  # [batch]


    # ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────

    def train_lstm(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        max_len: int = 72,
        hidden_size: int = 64,
        patience: int = 5
    ) -> Tuple[nn.Module, float, Dict]:
        """
        Train the Bidirectional LSTM model.

        Returns:
            (trained_model, best_val_auroc, history)
        """
        print("\n  Building SepsisSequenceDataset...")
        train_ds = SepsisSequenceDataset(train_df, feature_cols, max_len=max_len)
        val_ds   = SepsisSequenceDataset(val_df,   feature_cols, max_len=max_len)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )

        input_size = len(feature_cols)
        model = SepsisLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  SepsisLSTM: {n_params:,} trainable parameters")

        pos_weight_val = (train_df["target"] == 0).sum() / max((train_df["target"] == 1).sum(), 1)
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

        best_auroc = 0.0
        best_state = None
        history = {"train_loss": [], "val_auroc": [], "val_auprc": []}
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch, mask_batch in train_loader:
                X_batch  = X_batch.to(DEVICE)
                y_batch  = y_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)

                optimizer.zero_grad()
                logits = model(X_batch, mask_batch, return_sequence=True)

                loss = criterion(logits, y_batch)
                masked_loss = (loss * mask_batch).sum() / (mask_batch.sum() + 1e-8)

                masked_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += masked_loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            all_probs, all_labels = [], []

            with torch.no_grad():
                for X_batch, y_batch, mask_batch in val_loader:
                    X_batch    = X_batch.to(DEVICE)
                    mask_batch = mask_batch.to(DEVICE)
                    logits = model(X_batch, mask_batch, return_sequence=True)
                    probs  = torch.sigmoid(logits).cpu().numpy()
                    labels = y_batch.numpy()
                    masks  = mask_batch.cpu().numpy()

                    for i in range(len(probs)):
                        m = masks[i].astype(bool)
                        all_probs.append(probs[i][m])
                        all_labels.append(labels[i][m])

            all_probs  = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels).astype(int)

            if all_labels.sum() == 0:
                val_auroc = val_auprc = 0.5
            else:
                val_auroc = roc_auc_score(all_labels, all_probs)
                val_auprc = average_precision_score(all_labels, all_probs)

            history["train_loss"].append(avg_loss)
            history["val_auroc"].append(val_auroc)
            history["val_auprc"].append(val_auprc)

            scheduler.step(val_auroc)

            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Loss={avg_loss:.4f} | "
                  f"Val AUROC={val_auroc:.4f} | "
                  f"Val AUPRC={val_auprc:.4f}")

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        print(f"\n  Best Val AUROC: {best_auroc:.4f}")

        save_path = str(MODELS_DIR / "lstm_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "feature_cols": feature_cols,
            "max_len": max_len,
            "best_val_auroc": best_auroc
        }, save_path)
        print(f"  Saved: {save_path}")

        return model, best_auroc, history


    def predict_lstm(
        model: nn.Module,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_len: int = 72,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from trained LSTM on a DataFrame.

        Returns:
            probs: predicted probabilities (row-level)
            labels: ground truth (row-level)
        """
        ds = SepsisSequenceDataset(df, feature_cols, max_len=max_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in loader:
                X_batch    = X_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)
                logits = model(X_batch, mask_batch, return_sequence=True)
                probs  = torch.sigmoid(logits).cpu().numpy()
                labels = y_batch.numpy()
                masks  = mask_batch.cpu().numpy()

                for i in range(len(probs)):
                    m = masks[i].astype(bool)
                    all_probs.append(probs[i][m])
                    all_labels.append(labels[i][m])

        return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


else:
    class SepsisLSTM:
        pass

    def train_lstm(*args, **kwargs):
        print("PyTorch not available — LSTM skipped.")
        return None, 0.0, {}

    def predict_lstm(*args, **kwargs):
        return np.array([]), np.array([])


if __name__ == "__main__":
    if not HAS_TORCH:
        print("Install PyTorch to test LSTM: pip install torch")
        sys.exit(0)

    from preprocessing import run_preprocessing_pipeline
    train_df, val_df, test_df, feature_cols = run_preprocessing_pipeline(
        use_synthetic=True, n_synthetic=300
    )

    model, best_auroc, history = train_lstm(
        train_df, val_df, feature_cols,
        epochs=5, batch_size=16, max_len=48
    )
    if model is not None:
        probs, labels = predict_lstm(model, test_df, feature_cols, max_len=48)
        if labels.sum() > 0:
            auroc = roc_auc_score(labels, probs)
            print(f"Test AUROC: {auroc:.4f}")
