"""
Shared utilities for the Sepsis Prediction Pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────────
# PYTHON PATH FOR LOCAL PYTHON INSTALL
# ─────────────────────────────────────────────────

PYTHON_EXE = r"C:\Users\abdul\AppData\Local\Python\bin\python.exe"

# ─────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────

def setup_logger(name: str = "sepsis", log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUTS_DIR, MODELS_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────
# TIMER CONTEXT MANAGER
# ─────────────────────────────────────────────────

class Timer:
    """Simple context manager timer."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─────────────────────────────────────────────────
# PHYSIONET 2019 DOWNLOAD HELPERS
# ─────────────────────────────────────────────────

PHYSIONET_TRAINING_A = (
    "https://archive.physionet.org/users/shared/challenge-2019/"
    "training_setA.zip"
)
PHYSIONET_TRAINING_B = (
    "https://archive.physionet.org/users/shared/challenge-2019/"
    "training_setB.zip"
)

def download_physionet_2019(dest_dir: str = None) -> str:
    """
    Download PhysioNet 2019 Challenge training data (Set A — open access).
    Set A has 20,336 patients (no login required).

    Returns path to extracted data directory.
    """
    import requests
    import zipfile
    from tqdm import tqdm

    dest = Path(dest_dir or DATA_DIR / "physionet2019")
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    psv_files = list(dest.glob("**/*.psv"))
    if len(psv_files) > 100:
        print(f"  Data already present: {len(psv_files)} PSV files in {dest}")
        return str(dest)

    zip_path = dest / "training_setA.zip"

    if not zip_path.exists():
        print(f"  Downloading PhysioNet 2019 Training Set A (~67 MB)...")
        print(f"  URL: {PHYSIONET_TRAINING_A}")
        try:
            response = requests.get(PHYSIONET_TRAINING_A, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True, desc="Downloading"
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            print(f"  Downloaded: {zip_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Falling back to synthetic data.")
            return None

    # Extract
    print(f"  Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print(f"  Extracted to {dest}")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return None

    psv_files = list(dest.glob("**/*.psv"))
    print(f"  Found {len(psv_files)} PSV files after extraction.")
    return str(dest)


# ─────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
