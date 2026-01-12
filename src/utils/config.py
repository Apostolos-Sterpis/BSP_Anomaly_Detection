from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# =============================================================================
# Project paths
# =============================================================================
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
ALL_RAW_DIR: Path = DATA_DIR / "all_raw"

RESULTS_DIR: Path = ROOT_DIR / "results"
DOCS_DIR: Path = ROOT_DIR / "docs"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
SRC_DIR: Path = ROOT_DIR / "src"

# =============================================================================
# Datasets
# =============================================================================
DATASETS: List[str] = [
    "respiration1",
    "PowerDemand1",
    "gaitHunt2",
    "InternalBleeding4",
    "InternalBleeding16",
]

PINNED_RAW_FILES: Dict[str, str] = {
    "respiration1":       "186_UCR_Anomaly_respiration1_100000_110260_110412.txt",
    "PowerDemand1":       "152_UCR_Anomaly_PowerDemand1_9000_18485_18821.txt",
    "gaitHunt2":          "171_UCR_Anomaly_gaitHunt2_18500_31200_31850.txt",
    "InternalBleeding4":  "032_UCR_Anomaly_DISTORTEDInternalBleeding4_1000_4675_5033.txt",
    "InternalBleeding16": "098_UCR_Anomaly_NOISEInternalBleeding16_1200_4187_4199.txt",
}

# =============================================================================
# Preprocessing + windowing
# =============================================================================
RANDOM_SEED: int = 42

WINDOW_SIZE: int = 128
STRIDE: int = 4

WINDOW_LABEL_MODE: str = "any"

EXPORT_VARIANTS: List[str] = ["raw", "z", "robust"]

META_KEYS = {
    "train_end": "train_end",
    "anomaly_start": "anomaly_start",
    "anomaly_end": "anomaly_end",
}

# =============================================================================
# Method-specific defaults
# =============================================================================

# Statistical baselines
Z_SCORE_THRESHOLD: float = 3.0
MOVING_AVG_WINDOW: int = 128
MOVING_AVG_THRESHOLD_STD: float = 3.0
EWMA_SPAN: int = 64
EWMA_THRESHOLD_STD: float = 3.0

BASELINE_THR_QUANTILE: float = 0.995
PLOT_ZOOM_MARGIN: int = 2000

# Shallow learning
IFOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.01,
    "random_state": RANDOM_SEED,
}
OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": 0.01,
    "gamma": "scale",
}

OCSVM_MAX_TRAIN_WINDOWS: int = 20_000

# Deep learning (LSTM Autoencoder)
AE_PARAMS = {
    "window_size": WINDOW_SIZE,
    "epochs": 15,
    "batch_size": 64,
    "validation_split": 0.10,
    "early_stopping_patience": 5,
    "lstm_units": 64,
    "clipnorm": 1.0,
    "threshold_quantile": 0.99,
}

# Matrix Profile (STUMPY)
MP_PARAMS = {
    "m": WINDOW_SIZE,
    "top_k": 1,
}

# =============================================================================
# Plot settings
# =============================================================================
PLOT_DOWNSAMPLE: int = 10
FIG_DPI: int = 150

# =============================================================================
# Path builders
# =============================================================================
def dataset_dir(dataset: str) -> Path:
    return DATA_DIR / dataset

def cleaned_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "cleaned"

def method_ready_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "method_ready"

def results_method_dir(method_name: str) -> Path:
    return RESULTS_DIR / method_name

def results_dataset_dir(method_name: str, dataset: str) -> Path:
    return results_method_dir(method_name) / dataset

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# =============================================================================
# Raw file resolution
# =============================================================================
def iter_raw_files() -> List[Path]:
    """Pinned raw files in the same order as DATASETS."""
    paths: List[Path] = []
    for ds in DATASETS:
        if ds not in PINNED_RAW_FILES:
            raise KeyError
        p = ALL_RAW_DIR / PINNED_RAW_FILES[ds]
        if not p.exists():
            raise FileNotFoundError
        paths.append(p)
    return paths
