import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import config


def load_cleaned(dataset_name):
    """
    Load cleaned signal + point labels + metadata from:
      data/<dataset>/cleaned/cleaned.csv
      data/<dataset>/cleaned/metadata.json
    """
    clean_dir = config.cleaned_dir(dataset_name)

    df = pd.read_csv(clean_dir / "cleaned.csv")
    with open(clean_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    series = df["value"].to_numpy(dtype=float)
    labels = df["is_anomaly"].to_numpy(dtype=int)
    return series, labels, meta


def load_method_ready(dataset_name):
    """
    Load standard method-ready arrays from:
      data/<dataset>/method_ready/
    Returns a dict.
    """
    mr_dir = config.method_ready_dir(dataset_name)

    data = {
        "train_raw": np.load(mr_dir / "train_raw.npy"),
        "test_raw": np.load(mr_dir / "test_raw.npy"),
        "train_z": np.load(mr_dir / "train_z.npy"),
        "test_z": np.load(mr_dir / "test_z.npy"),
        "train_robust": np.load(mr_dir / "train_robust.npy"),
        "test_robust": np.load(mr_dir / "test_robust.npy"),
        "train_win_starts": np.load(mr_dir / "train_win_starts.npy"),
        "test_win_starts": np.load(mr_dir / "test_win_starts.npy"),
        "train_win_labels": np.load(mr_dir / "train_win_labels.npy"),
        "test_win_labels": np.load(mr_dir / "test_win_labels.npy"),
    }

    meta_path = mr_dir / "method_ready_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            data["meta"] = json.load(f)
    else:
        data["meta"] = {}

    return data


def results_dir(method_name, dataset_name=None):
    """
    Return results directory:
      results/<method>/             if dataset_name is None
      results/<method>/<dataset>/   otherwise
    """
    base = config.RESULTS_DIR / method_name
    path = base if dataset_name is None else base / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, obj, indent=2):
    """Save a dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def append_csv_row(csv_path, row_dict):
    """
    Append one row (dict) to a CSV.
    Creates the file with header if it doesn't exist.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row_dict])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
