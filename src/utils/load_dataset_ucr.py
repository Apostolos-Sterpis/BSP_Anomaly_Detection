import os
import numpy as np
import pandas as pd

def parse_ucr_filename(filepath):
    """
    Parses metadata from UCR filename format:
    ID_UCR_Anomaly_Name_TrainEnd_AnomStart_AnomEnd.txt
    """
    filename = os.path.basename(filepath)
    # Remove extension and split by underscore
    name_clean = os.path.splitext(filename)[0]
    parts = name_clean.split('_')

    return {
        "source_file": filename,
        "train_end": int(parts[-3]),
        "anomaly_start": int(parts[-2]),
        "anomaly_end": int(parts[-1])
    }

def load_raw_data(filepath):
    """
    Loads the single-column text file.
    """
    try:
        return np.loadtxt(filepath)
    except ValueError:
        return pd.read_csv(filepath, header=None).iloc[:, 0].values
