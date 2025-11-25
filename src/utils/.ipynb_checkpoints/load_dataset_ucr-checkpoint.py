import numpy as np
import os

def load_dataset(path):
    data = np.loadtxt(path)

    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')

    dataset_name = parts[3]
    train_end = int(parts[4])
    anomaly_start = int(parts[5])
    anomaly_end = int(parts[6])

    metadata = {
        "name": dataset_name,
        "train_end": train_end,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
        "length": len(data),
    }

    return data, metadata
