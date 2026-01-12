import numpy as np
import pandas as pd

def clean_series(data):
    """
    Performs basic sanitary checks and fixes on a time-series.
    Returns: (cleaned_data, report_dict)
    """
    # Ensure it's a float array
    data = np.array(data, dtype=np.float64).flatten()
    
    report = {
        "initial_len": len(data),
        "nan_count": 0,
        "inf_count": 0,
        "was_modified": False
    }
    
    # 1. Detect and Replace Infinity with NaN
    if not np.isfinite(data).all():
        inf_mask = ~np.isfinite(data)
        report["inf_count"] = int(np.sum(inf_mask))
        data[inf_mask] = np.nan
        report["was_modified"] = True

    # 2. Handle NaNs via Linear Interpolation
    if np.isnan(data).any():
        report["nan_count"] = int(np.sum(np.isnan(data)))
        
        series = pd.Series(data)
        # Linear interpolation handles the gaps
        data = series.interpolate(method='linear', limit_direction='both').values
        report["was_modified"] = True
    
    return data, report
