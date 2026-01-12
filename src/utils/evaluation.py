import numpy as np


def _to_1d_int(x):
    """Convert to 1D int array (0/1)."""
    arr = np.asarray(x).reshape(-1)
    return arr.astype(int)


def align_lengths(*arrays):
    """
    Trim all provided arrays to the same length (min length).
    Pass None to skip an array.
    Returns aligned arrays in the same order.
    """
    valid = [a for a in arrays if a is not None]
    if len(valid) == 0:
        return arrays

    min_len = min(len(np.asarray(a).reshape(-1)) for a in valid)

    out = []
    for a in arrays:
        if a is None:
            out.append(None)
        else:
            out.append(np.asarray(a).reshape(-1)[:min_len])
    return tuple(out)


def confusion_counts(y_true, y_pred):
    """
    Return (tn, fp, fn, tp) for binary labels (0/1).
    """
    y_true = _to_1d_int(y_true)
    y_pred = _to_1d_int(y_pred)
    y_true, y_pred = align_lengths(y_true, y_pred)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def compute_binary_metrics(y_true, y_pred):
    """
    Compute standard binary metrics.
    Returns a dict with: precision, recall, f1, accuracy, tn, fp, fn, tp, eval_len.
    """
    y_true = _to_1d_int(y_true)
    y_pred = _to_1d_int(y_pred)
    y_true, y_pred = align_lengths(y_true, y_pred)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    denom_p = (tp + fp)
    denom_r = (tp + fn)

    precision = tp / denom_p if denom_p > 0 else 0.0
    recall = tp / denom_r if denom_r > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "eval_len": int(len(y_true)),
    }
   