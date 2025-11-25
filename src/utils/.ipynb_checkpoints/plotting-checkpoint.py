import numpy as np
import matplotlib.pyplot as plt

# Save plot
def save_plot(fig, outpath):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")

# Downsample
def _downsample(series, max_points=5000):
    n = len(series)
    if n <= max_points:
        return series
    stride = n // max_points
    return series[::stride]


def plot_full_series(data, meta, outpath, title_prefix=""):

    ds = _downsample(data)

    # Map anomaly bounds
    factor = len(data) / len(ds)
    a_start_ds = int(meta["anomaly_start"] / factor)
    a_end_ds   = int(meta["anomaly_end"]   / factor)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot line
    ax.plot(ds, color="black", linewidth=0.4)

    # Highlight anomaly
    ax.axvspan(a_start_ds, a_end_ds, color="red", alpha=0.15)

    # Boundary lines
    ax.axvline(a_start_ds, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(a_end_ds,   color="red", linestyle="--", linewidth=0.8)

    # Labels
    ax.set_title(f"{title_prefix} — Full Series (Anomaly Highlighted)")
    ax.set_xlabel("Time Index (downsampled)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(alpha=0.3, linestyle="--")

    save_plot(fig, outpath)
    return fig 

# Zoomed anomaly region
def plot_zoom_anomaly(data, meta, outpath, title_prefix="", margin=300):

    start = max(0, meta["anomaly_start"] - margin)
    end   = min(len(data), meta["anomaly_end"] + margin)
    segment = data[start:end]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(segment, color="black", linewidth=0.8)

    local_start = meta["anomaly_start"] - start
    local_end   = meta["anomaly_end"]   - start

    ax.axvspan(local_start, local_end, color="red", alpha=0.15)
    ax.axvline(local_start, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(local_end,   color="red", linestyle="--", linewidth=0.8)

    ax.set_title(f"{title_prefix} — Zoomed Anomaly Region")
    ax.set_xlabel("Time Index (local)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(alpha=0.3, linestyle="--")

    save_plot(fig, outpath)
    return fig

# Zoomed anomaly region
def plot_zoom_normal(data, meta, outpath, title_prefix="", margin=300):

    end = max(0, meta["anomaly_start"] - 1)
    start = max(0, end - margin)

    if end - start < 10:
        return None

    segment = data[start:end]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(segment, color="black", linewidth=0.8)

    ax.set_title(f"{title_prefix} — Zoomed Normal Region (Before Anomaly)")
    ax.set_xlabel("Time Index (local)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(alpha=0.3, linestyle="--")

    save_plot(fig, outpath)
    return fig
