from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Style
# =============================================================================
SIGNAL_COLOR = "0.15"       # dark gray
SCORE_COLOR = "0.15"        # dark gray
TRUE_SHADE_COLOR = "tab:orange"
PRED_SHADE_COLOR = "tab:red"
THRESH_COLOR = "tab:purple"

SHADE_ALPHA_TRUE = 0.22
SHADE_ALPHA_PRED = 0.18


def _to_1d(x):
    return np.asarray(x).reshape(-1)


def _align_min(*arrays):
    valid = [a for a in arrays if a is not None]
    if len(valid) == 0:
        return arrays

    m = min(len(_to_1d(a)) for a in valid)
    out = []
    for a in arrays:
        out.append(None if a is None else _to_1d(a)[:m])
    return tuple(out)


def _segments_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous (start, end) segments where labels == 1 (end exclusive)."""
    y = _to_1d(labels).astype(int)
    segs: List[Tuple[int, int]] = []
    in_seg = False
    start = 0

    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            in_seg = False
            segs.append((start, i))

    if in_seg:
        segs.append((start, len(y)))

    return segs


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_signal(
    signal: Sequence[float],
    true_labels: Optional[Sequence[int]] = None,
    pred_labels: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    x_offset: int = 0,
    max_points: Optional[int] = None,
):
    """
    Plot the signal with optional true/pred anomaly shading.

    - x_offset: shifts the x-axis (zoom plots).
    - max_points: downsample to at most this many points (overview plots).
    """
    signal, true_labels, pred_labels = _align_min(signal, true_labels, pred_labels)
    x = np.arange(len(signal)) + int(x_offset)

    # Downsample (overview only)
    if max_points is not None and len(signal) > max_points:
        step = int(np.ceil(len(signal) / max_points))
        signal = signal[::step]
        x = x[::step]
        if true_labels is not None:
            true_labels = true_labels[::step]
        if pred_labels is not None:
            pred_labels = pred_labels[::step]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, signal, linewidth=1.2, color=SIGNAL_COLOR)

    # True anomaly shading (orange)
    if true_labels is not None:
        segs = _segments_from_labels(true_labels)
        for i, (s, e) in enumerate(segs):
            ax.axvspan(
                x[s], x[e - 1] + 1,
                color=TRUE_SHADE_COLOR,
                alpha=SHADE_ALPHA_TRUE,
                label="True anomaly" if i == 0 else None,
            )

    # Pred anomaly shading (red)
    if pred_labels is not None:
        segs = _segments_from_labels(pred_labels)
        for i, (s, e) in enumerate(segs):
            ax.axvspan(
                x[s], x[e - 1] + 1,
                color=PRED_SHADE_COLOR,
                alpha=SHADE_ALPHA_PRED,
                label="Pred anomaly" if i == 0 else None,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    _style_axes(ax)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_scores(
    scores: Sequence[float],
    threshold: Optional[float] = None,
    true_labels: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    x_offset: int = 0,
    max_points: Optional[int] = None,
):
    """
    Plot scores with optional threshold and true anomaly shading.

    - x_offset: shifts the x-axis (zoom plots).
    - max_points: downsample to at most this many points (overview plots).
    """
    scores, true_labels = _align_min(scores, true_labels)
    x = np.arange(len(scores)) + int(x_offset)

    # Downsample (overview only)
    if max_points is not None and len(scores) > max_points:
        step = int(np.ceil(len(scores) / max_points))
        scores = scores[::step]
        x = x[::step]
        if true_labels is not None:
            true_labels = true_labels[::step]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, scores, linewidth=1.2, color=SCORE_COLOR)

    if threshold is not None:
        ax.axhline(
            float(threshold),
            linestyle="--",
            linewidth=1.2,
            color=THRESH_COLOR,
            label="Threshold",
        )

    # Keep score plots readable: shade only TRUE anomaly (orange)
    if true_labels is not None:
        segs = _segments_from_labels(true_labels)
        for (s, e) in segs:
            ax.axvspan(
                x[s], x[e - 1] + 1,
                color=TRUE_SHADE_COLOR,
                alpha=SHADE_ALPHA_TRUE,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Score")

    _style_axes(ax)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
