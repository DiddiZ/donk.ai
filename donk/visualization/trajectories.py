from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_line_set(
    x: np.ndarray, ys: np.ndarray, color=None, individual_lines: str | bool = "auto", label: str | None = None
):
    """Plot a set of lines with a mean and confidence interval."""
    N, _ = ys.shape

    # Only show individual lines if there are only a few in auto mode
    if individual_lines == "auto":
        individual_lines = N <= 5

    # Plot indiviadual data lines
    if individual_lines:
        for n in range(N):
            l = plt.plot(x, ys[n], c=color, linewidth=0.5)[0]
            color = l.get_color()  # Remember color

    # Plot distriution
    mean = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    conf = 1.96 * std / np.sqrt(N)
    l = plt.plot(mean, c=color, label=label)[0]
    color = l.get_color()  # Remember color
    plt.fill_between(
        x,
        mean - conf,
        mean + conf,
        facecolor=color,
        alpha=0.25,
        interpolate=True,
    )


def visualize_trajectories(file: Path | str, X: np.ndarray, U: np.ndarray):
    """Visualize trajectories with states and action in seperate plots.

    Args:
        file: File to save figure to.
        X: (T+1, dX), states
        U: (T, dU), actions
    """
    plt.figure(figsize=(16, 9))

    ax1 = plt.subplot(2, 1, 1)
    for d in range(X.shape[-1]):
        plot_line_set(np.arange(X.shape[1]), X[:, :, d], label=f"$x_{{{d}}}$")
    plt.ylabel("state")
    plt.legend()
    plt.grid(linestyle=":")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplot(2, 1, 2, sharex=ax1)
    for d in range(U.shape[-1]):
        plot_line_set(np.arange(U.shape[1]), U[:, :, d], label=f"$u_{{{d}}}$")
    plt.ylabel("action")
    plt.xlabel("$t$")
    plt.legend()
    plt.grid(linestyle=":")

    plt.tight_layout()
    plt.savefig(file)
    plt.close()
