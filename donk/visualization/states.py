"""Visualization tool for state spaces."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_correlation(output_file: Path | str | None, X: np.ndarray) -> None:
    """Visualize the correlation between states.

    Args:
        output_file: File to write the plot to.
        X: shape (N, dX), states
    """
    corr = np.corrcoef(X, rowvar=False)
    sns.heatmap(
        corr,
        cmap="bwr",
        vmin=-1,
        vmax=1,
        mask=np.triu(np.ones_like(corr, dtype=bool), 1),  # Mask upper triangle
    )
    plt.xlabel("$x$")
    plt.ylabel("$x$")
    plt.title("Correlation")

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
