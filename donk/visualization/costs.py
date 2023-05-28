from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_costs(
    output_file: Path,
    costs: list[np.ndarray],
    cost_labels: list[str],
    include_total: bool = True,
) -> None:
    """Plots mutiple cost curves.

    Args:
        output_file: File to write plot to. Use `None` to show plot.
        costs: [(..., T+1)] Costs
        cost_labels: Names for the costs
        include_total: Sum up all costs and show a total
    """
    T = costs[0].shape[-1] - 1

    if include_total:
        costs.append(np.sum(costs, axis=0))
        cost_labels.append("total")

    for name, cost in zip(cost_labels, costs):
        cost = cost.reshape(-1, T + 1)
        l = plt.plot(np.mean(cost, axis=0), label=name)[0]

        plt.fill_between(
            np.arange(T + 1),
            np.min(cost, axis=0),
            np.max(cost, axis=0),
            facecolor=l.get_color(),
            alpha=0.25,
            interpolate=True,
        )

    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("costs")

    # Export
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
