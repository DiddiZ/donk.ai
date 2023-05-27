from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def visualize_iLQR(
    output_file: Path,
    results,
    kl_step: float,
    opt_result=None,
    cost_color="C0",
    constraint_color="C1",
    export_data: bool = True,
):
    """Visualize the results of an iLQR optimization."""
    results = pd.DataFrame(results).sort_values("eta")

    fig, ax1 = plt.subplots()
    plt.xlabel(r"$\eta$")
    plt.xscale("log")
    plt.ylabel("costs")

    # Cost plots
    plt.plot(results["eta"], results["expected_costs"], color=cost_color, label="costs")
    if opt_result is not None:
        plt.scatter([opt_result.eta], [opt_result.expected_costs], color=cost_color, label="minimal constrained costs")

    ax2 = ax1.twinx()  # Second y-axis
    plt.ylabel("constraint violation")
    plt.yscale("log")

    # Constraint plots
    ax2.plot(results["eta"], results["kl_div"], label="constraint violation", color=constraint_color)
    ax2.axhline(kl_step, color=constraint_color, linewidth=1, linestyle="dashed", label="constraint threshold")
    ax2.fill_between(
        results["eta"],
        0,
        1,
        where=results["kl_div"] <= kl_step,
        facecolor=constraint_color,
        alpha=0.1,
        transform=mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes),
        label="constraint fulfilled",
    )

    # Figure legend to include both axes
    fig.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    if export_data:  # Write data to csv
        results.to_csv(
            output_file.parent / (output_file.stem + ".csv"),
            columns=["eta", "kl_div", "expected_costs"],
            index=False,
        )


def visualize_step_adjust(
    output_file: Path,
    kl_step: np.ndarray,
    predicted_new_costs: np.ndarray,
    actual_new_costs: np.ndarray,
):
    """Visualize expected vs. actual costs in step adjust and resulting KL-step.

    Args:
        output_file: File to write the plot to.
        kl_step: (iterations, )
        predicted_new_costs:  (iterations-1, )
        actual_new_costs:  (iterations-1, )
    """
    iterations = kl_step.shape[0]
    base_kl_step = kl_step[0]

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(kl_step, label="KL-step")
    plt.axhline(base_kl_step, linewidth=1, color="black", linestyle="dashed", label="Base KL-step")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("KL divergence")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()

    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(np.arange(1, iterations), predicted_new_costs, label="predicted_new_costs")
    plt.plot(np.arange(1, iterations), actual_new_costs, label="actual_new_costs")
    plt.xlabel("iteration")
    plt.ylabel("costs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
