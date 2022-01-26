from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd


def visualize_iLQR(output_file: Path, results, kl_step: float, opt_result=None, cost_color="C0", constraint_color="C1"):

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

    # Write data to csv
    results.to_csv(
        output_file.parent / (output_file.stem + ".csv"),
        columns=["eta", "kl_div", "expected_costs"],
        index=False,
    )
