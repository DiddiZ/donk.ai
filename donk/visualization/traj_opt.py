import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd


def visualize_iLQR(output_file, results, kl_step: float, opt_result=None, cost_color="C0", constraint_color="C1"):

    results = pd.DataFrame(results).sort_values("eta")

    plt.plot(results["eta"], results["expected_costs"], color=cost_color, label="costs")
    if opt_result is not None:
        plt.scatter([opt_result.eta], [opt_result.expected_costs], color=cost_color, label="minimal constrained costs")
    plt.plot(results["eta"], results["kl_div"], label="constraint surface", color=constraint_color)
    plt.axhline(kl_step, color=constraint_color, linewidth=1, linestyle="dashed", label="constraint threshold")

    plt.fill_between(
        results["eta"],
        0,
        1,
        where=results["kl_div"] <= kl_step,
        facecolor=constraint_color,
        alpha=0.1,
        transform=mtransforms.blended_transform_factory(plt.gca().transData,
                                                        plt.gca().transAxes),
        label="constraint",
    )

    plt.xlabel(r"$\eta$")
    plt.xscale("log")
    plt.yscale("log")

    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
