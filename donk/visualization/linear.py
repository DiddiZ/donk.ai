"""Visualization tool for linear models."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


def visualize_linear_model(
    output_file: Path,
    coeff,
    intercept,
    cov,
    x,
    y=None,
    N=100,
    coeff_label='coefficients',
    intercept_label='intercept',
    cov_label='covariance',
    y_label='prediction',
    time_label='$t$',
    export_data=True,
):
    """Creates a figure visualizing a timeseries of linear Gausian models.

    Args:
        output_file: File to write the plot to.
        coeff: Linear coefficients. Shape: (T, dY, dX)
        intercept: Constants. Shape: (T, dY)
        cov: Covariances. Shape: (T, dY, dY)
        x: Shape (T, dX)
        y: Optional. Shape (T, dY)
        N: Number of random samples drawn to visualize variance.
    """
    fig = plt.figure(figsize=(16, 12))

    T, dY, dX = coeff.shape

    # Check shapes
    assert intercept.shape == (T, dY), f"{intercept.shape} != {(T, dY, dX)}"
    assert cov.shape == (T, dY, dY), f"{cov.shape} != {(T, dY, dY)}"
    assert x.shape == (T, dX), f"{x.shape} != {(T, dX)}"
    if y is not None:
        assert y.shape == (T, dY), f"{y.shape} != {(T, dY)}"

    # Intercept
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel(intercept_label)
    ax1.set_xlabel(time_label)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(linestyle=':')
    for dim in range(dY):
        line, = ax1.plot(np.arange(T), intercept[:, dim], linewidth=1)

    # Coefficients
    ax2 = fig.add_subplot(222, sharex=ax1)
    ax2.set_ylabel(coeff_label)
    ax2.set_xlabel(time_label)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dX):
            line, = ax2.plot(np.arange(T, dtype=int), coeff[:, dim1, dim2], linewidth=1)

    # Covariance
    ax3 = fig.add_subplot(223, sharex=ax1)
    ax3.set_ylabel(cov_label)
    ax3.set_xlabel(time_label)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dY):
            line, = ax3.plot(np.arange(T), cov[:, dim1, dim2], linewidth=1)

    # Prediction
    y_ = np.empty((N, T, dY))  # Approx y using the model
    for t in range(T):
        mu = np.dot(coeff[t], x[t]) + intercept[t]
        y_[:, t] = np.random.multivariate_normal(mean=mu, cov=cov[t], size=N)
    y_mean = np.mean(y_, axis=0)
    y_std = np.std(y_, axis=0)

    ax4 = fig.add_subplot(224, sharex=ax1)
    ax4.set_ylabel(y_label)
    ax4.set_xlabel(time_label)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.grid(linestyle=':')
    for dim in range(dY):
        line, = ax4.plot(np.arange(T), y_mean[:, dim], linewidth=1)
        c = line.get_color()
        if y is not None:
            ax4.plot(np.arange(T), y[:, dim], ':', color=c)
        ax4.fill_between(
            np.arange(T),
            y_mean[:, dim] - y_std[:, dim],
            y_mean[:, dim] + y_std[:, dim],
            facecolor=c,
            alpha=0.25,
            interpolate=True
        )

    # Export
    plt.tight_layout()
    if output_file is not None:
        fig.savefig(output_file)
        if export_data:
            data_file = output_file.parent / output_file.stem
            np.savez_compressed(data_file, coeff=coeff, intercept=intercept, cov=cov, y=y, y_mean=y_mean, y_std=y_std)
    else:
        plt.show()
    plt.close(fig)


def visualize_coefficients(output_file_pattern, coeff):
    """Visualize coefficietns of a linear model.

    Args:
        output_file_pattern: Pattern for files to write the plots to.
        coeff: shape (T, dY, dX), linear coefficients
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    T, dY, dX = coeff.shape

    for y in range(dY):
        df = pd.DataFrame(
            ((t, x, coeff[t, y, x]) for t in range(T) for x in range(dX)),
            columns=["t", "x", "coeff"],
        )

        sns.boxplot(data=df, x="x", y="coeff")
        plt.ylabel(f"$coeff_{{{y},x}}$")

        plt.tight_layout()
        if output_file_pattern is not None:
            plt.savefig(output_file_pattern.format(y))
        else:
            plt.show()
        plt.close()


def visualize_covariance(output_file, covar):
    """Visualize the covariance matrix of a linear model.

    Args:
        output_file: File to write the plot to.
        covar: shape (dX, dX), covariance matrix
    """
    sns.heatmap(
        covar,
        mask=np.triu(np.ones_like(covar, dtype=bool), 1),  # Mask upper triangle
    )

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()


def visualize_prediction(output_file_pattern, prediction, truth):
    """Visualize the prediction errors of a linear model.

    Args:
        output_file_pattern: Pattern for files to write the plots to.
        prediction: shape (N, T, dY), model predictions
        truth: shape (N, T, dY), target values
    """
    N, T, dY = prediction.shape

    for y in range(dY):
        if N <= 4:
            for n in range(N):
                plt.subplot(2, 2, 1 + n)
                plt.plot(prediction[n, :, y], c="C0", linewidth=1, label="prediction" if n + 1 == N else None)
                plt.plot(truth[n, :, y], c="C1", linewidth=1, label="truth" if n + 1 == N else None)
            plt.legend()
        else:
            plt.plot(prediction[:, :, y].mean(axis=0), c="C0", linewidth=1)
            plt.fill_between(
                np.arange(T),
                prediction[:, :, y].min(axis=0),
                prediction[:, :, y].max(axis=0),
                facecolor="C0",
                alpha=0.25,
                interpolate=True
            )
            plt.plot(truth[:, :, y].mean(axis=0), c="C1", linewidth=1)
            plt.fill_between(
                np.arange(T),
                truth[:, :, y].min(axis=0),
                truth[:, :, y].max(axis=0),
                facecolor="C1",
                alpha=0.25,
                interpolate=True
            )

        plt.tight_layout()
        if output_file_pattern is not None:
            plt.savefig(output_file_pattern.format(y))
        else:
            plt.show()
        plt.close()


def visualize_prediction_error(output_file, errors):
    """Visualize the prediction errors of a linear model.

    Args:
        output_file: File to write the plot to.
        errors: shape (N, T), model prediction errors
    """
    plt.plot(errors.mean(axis=0))

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()


def visualize_predictor_target_correlation(output_file, X, Y, xlabel="$x$", ylabel="$y$"):
    """Visualize the correlation between predictors and targets.

    Args:
        output_file: File to write the plot to.
        X: shape (N, dX), predictors
        Y: shape (N, dY), targets
    """
    dX, dY = X.shape[-1], Y.shape[-1]
    corr = np.empty((dY, dX))
    for i in range(dX):
        for j in range(dY):
            corr[j, i] = np.cov(X[:, i], Y[:, j])[0, 1]  # TODO can be calculated more efficiently
    sns.heatmap(corr, cmap="bwr", vmin=-1, vmax=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Correlation")

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()


def visualize_predictor_target_scatter(output_file, X, Y, cmap=plt.cm.plasma):
    """Visualize the correlation between predictors and targets.

    Args:
        output_file: File to write the plot to.
        X: shape (N, T, dX), predictors
        Y: shape (N, T, dY), targets
    """
    N, T, dX = X.shape
    dY = Y.shape[-1]

    _, axes = plt.subplots(dX, dY, squeeze=False, figsize=(dY * 2, dX * 2))
    colors = cmap(np.tile(np.linspace(0, 1, T), (N, 1)).flatten())
    for i in range(dX):  # Columns
        for j in range(dY):  # Rows
            ax = axes[i][j]
            ax.scatter(X[:, :, i].flatten(), Y[:, :, j].flatten(), c=colors, alpha=0.5)
            ax.axis("equal")
            ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    plt.close()
