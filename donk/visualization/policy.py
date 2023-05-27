from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from donk.policy import Policy


def visualize_policy_actions(
    output_file: Path | str | None,
    X: np.ndarray,
    policies: list[Policy],
    policy_labels: list[str],
    actions: list[int],
    action_labels: list[str],
    action_scaler=None,
) -> None:
    """Compares the actions of different policies of the same sates.

    Args:
        output_file: Path to save the figure to. `None` indicates showing the figure.
        X: (N, T+1, dX) States
        policies: List of policies
        policy_labels: Names for the policies
        actions: Indices of actions to plot
        action_labels: Names for the actions
        action_scaler: sklearn Scaler
    """
    N, _, _ = X.shape
    T = X.shape[1] - 1

    # Compute actions
    U = []
    for pol in policies:
        u = pol.act(X[:, :-1], t=None)
        if action_scaler is not None:  # Inverse transform
            u = action_scaler.inverse_transform(u.reshape(N * T, pol.dU)).reshape(N, T, pol.dU)
        U.append(u)

    for j, action in enumerate(actions):
        plt.subplot(1, len(actions), j + 1)
        plt.title(action_labels[j])
        for i, u in enumerate(U):
            for n in range(N):
                plt.plot(u[n][:, action], color=f"C{i}", label=policy_labels[i] if n == 0 else None)

        plt.xlabel("$t$")
        plt.ylabel(r"$\mathbf{u}_t$")
        if j == len(actions) - 1:
            plt.legend()

    # Export
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
