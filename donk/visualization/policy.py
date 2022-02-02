from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from donk.policy import Policy


def visualize_policy_actions(
    output_file: Path,
    X: np.ndarray,
    policies: List[Policy],
    policy_labels: List[str],
    actions: List[int],
    action_labels: List[str],
    action_scaler=None,
):
    """Compares the actions of different policies of the same sates.

    Args:
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
