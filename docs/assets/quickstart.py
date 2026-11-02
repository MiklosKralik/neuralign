"""Simulate, fit and plot a rank-two Neuralign model; no external data needed.

Run: python examples/quickstart.py
The figure is saved next to this script as quickstart.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuralign import NeuralignConfig, NeuralignModel
from neuralign.simulation import simulate_sessions


def main():
    sessions, truth = simulate_sessions(seed=42)
    config = NeuralignConfig(
        rank=2, max_iter=500, patience=100, seed=42, progress=False
    )
    factors = NeuralignModel(sessions, config=config).fit("cpu")
    view = factors.session(0)
    reconstructed = view.reconstructed_tensor().detach().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), layout="constrained")
    time = truth["time"]
    for column, (V, title) in enumerate(
        (
            (truth["V"], "Generating temporal factors"),
            (factors.V.detach().numpy(), "Fitted temporal factors"),
        )
    ):
        axes[0, column].plot(time, V)
        axes[0, column].set(title=title, xlabel="Time (s)", ylabel="Factor value")
    axes[0, 2].plot(view.trial_factors.detach().numpy())
    axes[0, 2].set(
        title="Fitted trial factors · session 1",
        xlabel="Trial index",
        ylabel="Factor value",
    )

    measured = sessions[0].numpy()[:, :, 0]
    predicted = reconstructed[:, :, 0]
    limit = np.quantile(np.abs(np.stack([measured, predicted])), 0.99)
    for ax, values, title in zip(
        axes[1, :2],
        (measured, predicted),
        ("Input · first trial", "Reconstruction · first trial"),
    ):
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(time[0], time[-1] + 0.05, len(values), 0),
        )
        ax.set(title=title, xlabel="Time (s)", ylabel="Neuron index")
    fig.colorbar(
        image,
        ax=axes[1, :2],
        label="Model units",
        orientation="horizontal",
        shrink=0.7,
        pad=0.15,
    )
    errors = [
        float(
            np.linalg.norm(Z.numpy() - v.reconstructed_tensor().detach().numpy())
            / np.linalg.norm(Z.numpy())
        )
        for Z, v in zip(sessions, factors)
    ]
    axes[1, 2].bar(np.arange(1, len(errors) + 1), errors, color="#536B82")
    axes[1, 2].set(
        title="Reconstruction error",
        xlabel="Session",
        ylabel="Relative Frobenius error",
        xticks=np.arange(1, len(errors) + 1),
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    # Component order, sign and scale can differ from the generating factors.
    destination = Path(__file__).with_suffix(".png")
    fig.savefig(destination, dpi=160)
    print(f"Saved {destination}; mean relative error: {np.mean(errors):.3f}")
    plt.show()
    return factors


if __name__ == "__main__":
    main()
