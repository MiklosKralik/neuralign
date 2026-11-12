"""Simulate, fit and plot Neuralign. Run: python examples/quickstart.py.

Requires Matplotlib: python -m pip install matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from neuralign import NeuralignModel
from neuralign.simulation import simulate_sessions


def main():
    sessions, truth = simulate_sessions(seed=42)
    factors = NeuralignModel(
        sessions, rank=2, max_iter=500, seed=42, progress=False
    ).fit("cpu")
    session = factors[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    axes[0].plot(truth["time"], factors.V.detach().numpy())
    axes[0].set(title="Shared temporal factors", xlabel="Time (s)", ylabel="Weight")

    image = axes[1].imshow(
        session.neuron_factors.detach().numpy(), aspect="auto", cmap="RdBu_r"
    )
    axes[1].set(
        title="Neuron factors · session 1", xlabel="Component", ylabel="Neuron",
        xticks=[0, 1], xticklabels=[1, 2],
    )
    fig.colorbar(image, ax=axes[1], label="Weight")

    axes[2].plot(session.trial_factors.detach().numpy())
    axes[2].set(title="Trial factors · session 1", xlabel="Trial", ylabel="Weight")

    destination = Path(__file__).with_suffix(".png")
    fig.savefig(destination, dpi=160)
    print(f"Saved {destination}")
    plt.show()
    return factors


if __name__ == "__main__":
    main()
