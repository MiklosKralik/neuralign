"""Simulate two sessions, fit Neuralign and plot their factors."""

import matplotlib.pyplot as plt

from neuralign import NeuralignModel
from neuralign.simulation import simulate_sessions

sessions, truth = simulate_sessions(n_sessions=2, seed=42)
factors = NeuralignModel(
    sessions, rank=2, max_iter=500, seed=42, progress=False
).fit("cpu")

fig, axes = plt.subplot_mosaic(
    [["time", "neurons1", "trials1"], ["time", "neurons2", "trials2"]],
    figsize=(11, 6), layout="constrained",
)
axes["time"].plot(truth["time"], factors.V.detach().numpy())
axes["time"].set(title="Shared temporal factors", xlabel="Time (s)", ylabel="Weight")

for index in range(2):
    session = factors[index]
    neurons = axes[f"neurons{index + 1}"]
    image = neurons.imshow(
        session.neuron_factors.detach().numpy(), aspect="auto", cmap="RdBu_r"
    )
    neurons.set(
        title=f"Neuron factors · session {index + 1}", ylabel="Neuron",
        xlabel="Component", xticks=[0, 1], xticklabels=[1, 2],
    )
    fig.colorbar(image, ax=neurons, label="Weight")

    trials = axes[f"trials{index + 1}"]
    trials.plot(session.trial_factors.detach().numpy())
    trials.set(
        title=f"Trial factors · session {index + 1}", xlabel="Trial", ylabel="Weight"
    )

fig.savefig("quickstart.png", dpi=160)
plt.show()
