"""Minimal, self-contained Neuralign example.

The example generates three sessions with shared temporal factors but different
neurons and trials, fits a rank-two model, inspects the exported factors, and
adapts the learned global representation to one new session.
"""

from __future__ import annotations

import torch

from neuralign import NeuralignConfig, NeuralignModel


def simulate_session(
    generator: torch.Generator,
    temporal_factors: torch.Tensor,
    n_neurons: int,
    n_trials: int,
) -> torch.Tensor:
    rank = temporal_factors.shape[1]
    neuron_factors = torch.randn(n_neurons, rank, generator=generator)
    trial_factors = torch.randn(n_trials, rank, generator=generator)
    signal = torch.einsum("nr,tr,kr->ntk", neuron_factors, temporal_factors, trial_factors)
    noise = 0.05 * torch.randn(signal.shape, generator=generator)
    return signal + noise


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    time = torch.linspace(0, 1, 40)
    temporal_factors = torch.stack(
        (torch.sin(2 * torch.pi * time), torch.cos(torch.pi * time)), dim=1
    )

    sessions = [
        simulate_session(generator, temporal_factors, n_neurons=14, n_trials=24),
        simulate_session(generator, temporal_factors, n_neurons=18, n_trials=29),
        simulate_session(generator, temporal_factors, n_neurons=16, n_trials=21),
    ]
    config = NeuralignConfig(
        rank=2,
        max_iter=300,
        patience=80,
        progress=False,
        seed=11,
    )
    factors = NeuralignModel(sessions, config=config).fit("cpu")

    print("shared temporal factors:", tuple(factors.V.shape))
    for index, (session, view) in enumerate(zip(sessions, factors)):
        relative_error = torch.linalg.norm(session - view.reconstructed_tensor()) / torch.linalg.norm(session)
        print(
            f"session {index}: neurons={tuple(view.neuron_factors.shape)}, "
            f"trials={tuple(view.trial_factors.shape)}, "
            f"latent={tuple(view.latent_trajectories().shape)}, "
            f"relative reconstruction error={relative_error:.3f}"
        )

    held_out = simulate_session(generator, temporal_factors, n_neurons=12, n_trials=19)
    adapted = factors.partial_fit(
        held_out,
        device="cpu",
        max_iter=150,
        patience=50,
        progress=False,
        seed=19,
    )
    print("sessions after held-out adaptation:", len(adapted))


if __name__ == "__main__":
    main()
