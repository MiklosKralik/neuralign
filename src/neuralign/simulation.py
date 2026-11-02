"""Small rank-two simulation inspired by the manuscript's recovery example."""

import numpy as np
import torch


def simulate_sessions(n_sessions=4, *, n_time=64, noise_std=0.15, seed=42):
    """Return ``(sessions, truth)`` for a small, reproducible simulation.

    Sessions have different neuron/trial counts and share sigmoid and transient
    temporal factors. Light Gaussian noise is added to each rank-two signal. Inputs use
    ``(neurons, time, trials)`` order, ready for the default Frobenius loss.
    ``noise_std`` is relative to each signal's standard deviation.
    ``truth`` contains time in seconds, V, and per-session A, U and S.
    This is a teaching example, not the full manuscript benchmark.
    """
    if (
        not isinstance(n_sessions, int)
        or isinstance(n_sessions, bool)
        or n_sessions < 1
    ):
        raise ValueError("n_sessions must be a positive integer")
    if not isinstance(n_time, int) or isinstance(n_time, bool) or n_time < 8:
        raise ValueError("n_time must be an integer of at least eight")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    time = np.arange(n_time) * 0.05
    phase = np.linspace(0, 1, n_time)
    V = np.column_stack(
        (1 / (1 + np.exp(-10 * (phase - 0.5))), phase**4 * np.exp(-8 * phase))
    )
    V /= np.linalg.norm(V, axis=0)
    H = np.array([[1.0, 0.1], [0.1, 1.0]])
    H /= np.linalg.norm(H, axis=0)
    sessions, records = [], []
    for index in range(n_sessions):
        n_neurons, n_trials = int(rng.integers(16, 33)), int(rng.integers(24, 41))
        Q, _ = np.linalg.qr(rng.normal(size=(n_neurons, 2)))
        gains = np.array([1.0, 0.35] if index % 2 == 0 else [0.35, 1.0])
        S = gains * rng.lognormal(0, 0.15, 2)
        A = (Q @ H) * S
        U = np.column_stack(
            (
                np.maximum(np.linspace(1, -0.35, n_trials), 0.1),
                np.exp(np.linspace(1, 4, n_trials)),
            )
        )
        U /= np.linalg.norm(U, axis=0)
        signal = np.einsum("nr,tr,kr->ntk", A, V, U)
        noisy = signal + noise_std * signal.std() * rng.normal(size=signal.shape)
        sessions.append(torch.tensor(noisy, dtype=torch.float32))
        records.append({"A": A, "U": U, "S": S})
    return sessions, {"time": time, "V": V, "H": H, "sessions": records}
