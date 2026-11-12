"""Exercise the installed package through the documented public API."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from neuralign import NeuralignFactors, NeuralignModel
from neuralign.simulation import simulate_sessions


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep these small CPU fits fast on machines with many CPU cores.
        torch.set_num_threads(1)

    def test_simulation_is_reproducible_and_matches_truth(self):
        sessions, truth = simulate_sessions(n_sessions=2, n_time=16, noise_std=0, seed=7)
        repeated, _ = simulate_sessions(n_sessions=2, n_time=16, noise_std=0, seed=7)
        self.assertEqual(len(sessions), 2)
        self.assertEqual(truth["V"].shape, (16, 2))
        for actual, same, factors in zip(sessions, repeated, truth["sessions"]):
            torch.testing.assert_close(actual, same, rtol=0, atol=0)
            expected = np.einsum("nr,tr,kr->ntk", factors["A"], truth["V"], factors["U"])
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype))

    def test_fit_reconstruct_and_save(self):
        sessions, _ = simulate_sessions(n_sessions=2, n_time=16, seed=11)
        factors = NeuralignModel(
            sessions, rank=2, max_iter=12, seed=11, progress=False
        ).fit("cpu")
        self.assertEqual(len(factors), len(sessions))
        self.assertEqual(factors.V.shape, (16, 2))
        self.assertTrue(torch.isfinite(factors.V).all())
        losses = torch.tensor(factors.metadata["loss"])
        self.assertEqual(len(losses), 12)
        self.assertTrue(torch.isfinite(losses).all())
        self.assertLess(losses[-1], losses[0])
        for index, original in enumerate(sessions):
            session = factors[index]
            self.assertEqual(session.neuron_factors.shape, (original.shape[0], 2))
            self.assertEqual(session.trial_factors.shape, (original.shape[2], 2))
            reconstructed = session.reconstructed_tensor()
            self.assertEqual(reconstructed.shape, original.shape)
            self.assertTrue(torch.isfinite(reconstructed).all())
            torch.testing.assert_close(
                reconstructed,
                session.reconstructed_tensor([0]) + session.reconstructed_tensor([1]),
            )
            projection = session.projection_matrix
            torch.testing.assert_close(projection.T @ projection, torch.eye(2), atol=1e-5, rtol=1e-5)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "factors.pt"
            factors.save(destination)
            restored = NeuralignFactors.load(destination)
        self.assertEqual(restored.metadata, factors.metadata)
        self.assertEqual(restored.config, factors.config)
        torch.testing.assert_close(restored.V, factors.V, rtol=0, atol=0)
        for index in range(len(factors)):
            torch.testing.assert_close(
                restored[index].reconstructed_tensor(),
                factors[index].reconstructed_tensor(),
                rtol=0,
                atol=0,
            )

    def test_simulation_rejects_invalid_arguments(self):
        for arguments in (
            {"n_sessions": 0},
            {"n_sessions": True},
            {"n_time": 7},
            {"noise_std": -1},
            {"noise_std": float("nan")},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                simulate_sessions(**arguments)


if __name__ == "__main__":
    unittest.main()
