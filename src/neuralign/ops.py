"""Tensor reconstruction, projection and alignment operations."""
import torch


def scale_mixing(H, S):
    """Return H @ diag(S)."""
    return H @ torch.diag(S)


def reconstruct_A(Q, H, S):
    """Return Q @ H @ diag(S)."""
    return Q @ scale_mixing(H, S)


def reconstruct_slice(w, A, V, U):
    """Reconstruct a (neurons, time) slice using trial-averaged U."""
    return torch.einsum('r, ir,jr,r->ij', w, A, V, U.mean(dim=0))


def reconstruct_tensor(w, A, V, U):
    """Reconstruct the (neurons, time, trials) tensor from CP factors."""
    return torch.einsum('r, ir,jr,kr->ijk', w, A, V, U)


def predicted_latent_trajectories(w, H, S, V, U):
    """Return model-predicted trajectories as (trials, time, rank)."""
    return torch.einsum('ar,r,tr,kr->kta', scale_mixing(H, S), w, V, U)


def projected_latent_trajectories(Q, Z):
    """Project Z (neurons, time, trials) to (trials, time, rank) using Q."""
    return torch.einsum('ia,itk->kta', Q, Z)


def predicted_neuron_factors(Q, H, S):
    """Return Q @ H @ diag(S), shaped (neurons, rank)."""
    return Q @ scale_mixing(H, S)


@torch.no_grad()
def solve_procrustes(A, B):
    U, _, Vh = torch.linalg.svd(B @ A.T, full_matrices=False)
    return (U @ Vh).T


def gram_from_A(A):
    """Return A.T @ A."""
    return A.T @ A


def gram_from_H(H, S):
    """Return the Gram matrix of H @ diag(S)."""
    scaled_H = scale_mixing(H, S)
    return scaled_H.T @ scaled_H
