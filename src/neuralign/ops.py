"""
Pure tensor operation functions for neuralign decomposition
"""
import torch

def scale_mixing(H, S):
    """
    Scale the mixing matrix H by the session scaling factors S.
    """
    return H @ torch.diag(S)

def reconstruct_A(Q, H, S):
    """
    Reconstruct the neuron factors A from the projections Q, mixing matrix H, and scaling factors S.
    """
    return Q @ scale_mixing(H, S)

def reconstruct_slice(w, A, V, U):
    """
    Reconstruct a single slice of the tensor from global weights w neuron factors A and temporal factors V.
    Trial factors U are averaged across trials to produce a single slice.
    """
    return torch.einsum('r, ir,jr,r->ij', w, A, V, U.mean(dim=0))

def reconstruct_tensor(w, A, V, U):
    """
    Reconstruct the full tensor from global scale w, neuron factors A, temporal factors V, and trial factors U.
    """
    return torch.einsum('r, ir,jr,kr->ijk', w, A, V, U)

def predicted_latent_trajectories(w, H, S, V, U):
    """
    Compute model-predicted aligned trajectories for a session.

    Returns a tensor with shape (trials, time, rank).
    """
    return torch.einsum('ar,r,tr,kr->kta', scale_mixing(H, S), w, V, U)

def projected_latent_trajectories(Q, Z):
    """
    Project a neural activity tensor into the latent space using the projection matrix Q
    and the neural activity tensor Z (neurons, time, trials).

    Returns a tensor with shape (trials, time, rank).
    """
    return torch.einsum('ia,itk->kta', Q, Z)

def predicted_neuron_factors(Q, H, S):
    """
    Compute alignment-predicted neuron factors for a session.

    Returns a tensor with shape (neurons, rank).
    """
    return Q @ scale_mixing(H, S)

@torch.no_grad()
def solve_procrustes(A, B):
    U, _, Vh = torch.linalg.svd(B @ A.T, full_matrices=False)
    return (U @ Vh).T

def gram_from_A(A):
    """
    Compute the Gram matrix from neuron factors A.
    """
    return A.T @ A

def gram_from_H(H, S):
    """
    Compute the Gram matrix from mixing matrix H and scaling factors S.
    """
    scaled_H = scale_mixing(H, S)
    return scaled_H.T @ scaled_H
