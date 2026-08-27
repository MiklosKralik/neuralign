from typing import Literal, Sequence

import torch

from .neuralign_factors import NeuralignFactors, TensorFactors

Initialization = Literal["svd", "random"]


def initialize_trial_factors(
    tensor: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Initialize trial factors by least squares with w, A, and V fixed."""
    basis = torch.einsum("r,ir,jr->ijr", w, A, V).reshape(-1, len(w))
    values = tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1).T
    return torch.linalg.lstsq(basis, values).solution.T


def initialize_neuralign_factors(
    tensors: Sequence[torch.Tensor],
    tensor_slices: Sequence[torch.Tensor],
    rank: int,
    init: Initialization = "svd") -> NeuralignFactors:
    """Initialize PARAFAC2 factors and projections using only PyTorch on the cpu."""
    
    assert len(tensors) == len(tensor_slices), "tensors and tensor_slices must have the same length"
    assert all(t.ndim == 3 for t in tensors), "all tensors must be 3D"
    assert tensors[0].shape[1] == tensor_slices[0].shape[1], "tensor slices must have the same number of time points as the tensors"
    assert tensors[0].dtype == tensor_slices[0].dtype, "tensor slices must have the same dtype as the tensors"
    assert tensors[0].device == torch.device('cpu') and tensor_slices[0].device == torch.device('cpu'), "tensors and tensor_slices must be on the cpu"
    
    dtype = tensors[0].dtype
    n_tensors = len(tensors)
    n_time = tensors[0].shape[1]
    
        
    # initialize projections
    if init == "random":
        tensor_factors = []
        for t in range(n_tensors):
            n_neurons = tensors[t].shape[0]
            n_trials = tensors[t].shape[2]
            
            A = torch.randn(n_neurons, rank, dtype=dtype)
            U = torch.randn(n_trials, rank, dtype=dtype)
            Q = torch.linalg.qr(A).Q  # orthonormalize A to get Q
            S = torch.ones(rank, dtype=dtype)
            
            tensor_factors.append(TensorFactors(A=A, U=U, Q=Q, S=S))
            
        neuralign_factors = NeuralignFactors(
            w=torch.ones(rank, dtype=dtype),
            V=torch.randn(n_time, rank, dtype=dtype),
            H=torch.eye(rank, dtype=dtype),
            tensor_factors=tensor_factors
        )
            
    elif init == "svd":
        tensor_factors = []
        weights = torch.ones(rank, dtype=dtype)
        session_scales = [torch.ones(rank, dtype=dtype) for _ in range(n_tensors)]
        mixing_matrix = torch.eye(rank, dtype=dtype)
        
        # use covariance matrix or concatenated slices to initialize global temporal factors
        total_neurons = sum(t.shape[0] for t in tensor_slices)
        if total_neurons > n_time:
            shared_matrix = sum(
                tensor_slices[t].T @ tensor_slices[t] for t in range(n_tensors) # (time, time)
            )
        else:
            shared_matrix = torch.cat(tensor_slices, dim=0).T # (time, total_neurons)
        V = torch.linalg.svd(shared_matrix, full_matrices=False).U[:, :rank] # (time, rank)
        
        # solve procrustus problem
        projections = []
        for t in range(n_tensors):
            scaled_time_factors = session_scales[t].unsqueeze(1) * V.T
            lhs = mixing_matrix @ scaled_time_factors # (rank, time)
            left, _, right_h = torch.linalg.svd(lhs @ tensor_slices[t].T, full_matrices=False)
            projections.append((left @ right_h).T) # (neurons, rank)
        
        # populate tensor factors
        for t in range(n_tensors):
            tensor_factors.append(TensorFactors(
                A=projections[t], # because scales and mixing are exactly 1
                U=initialize_trial_factors(
                    tensors[t], weights, projections[t], V
                ),
                Q=projections[t],
                S=session_scales[t],
            ))
        
        neuralign_factors = NeuralignFactors(
            w=weights,
            V=V,
            H=mixing_matrix,
            tensor_factors=tensor_factors
        )
            
    else:
        raise ValueError(f"Invalid initialization method: {init}")

    return neuralign_factors
