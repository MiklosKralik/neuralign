"""
Neuralign decomposition factors

CP form:
A_n : neuron factors (neurons, rank)
V : global temporal factors (time, rank)
U_n : trial factors (trials, rank)

PARAFAC2 components:
w : tensor weights (diagonal) (rank,)
Q_n : orthonormal projection matrices (neurons, rank)
H : global mixing matrix (rank, rank)
S_n : component scaling matrix (diagonal) (rank,)
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import torch
from torch import Tensor

import neuralign.ops as ops
from neuralign.configs import NeuralignConfig

@dataclass
class TensorFactors:
    """
    Store the hybrid factors of a single 3-way tensor in CP/Parafac format
    """
    
    A: Tensor = None # individual neuron factors
    U: Tensor = None # individual trial factors
    Q: Tensor = None # individual neural orthonormal projection matrices
    S: Tensor = None # individual component scaling matrices (diagonal)
    
    ##### Helper methods #####
    
    def to(self, device, **kwargs):
        """Move all tensors to ``device`` in place and return this object."""
        for name in ("A", "U", "Q", "S"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=device, **kwargs))
        return self
    

@dataclass
class NeuralignFactors:
    """
    Store a set of tensor factors for many 3-way tensors from a joint neuralign decomposition
    """
    
    # core factors
    w: Tensor = None # global tensor weights (diagonal)
    V: Tensor = None # global temporal factors
    H: Tensor = None # global mixing matrix
    tensor_factors: List[TensorFactors] = None # list of individual tensor factors
    
    # optional condition prototypes
    C: Tensor = None # global condition prototypes (conditions, rank)
    
    config: Union[NeuralignConfig, Dict] = None # config object
    metadata: dict = None # optional metadata dictionary (losses, convergence, etc.)
    condition_labels: List[Any] = None # labels corresponding to rows of C
    
    ##### Helper methods #####
    
    def __len__(self):
        return len(self.tensor_factors)
    
    def to(self, device, **kwargs):
        """Move all tensors to  in place and return this object."""
        for name in ("w", "V", "H", "C"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=device, **kwargs))
        for factors in self.tensor_factors or []:
            factors.to(device, **kwargs)
        return self
    
    def save(self, path):
        """Save the factors and metadata to ``path``."""
        torch.save(
            {   "w": self.w,
                "V": self.V,
                "H": self.H,
                "tensor_factors": [
                    {
                        "A": factors.A,
                        "U": factors.U,
                        "Q": factors.Q,
                        "S": factors.S,
                    }
                    for factors in self.tensor_factors or []
                ],
                "C": self.C,
                "condition_labels": self.condition_labels,
                "config": self.config,
                "metadata": self.metadata,
            },
            path,
        )
    
    @classmethod
    def load(cls, path, map_location="cpu"):
        """Load factors saved by :meth:`save` from a trusted file."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            w=data["w"],
            V=data["V"],
            H=data["H"],
            tensor_factors=[
                TensorFactors(**factors) for factors in data["tensor_factors"]
            ],
            C=data.get("C"),
            condition_labels=data.get("condition_labels"),
            config=data.get("config"),
            metadata=data.get("metadata"),
        )

    def partial_fit(
        self,
        tensor,
        trial_labels: Sequence[Any] = None,
        *,
        device=None,
        config: Union[NeuralignConfig, Dict] = None,
        **overrides,
    ) -> "NeuralignFactors":
        """Fit one or more new tensors while keeping learned factors fixed.

        ``tensor`` may be one 3-way tensor or a sequence of them. A cloned
        factor collection is returned with the new sessions appended. For
        multiple tensors, provide one trial-label sequence per tensor. The
        stored config is inherited by default and can be changed with
        ``config`` or keyword overrides.
        """
        # Local import avoids a module cycle: adapt imports these factor types.
        from neuralign.adapt import partial_fit_factors

        return partial_fit_factors(
            self,
            tensor,
            trial_labels=trial_labels,
            device=device,
            config=config,
            **overrides,
        )
        
    def session(self, index: int) -> "SessionFactorView":
        """
        Return a read-only view of the factors for a single tensor/session.
        """
        if index < 0:
            index += len(self) # for negative indexing
        if not (0 <= index < len(self)):
            raise IndexError(f"index {index} is out of bounds for NeuralignFactors of length {len(self)}")
        
        return SessionFactorView(parent=self, index=index)
    
    def __getitem__(self, index: int) -> "SessionFactorView":
        return self.session(index)
    
    def reorder_factors(self) -> "NeuralignFactors":
        """
        Reorder the factors by the magnitude of the global tensor weights w.
        This is useful for visualization and interpretation.
        """
        order = torch.argsort(self.w, descending=True)
        self.w = self.w[order]
        self.V = self.V[:, order]
        self.H = self.H[order][:, order]
        if self.C is not None:
            self.C = self.C[:, order]
        for factors in self.tensor_factors or []:
            factors.A = factors.A[:, order]
            factors.U = factors.U[:, order]
            factors.Q = factors.Q[:, order]
            factors.S = factors.S[order]
        return self
    
    
@dataclass(frozen=True)
class SessionFactorView:
    """
    Read-only view of the factors for a single tensor/session in a NeuralignFactors object.
    """
    
    parent: "NeuralignFactors"
    index: int
    
    ### fixed session factors ###
    
    @property
    def neuron_factors(self):
        """
        Return A_n: neuron factors for this session (neurons, rank)
        """
        return self.parent.tensor_factors[self.index].A
    
    @property
    def trial_factors(self):
        """
        Return U_n: trial factors for this session (trials, rank)
        """
        return self.parent.tensor_factors[self.index].U
    
    @property
    def projection_matrix(self):
        """
        Return Q_n: orthonormal projection matrix for this session (neurons, rank)
        """
        return self.parent.tensor_factors[self.index].Q
    
    @property
    def scaling_matrix(self):
        """
        Return S_n: component scaling matrix for this session (diagonal) (rank,)
        """
        return self.parent.tensor_factors[self.index].S
    
    ##### global factors #####
    
    @property
    def weight_matrix(self):
        """
        Return w: global tensor weights (diagonal) (rank,)
        """
        return self.parent.w
    
    @property
    def temporal_factors(self):
        """
        Return V: global temporal factors (time, rank)
        """
        return self.parent.V
    
    @property
    def mixing_matrix(self):
        """
        Return H: global mixing matrix (rank, rank)
        """
        return self.parent.H
    
    ##### derived factors #####
    
    def scaled_mixing_matrix(self):
        """
        Return H_n = H @ diag(S_n): scaled mixing matrix for this session (rank, rank)
        """
        return ops.scale_mixing(self.mixing_matrix, self.scaling_matrix)
    
    def predicted_neuron_factors(self):
        """
        These are the coordinate-aligned neuron factors (Hard alignment constraint)
        The learned neurons factors have a softer alignment constraint
        Return A_n_hat = Q_n @ H_n: predicted neuron factors for this session (neurons, rank)
        """
        return ops.predicted_neuron_factors(self.projection_matrix, self.mixing_matrix, self.scaling_matrix)
    
    def latent_trajectories(self, layout="ktr"):
        """
        Return the latent latent trajectories for this session, 
        i.e. the predicted trajectoris in the low-dimensional latent space (trials (k), time (t), rank (r))
        """
        out = ops.predicted_latent_trajectories(
            self.weight_matrix, self.mixing_matrix, self.scaling_matrix,
            self.temporal_factors, self.trial_factors
        )
        return _apply_layout(out, layout)
    
    def projected_trajectories(self, Z, layout="ktr"):
        """
        Return the Q-projected latent trajectories for this session,
        Z is a neural activity tensor (neurons, time, trials) 
        i.e. the neural activity tensor projected into the low-dimensional latent space (trials (k), time (t), rank (r))
        """
        out = ops.projected_latent_trajectories(
            self.projection_matrix, Z
        )
        return _apply_layout(out, layout)
    
    def reconstructed_tensor(self):
        """
        Return the reconstructed neural activity tensor for this session (neurons, time, trials)
        """
        return ops.reconstruct_tensor(
            self.weight_matrix, self.neuron_factors, self.temporal_factors, self.trial_factors
        )


def _apply_layout(value, layout):
    # Canonical input is (trials, time, components).
    if layout == "ktr":
        return value
    if layout == "tkr":
        return value.permute(1, 0, 2)
    raise ValueError("layout must be 'ktr' or 'tkr'")
