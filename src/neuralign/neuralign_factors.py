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
import operator
from typing import Any, Dict, List, Sequence, Union

import torch
from torch import Tensor

import neuralign.ops as ops
from neuralign.configs import NeuralignConfig


@dataclass
class TensorFactors:
    """Neuron, trial, projection and gain factors for one session."""
    
    A: Tensor = None # individual neuron factors
    U: Tensor = None # individual trial factors
    Q: Tensor = None # individual neural orthonormal projection matrices
    S: Tensor = None # individual component scaling matrices (diagonal)

    def to(self, device, **kwargs):
        """Move all tensors to ``device`` in place and return this object."""
        for name in ("A", "U", "Q", "S"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=device, **kwargs))
        return self


@dataclass
class NeuralignFactors:
    """Shared factors and per-session factors from a joint decomposition."""
    
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

    def __len__(self):
        return len(self.tensor_factors)
    
    def to(self, device, **kwargs):
        """Move all tensors to ``device`` in place and return this collection."""
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
        trial_mask: Sequence[Any] = None,
        *,
        device=None,
        config: Union[NeuralignConfig, Dict] = None,
        **overrides,
    ) -> "NeuralignFactors":
        """Fit new sessions with w, V, H and condition prototypes C frozen.

        ``tensor`` accepts one (neurons, time, trials) tensor or a sequence.
        Supply labels and masks per session; True mask entries exclude alignment
        only. The stored config is inherited, with optional overrides.

        Known labels anchor alignment to frozen prototypes; unknown labels are
        warned about and ignored. Sessions with no recognized labels use trial
        means. ``lambda_condition_centroid=None`` uses the default supervised
        penalty toward prototypes; set it to 0.0 to disable, or a positive value
        to choose its strength. Only recognized, unmasked conditions contribute.

        Returns a copy with new sessions appended. Effective settings and fit
        histories are stored in ``metadata["adaptations"]``.
        """
        # Local import avoids a module cycle: adapt imports these factor types.
        from neuralign.adapt import partial_fit_factors

        return partial_fit_factors(
            self,
            tensor,
            trial_labels=trial_labels,
            trial_mask=trial_mask,
            device=device,
            config=config,
            **overrides,
        )
        
    def session(self, index: int) -> "SessionFactorView":
        """Return a read-only factor view for one session."""
        if index < 0:
            index += len(self) # for negative indexing
        if not (0 <= index < len(self)):
            raise IndexError(f"index {index} is out of bounds for NeuralignFactors of length {len(self)}")
        
        return SessionFactorView(parent=self, index=index)
    
    def __getitem__(self, index: int) -> "SessionFactorView":
        return self.session(index)
    
    def reorder_factors(self) -> "NeuralignFactors":
        """Order components by descending global weights w for display."""
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
    """Named access to one session and its shared factors."""
    
    parent: "NeuralignFactors"
    index: int

    @property
    def neuron_factors(self):
        """Neuron factors A, shaped (neurons, rank)."""
        return self.parent.tensor_factors[self.index].A
    
    @property
    def trial_factors(self):
        """Trial factors U, shaped (trials, rank)."""
        return self.parent.tensor_factors[self.index].U
    
    @property
    def projection_matrix(self):
        """Orthonormal projection Q, shaped (neurons, rank)."""
        return self.parent.tensor_factors[self.index].Q
    
    @property
    def scaling_matrix(self):
        """Session gains S, shaped (rank,)."""
        return self.parent.tensor_factors[self.index].S

    @property
    def weight_matrix(self):
        """Shared component weights w, shaped (rank,)."""
        return self.parent.w
    
    @property
    def temporal_factors(self):
        """Shared temporal factors V, shaped (time, rank)."""
        return self.parent.V
    
    @property
    def mixing_matrix(self):
        """Shared mixing matrix H, shaped (rank, rank)."""
        return self.parent.H

    def scaled_mixing_matrix(self):
        """Return H @ diag(S), shaped (rank, rank)."""
        return ops.scale_mixing(self.mixing_matrix, self.scaling_matrix)
    
    def predicted_neuron_factors(self):
        """Return Q @ H @ diag(S), shaped (neurons, rank); soft coupling allows A to differ."""
        return ops.predicted_neuron_factors(self.projection_matrix, self.mixing_matrix, self.scaling_matrix)
    
    def latent_trajectories(self, layout="ktr", *, components=None, normalized=False):
        """Return selected-component predictions in ``ktr`` or ``tkr`` layout.

        ``components`` follows :meth:`reconstructed_tensor`. All latent
        coordinates are retained, even when selecting a single component.
        ``normalized=True`` replaces global weights w with ones.
        """
        out = self.reconstructed_latent_tensor(components, normalized=normalized).permute(2, 1, 0)
        return _apply_layout(out, layout)
    
    def projected_trajectories(self, Z, layout="ktr"):
        """Project Z (neurons, time, trials) through Q.

        Returns (trials, time, rank), or (time, trials, rank) for layout="tkr".
        """
        out = ops.projected_latent_trajectories(
            self.projection_matrix, Z
        )
        return _apply_layout(out, layout)
    
    def reconstructed_tensor(self, components=None, *, normalized=False):
        """Reconstruct selected CP components as ``(neurons, time, trials)``.

        Pass a zero-based integer or a one-dimensional sequence of distinct
        nonnegative integer indices. ``None`` uses all stored components;
        an empty sequence returns zeros. Selecting n components gives a
        rank-at-most-n contribution, without modifying or reordering factors.
        Dtype, device, and autograd connections are preserved.
        ``normalized=True`` uses unit global weights instead of w, leaving
        every other factor and the stored weights unchanged. It does not
        normalize the output norm or session gains; gradients then bypass w.
        """
        indices = _component_indices(components, self.weight_matrix)
        w = torch.ones_like(self.weight_matrix) if normalized else self.weight_matrix
        if indices is not None:
            return ops.reconstruct_tensor(
                w.index_select(0, indices),
                self.neuron_factors.index_select(1, indices),
                self.temporal_factors.index_select(1, indices),
                self.trial_factors.index_select(1, indices),
            )
        return ops.reconstruct_tensor(
            w, self.neuron_factors, self.temporal_factors, self.trial_factors
        )

    def reconstructed_latent_tensor(self, components=None, *, normalized=False):
        """Reconstruct ``(latent coordinates, time, trials)`` using w, H, S, V, U.

        ``components`` follows :meth:`reconstructed_tensor`. Selection restricts
        the contributing columns of H, not its rows: all latent coordinates
        remain in the output. This is the aligned prediction; multiplying by Q
        uses Q H diag(S), which need not equal the fitted neuron factors A.
        ``normalized=True`` replaces only global weights w with ones,
        preserving H, S, V, U and all stored factors, and bypassing w gradients.
        """
        indices = _component_indices(components, self.weight_matrix)
        w, H, S = self.weight_matrix, self.mixing_matrix, self.scaling_matrix
        if normalized:
            w = torch.ones_like(w)
        V, U = self.temporal_factors, self.trial_factors
        if indices is not None:
            w, S = w.index_select(0, indices), S.index_select(0, indices)
            H = H.index_select(1, indices)
            V, U = V.index_select(1, indices), U.index_select(1, indices)
        return ops.predicted_latent_trajectories(w, H, S, V, U).permute(2, 1, 0)


def _component_indices(components, weights):
    """Validate component selection without silently coercing floats or booleans."""
    if components is None:
        return None
    dtype = getattr(components, "dtype", None)
    if isinstance(components, bool) or dtype == torch.bool or getattr(dtype, "kind", None) == "b":
        raise TypeError("components must contain integer indices, not booleans")
    ndim = getattr(components, "ndim", 0)
    if ndim == 1:
        values = list(components)
    elif ndim != 0:
        raise TypeError("components must be one-dimensional")
    else:
        try:
            operator.index(components)
            values = [components]
        except TypeError:
            if isinstance(components, (str, bytes)):
                raise TypeError("components must be an integer or a one-dimensional integer sequence")
            try:
                values = list(components)
            except TypeError:
                raise TypeError("components must be an integer or a one-dimensional integer sequence") from None
    indices = []
    for value in values:
        dtype = getattr(value, "dtype", None)
        if (isinstance(value, bool) or dtype == torch.bool
                or getattr(dtype, "kind", None) == "b"
                or getattr(value, "ndim", 0) != 0):
            raise TypeError("components must contain scalar integer indices, not booleans or nested sequences")
        try:
            index = operator.index(value)
        except TypeError:
            raise TypeError("components must contain integer indices") from None
        if not 0 <= index < weights.numel():
            raise IndexError(f"component index {index} is out of bounds for rank {weights.numel()}")
        if index in indices:
            raise ValueError("components must contain distinct indices")
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long, device=weights.device)


def _apply_layout(value, layout):
    # Canonical input is (trials, time, components).
    if layout == "ktr":
        return value
    if layout == "tkr":
        return value.permute(1, 0, 2)
    raise ValueError("layout must be 'ktr' or 'tkr'")
