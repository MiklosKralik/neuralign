"""Joint tensor decomposition and session alignment."""
from typing import Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from neuralign.callbacks import EarlyStopping, ProgressBar, GaugeFixing
from neuralign.configs import NeuralignConfig, override_config
from neuralign.initializers import initialize_neuralign_factors
from neuralign.losses import (TensorReconstructionLoss, GlobalGramLoss,
                              GlobalProcrustesLoss, ConditionProcrustesLoss,
                              NeuronSparsityLoss)
from neuralign.neuralign_factors import NeuralignFactors, TensorFactors
from neuralign.trainer import Trainer
from neuralign.gauges import gauge_fix_model

TensorLike = Union[np.ndarray, torch.Tensor, list]


class NeuralignModel(torch.nn.Module):
    """Fit shared and session-specific factors to neuron × time × trial tensors."""
    
    def __init__(self, tensor_sequence: Sequence[TensorLike],
                 trial_labels: Sequence[Union[np.ndarray, list]] = None,
                 trial_mask: Sequence[Union[np.ndarray, list]] = None,
                 config: NeuralignConfig = None,
                 **overrides):
        super().__init__()

        self.cfg = config or NeuralignConfig()
        self.cfg = override_config(self.cfg, overrides)
        
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.randint(0, 2**32 - 1))
        self.set_seed_prescision(self.cfg.seed)

        self._tensor_names = []
        self._tensor_slice_names = []
        self._trial_mask_names = []
        self._alignment_index_names = []

        for n, value in enumerate(tensor_sequence):
            tensor = torch.as_tensor(value).detach().cpu()

            tensor_name = f"_tensor_{n}"
            self.register_buffer(tensor_name, tensor, persistent=False)
            self._tensor_names.append(tensor_name)

        self.tensors = [getattr(self, name) for name in self._tensor_names]
        self._validate_tensors()

        # Masked trials are retained for reconstruction but omitted from all
        # quantities used to align sessions.
        self.init_trial_mask(trial_mask)

        for n, tensor in enumerate(self.tensors):
            slice_name = f"_tensor_slice_{n}"
            tensor_slice = tensor.index_select(
                2, self.alignment_indices[n]
            ).mean(dim=2)
            self.register_buffer(
                slice_name, tensor_slice, persistent=False
            )
            self._tensor_slice_names.append(slice_name)

        self._refresh_tensor_views()

        self.trial_labels = trial_labels
        self.init_trial_labels()
        self._refresh_label_info_views()

        factors = initialize_neuralign_factors(
            self.tensors,
            self.tensor_slices,
            self.cfg.rank,
            init=self.cfg.init,
        )

        if self.cfg.non_negative:
            self._w = torch.nn.Parameter(factors.w.abs().clone())
        else:
            self.register_buffer("_w", factors.w.clone())
        self._Q_names = []
        for n, f in enumerate(factors.tensor_factors):
            Q_name = f"_Q_{n}"
            self.register_buffer(Q_name, f.Q.clone())
            self._Q_names.append(Q_name)

        self._refresh_Q_views()

        H = factors.H.abs() if self.cfg.non_negative else factors.H
        self._H = torch.nn.Parameter(H.clone())
        self._S_n = torch.nn.ParameterList([
            torch.nn.Parameter((f.S.abs() if self.cfg.non_negative else f.S).clone())
            for f in factors.tensor_factors
        ])
        V = factors.V.abs() if self.cfg.non_negative else factors.V
        self._V = torch.nn.Parameter(V.clone())
        self._A = torch.nn.ParameterList([
            torch.nn.Parameter((f.A.abs() if self.cfg.non_negative else f.A).clone())
            for f in factors.tensor_factors
        ])
        self._U = torch.nn.ParameterList([
            torch.nn.Parameter((f.U.abs() if self.cfg.non_negative else f.U).clone())
            for f in factors.tensor_factors
        ])

        self.init_condition_centroids()

        if self.cfg.progress:
            print(self)

    @property
    def H(self):
        return _positive(self._H) if self.cfg.non_negative else self._H

    @property
    def w(self):
        return _positive(self._w) if self.cfg.non_negative else self._w
    
    @property
    def S_n(self):
        return _FactorAccessor(self._S_n, transform=_positive if self.cfg.non_negative else None)

    @property
    def A(self):
        return _FactorAccessor(self._A, transform=_positive if self.cfg.non_negative else None)
    
    @property
    def U(self):
        return _FactorAccessor(self._U, transform=_positive if self.cfg.non_negative else None)
    
    @property
    def V(self):
        return _positive(self._V) if self.cfg.non_negative else self._V
    
    @property
    def C(self):
        if not hasattr(self, "_C"):
            return None
        return _positive(self._C) if self.cfg.non_negative else self._C
        
    def set_seed_prescision(self, seed: int):
        """Seed NumPy and PyTorch and use float32 with high matmul precision."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("high")

    def _refresh_tensor_views(self):
        """Cache directly indexable views of the registered tensor buffers."""
        self.tensors = [getattr(self, name) for name in self._tensor_names]
        self.tensor_slices = [
            getattr(self, name) for name in self._tensor_slice_names
        ]

    def init_trial_mask(self, trial_mask):
        """Store trials excluded from alignment and their retained indices."""
        if trial_mask is None:
            trial_mask = [None] * len(self.tensors)
        if len(trial_mask) != len(self.tensors):
            raise ValueError("Trial masks must contain one entry for each tensor.")

        for n, (tensor, mask) in enumerate(zip(self.tensors, trial_mask)):
            if mask is None:
                mask = torch.zeros(tensor.shape[2], dtype=torch.bool)
            else:
                mask = torch.as_tensor(mask, dtype=torch.bool)
            if mask.ndim != 1 or len(mask) != tensor.shape[2]:
                raise ValueError(
                    "Trial masks must match the number of trials in each tensor."
                )

            indices = torch.nonzero(~mask, as_tuple=True)[0]
            if len(indices) == 0:
                raise ValueError("Each tensor must contain an unmasked trial.")

            mask_name = f"_trial_mask_{n}"
            index_name = f"_alignment_indices_{n}"
            self.register_buffer(mask_name, mask, persistent=False)
            self.register_buffer(index_name, indices, persistent=False)
            self._trial_mask_names.append(mask_name)
            self._alignment_index_names.append(index_name)

        self._refresh_trial_mask_views()

    def _refresh_trial_mask_views(self):
        self.trial_masks = [
            getattr(self, name) for name in self._trial_mask_names
        ]
        self.alignment_indices = [
            getattr(self, name) for name in self._alignment_index_names
        ]
        
    def init_trial_labels(self):
        """Index each session's alignment trials by condition."""
        if self.trial_labels is None:
            self.n_conditions = 0
            self.condition_labels = None
            return
        if len(self.trial_labels) != len(self.tensors):
            raise ValueError("Trial labels must contain one entry for each tensor.")
        if not all(
            tensor.shape[2] == len(labels)
            for tensor, labels in zip(self.tensors, self.trial_labels)
        ):
            raise ValueError("Trial labels must match the number of trials in each tensor.")
            
        # Build the condition vocabulary from alignment trials only. Masked
        # trials remain in the tensor and in the trial factors.
        self.trial_labels = [np.asarray(labels) for labels in self.trial_labels]
        all_labels = np.concatenate([
            labels[indices.cpu().numpy()]
            for labels, indices in zip(
                self.trial_labels, self.alignment_indices
            )
        ])
        unique_labels = [unique.item() for unique in np.unique(all_labels)]
        self.condition_labels = unique_labels
        self.n_conditions = len(unique_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        self.label_info = [] # (session index, condition index, trial indices)
        self.session_label_info = []
        self._label_info_buffer_names = []
        for n, labels in enumerate(self.trial_labels):
            positions = {}
            for trial in self.alignment_indices[n].tolist():
                label = labels[trial].item()
                positions.setdefault(label_to_index[label], []).append(trial)
            condition_info = []

            for cond, trials in positions.items():
                cond_indices = torch.tensor(trials, dtype=torch.long)
                condition_info.append((n, cond, cond_indices))
                buffer_name = f"_label_info_{n}_{cond}"
                self.register_buffer(buffer_name, cond_indices, persistent=False)
                self._label_info_buffer_names.append(buffer_name)

            self.label_info.extend(condition_info)
            self.session_label_info.append(condition_info)
                
    def _refresh_label_info_views(self):
        """Cache directly indexable views of the registered label info buffers."""
        if self.trial_labels is None:
            return
        self.label_info = [(n, cond, getattr(self, name)) for (n, cond, _), name in \
            zip(self.label_info, self._label_info_buffer_names)]
                        
    def _refresh_Q_views(self):
        """Cache directly indexable views of the registered Q buffers."""
        self.Q_n = [getattr(self, name) for name in self._Q_names]

    def _apply(self, fn, recurse=True):
        """Refresh cached tensor views after a device or dtype change."""
        super()._apply(fn, recurse=recurse)
        self._refresh_tensor_views()
        self._refresh_trial_mask_views()
        self._refresh_Q_views()
        self._refresh_label_info_views()
        return self
    
    def init_condition_centroids(self):
        if self.trial_labels is None:
            return
        # Average each condition's trial factors across sessions.
        C = [[] for _ in range(self.n_conditions)]

        for n, cond, cond_indices in self.label_info:
            m_nc = torch.mean(self.U[n].index_select(0, cond_indices), dim=0)
            C[cond].append(m_nc)
        
        C = torch.stack([torch.mean(torch.stack(c), dim=0) for c in C])
        if self.cfg.non_negative:
            C = C.abs()
        self._C = torch.nn.Parameter(C) # shape (n_conditions, rank)
    
    def __repr__(self):
        max_print = 10
        print_str = f"NeuralignModel(rank={self.cfg.rank}, max_iter={self.cfg.max_iter}, lr={self.cfg.lr})\n"
        print_str += f"Number of tensors: {len(self.tensors)} Number of conditions: {self.n_conditions}\n"
        print_str += "Tensor\tneurons\ttime\ttrials\n"

        for n, tensor in enumerate(self.tensors[:max_print - 1]):
            print_str += f"{n}\t{tensor.shape[0]}\t{tensor.shape[1]}\t{tensor.shape[2]}\n"
        if len(self.tensors) > max_print:
            print_str += "...\n"
            tensor = self.tensors[-1]
            print_str += f"{len(self.tensors) - 1}\t{tensor.shape[0]}\t{tensor.shape[1]}\t{tensor.shape[2]}\n"
        return print_str

    def export_factors(self, history=None, loss_history=None):
        tensor_factors = []
        for n in range(len(self.A)):
            tensor_factors.append(TensorFactors(
                A=self.A[n].detach().clone(),
                U=self.U[n].detach().clone(),
                Q=self.Q_n[n].detach().clone(),
                S=self.S_n[n].detach().clone(),
            ))

        neuralign_factors = NeuralignFactors(
            w=self.w.detach().clone(),
            V=self.V.detach().clone(),
            H=self.H.detach().clone(),
            tensor_factors=tensor_factors,
            C=getattr(self, "C", None).detach().clone() if  getattr(self, "C", None) is not None else None,
            condition_labels=(
                list(self.condition_labels)
                if self.condition_labels is not None
                else None
            ),
            config=self.cfg,
            metadata={
                "loss": list(history or []),
                "losses": {
                    name: list(values)
                    for name, values in (loss_history or {}).items()
                },
            },
        )

        return neuralign_factors.reorder_factors()
    
    @torch.no_grad()
    def finalize(self, trainer):
        """Fix the gauge and recompute session projections after training."""

        if self.cfg.gauge_fix and not self.cfg.non_negative:
            gauge_fix_model(self, sign_fix=True if self.cfg.non_negative else False)

        trainer.losses["Coordinates"][0].update_projections(self)
        
    def fit(self, device):
        """Fit on ``device`` (``"cpu"`` or ``"cuda"``) and return NeuralignFactors."""
        print(f"Fitting NeuralignModel on device: {device}")

        self.to(device) 
        
        callbacks = [EarlyStopping(self.cfg.patience, self.cfg.tol)]
        if self.cfg.progress:
            callbacks.append(ProgressBar())
        if self.cfg.gauge_fix and not self.cfg.non_negative:
            callbacks.append(GaugeFixing())

        losses = {}

        if self.cfg.lambda_tensor_reconstruction > 0.0:
            losses["Tensor"] = (
                TensorReconstructionLoss([torch.linalg.norm(tensor) ** 2 for tensor in self.tensors],
                                         self.cfg.loss_type),
                self.cfg.lambda_tensor_reconstruction,
            )
        if self.cfg.lambda_global_gram > 0.0:
            losses["Geometry"] = (
                GlobalGramLoss(),
                self.cfg.lambda_global_gram,
            )
        if self.cfg.lambda_procrustes > 0.0:
            if self.trial_labels is not None:
                losses["Coordinates"] = (
                    ConditionProcrustesLoss(self, k=self.cfg.procrustes_every_k),
                    self.cfg.lambda_procrustes,
                )
            else:
                losses["Coordinates"] = (
                    GlobalProcrustesLoss(self, k=self.cfg.procrustes_every_k),
                    self.cfg.lambda_procrustes,
                )
        if self.cfg.lambda_neuron_sparsity > 0.0:
            losses["Sparsity"] = (
                NeuronSparsityLoss(),
                self.cfg.lambda_neuron_sparsity,
            )
        trainer = Trainer(
            self,
            losses,
            lr=self.cfg.lr,
            max_steps=self.cfg.max_iter,
            callbacks=callbacks,
        )
        return trainer.fit() # returns NeuralignFactors object with metadata
    
    def _validate_tensors(self):

        time_points = [tensor.shape[1] for tensor in self.tensors]
        if len(set(time_points)) > 1:
            raise ValueError("All tensors must have the same number of time points (second dimension).")

        if self.cfg.non_negative:
            for n, tensor in enumerate(self.tensors):
                if (tensor < 0).any():
                    raise ValueError(f"Tensor {n} contains negative values but non_negative is set to True.")        


class _FactorAccessor:
    def __init__(self, raw, transform=None):
        self.raw = raw
        self.transform = transform

    def __getitem__(self, idx):
        x = self.raw[idx]
        return self.transform(x) if self.transform is not None else x
    
    def __setitem__(self, idx, value):
        self.raw[idx] = value

    def __len__(self):
        return len(self.raw)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _positive(x):
    return F.softplus(x) + 1e-12
