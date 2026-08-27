"""
Neuralign decomposition model
"""
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
    """
    Performs a joint decomposition of multiple 3-way tensors using a hybrid CP/Parafac2 model.
    Returns a NeuralignFactors object containing the estimated global and individual tensor factors.
    """
    
    def __init__(self, tensor_sequence: Sequence[TensorLike],
                 trial_labels: Sequence[Union[np.ndarray, list]] = None,
                 config: NeuralignConfig = None,
                 **overrides):
        super().__init__()
        
        print("WARNING: Tensor format should be (neurons, time, trials) for each tensor in the sequence.")
        
        # setup config
        self.cfg = config or NeuralignConfig()
        self.cfg = override_config(self.cfg, overrides)
        
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.randint(0, 2**32 - 1))
        self.set_seed_prescision(self.cfg.seed)
        
        # configure tensor and projection sequence and config
        # tensors live on the CPU by default, cuda can be used during training if specified
        self._tensor_names = []
        self._tensor_slice_names = []

        for n, value in enumerate(tensor_sequence):
            tensor = torch.as_tensor(value).detach().cpu()

            tensor_name = f"_tensor_{n}"
            slice_name = f"_tensor_slice_{n}"

            self.register_buffer(tensor_name, tensor, persistent=False)
            self.register_buffer(
                slice_name,
                tensor.mean(dim=2),
                persistent=False,
            )

            self._tensor_names.append(tensor_name)
            self._tensor_slice_names.append(slice_name)
        
        # make tensors and tensor_slices indexable by session
        self._refresh_tensor_views()
        self._validate_tensors()
        
        
        # initialize trial labels if provided
        self.trial_labels = trial_labels
        self.init_trial_labels() # prepare trial labels and indicies for each tensor if provided
        self._refresh_label_info_views() # cache directly indexable views of the registered label info buffers
        
        # initialize factors
        factors = initialize_neuralign_factors(
            self.tensors,
            self.tensor_slices,
            self.cfg.rank,
            init=self.cfg.init,
        )
        
        # Model owns the parameters to be fit
        # Different losses may toggle which parameters are trainable, but defaults are registered here
        
        # usually do not require grad so register as buffers (derived quantities) (register similar to tensors and tensor_slices)
        if self.cfg.non_negative:
            self._w = torch.nn.Parameter(factors.w.abs().clone())
        else:
            self.register_buffer("_w", factors.w.clone())
        self._Q_names = []
        for n, f in enumerate(factors.tensor_factors):
            Q_name = f"_Q_{n}"
            self.register_buffer(Q_name, f.Q.clone())
            self._Q_names.append(Q_name)
        
        # make Q indexable by session
        self._refresh_Q_views()
        
        # usually require grad
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
        
        # initialize condition centroids self.C if trial labels are provided
        self.init_condition_centroids()
        
        # print model summary
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
        """Set the random seed for reproducibility and set precision to float32"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("high") # use high precision for matrix multiplications

    def _refresh_tensor_views(self):
        """Cache directly indexable views of the registered tensor buffers."""
        self.tensors = [getattr(self, name) for name in self._tensor_names]
        self.tensor_slices = [
            getattr(self, name) for name in self._tensor_slice_names
        ]
        
    def init_trial_labels(self):
        """Prepare trial labels and indicies for each tensor if provided"""
        if self.trial_labels is None:
            self.n_conditions = 0
            self.condition_labels = None
            return # no trial labels provided, skip
        if len(self.trial_labels) != len(self.tensors):
            raise ValueError("Trial labels must contain one entry for each tensor.")
        if not all(
            tensor.shape[2] == len(labels)
            for tensor, labels in zip(self.tensors, self.trial_labels)
        ):
            raise ValueError("Trial labels must match the number of trials in each tensor.")
            
        # convert trial labels to torch tensors and encode unique labels as integers
        self.trial_labels = [np.asarray(labels) for labels in self.trial_labels]
        all_labels = np.concatenate(self.trial_labels)
        unique_labels = [unique.item() for unique in np.unique(all_labels)]
        self.condition_labels = unique_labels
        self.n_conditions = len(unique_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        
        # collect trial indices for each condition for each tensor
        self.label_info = [] # model level label info (tensor index, condition index, trial indices)
        self.session_label_info = [] # session level label info (condition index, trial indices)
        self._label_info_buffer_names = []
        for n, labels in enumerate(self.trial_labels):
            indices = torch.tensor([label_to_index[label.item()] for label in labels], dtype=torch.long)
            unique_indices = torch.unique(indices)
            condition_info = []
            
            # collect trial indices for each condition
            max_warning_count = 5
            for cond in unique_indices:
                cond_indices = torch.nonzero(indices == cond, as_tuple=True)[0]
                if len(cond_indices) == 0:
                    max_warning_count -= 1
                    if max_warning_count > 0:
                        print(f"Warning: No trials found for condition {cond.item()} in tensor {n}.")
                    continue
                condition_info.append((n, cond.item(), cond_indices))
                buffer_name = f"_label_info_{n}_{cond.item()}"
                self.register_buffer(buffer_name, cond_indices, persistent=False)
                self._label_info_buffer_names.append(buffer_name)

            self.label_info.extend(condition_info) # can access as a full iterable
            self.session_label_info.append(condition_info) # can access iterable by session (tensor) index
                
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
        """ Make sure buffers move elegantly with the decomposition"""
        super()._apply(fn, recurse=recurse)
        self._refresh_tensor_views()
        self._refresh_Q_views()
        self._refresh_label_info_views()
        return self
    
    def init_condition_centroids(self):
        if self.trial_labels is None:
            return # no trial labels provided, skip
        # initialize condition centroids as the mean of the trial factors for each condition across all tensors
        C = [[] for _ in range(self.n_conditions)]

        for n, cond, cond_indices in self.label_info:
            m_nc = torch.mean(self.U[n].index_select(0, cond_indices), dim=0) # mean of the n-th tensor's trial factors for condition c
            C[cond].append(m_nc)
        
        C = torch.stack([torch.mean(torch.stack(c), dim=0) for c in C]) # mean across tensors for each condition
        if self.cfg.non_negative:
            C = C.abs()
        self._C = torch.nn.Parameter(C) # shape (n_conditions, rank)
    
    def __repr__(self):
        max_print = 10
        print_str = f"NeuralignModel(rank={self.cfg.rank}, max_iter={self.cfg.max_iter}, lr={self.cfg.lr})\n"
        print_str += f"Number of tensors: {len(self.tensors)} Number of conditions: {self.n_conditions}\n"
        print_str += "Tensor\tneurons\ttime\ttrials\n"
        
        # print the first max_print - 1 tensors and the last tensor if there are more than max_print tensors
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
        
        # reorder the factors based on global weights w
        return neuralign_factors.reorder_factors()
    
    @torch.no_grad()
    def finalize(self, trainer):
        """ Finalize the decomposition by fixing the gauge and re-computing Q_n for each tensor.
        Called by the trainer after training is complete
        """
        # Fix the gauge of the decomposition into the global weights
        
        if self.cfg.gauge_fix and not self.cfg.non_negative:
            gauge_fix_model(self, sign_fix=True if self.cfg.non_negative else False)

        # Recompute Q_n for each tensor
        trainer.losses["Coordinates"][0].update_projections(self)
        
    def fit(self, device):
        """ Fit the Neuralign decomposition model to the tensor sequence using 'cpu' or 'cuda' device.
        Returns a NeuralignFactors object 
        """
        print(f"Fitting NeuralignModel on device: {device}")
        
        # move model to device
        self.to(device) 
        
        callbacks = [EarlyStopping(self.cfg.patience, self.cfg.tol)]
        if self.cfg.progress:
            callbacks.append(ProgressBar())
        if self.cfg.gauge_fix and not self.cfg.non_negative:
            callbacks.append(GaugeFixing())

        losses = {}
        # losses
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
        
        # validate tensors have the same number of time points
        time_points = [tensor.shape[1] for tensor in self.tensors]
        if len(set(time_points)) > 1:
            raise ValueError("All tensors must have the same number of time points (second dimension).")
        
        # validate tensors are positive if non_negative is True
        if self.cfg.non_negative:
            for n, tensor in enumerate(self.tensors):
                if (tensor < 0).any():
                    raise ValueError(f"Tensor {n} contains negative values but non_negative is set to True.")        

# helper class
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
