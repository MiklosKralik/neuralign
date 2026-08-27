"""Fit new sessions into an existing Neuralign decomposition."""

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import neuralign.ops as ops
from neuralign.callbacks import EarlyStopping, ProgressBar
from neuralign.configs import NeuralignConfig, from_dict, override_config, to_dict
from neuralign.initializers import initialize_trial_factors
from neuralign.losses import (
    ConditionProcrustesLoss,
    GlobalGramLoss,
    GlobalProcrustesLoss,
    NeuronSparsityLoss,
    TensorReconstructionLoss,
)
from neuralign.neuralign_factors import NeuralignFactors, TensorFactors
from neuralign.trainer import Trainer


@dataclass
class _AdaptationResult:
    factors: list[TensorFactors]
    history: list[float]
    loss_history: dict[str, list[float]]


class _AdaptationModel(torch.nn.Module):
    """Neuralign model with fixed global factors and new local factors."""

    def __init__(self, source, tensors, config, label_info):
        super().__init__()
        self.cfg = config
        self.tensors = tensors
        self.label_info = label_info

        for name in ("w", "V", "H"):
            value = getattr(source, name).detach().clone().to(tensors[0].device)
            self.register_buffer(name, value)
        if source.C is not None:
            self.register_buffer(
                "C", source.C.detach().clone().to(tensors[0].device)
            )

        initial = [
            _initialize_tensor(tensor, self.w, self.V, self.H)
            for tensor in tensors
        ]
        self._A = torch.nn.ParameterList([
            torch.nn.Parameter((factors.A.abs() if self.cfg.non_negative else factors.A).clone())
            for factors in initial
        ])
        self._U = torch.nn.ParameterList([
            torch.nn.Parameter((factors.U.abs() if self.cfg.non_negative else factors.U).clone())
            for factors in initial
        ])
        self._S_n = torch.nn.ParameterList([
            torch.nn.Parameter((factors.S.abs() if self.cfg.non_negative else factors.S).clone())
            for factors in initial
        ])
        self.Q_n = []
        for n, factors in enumerate(initial):
            self.register_buffer(f"_Q_{n}", factors.Q)
            self.Q_n.append(getattr(self, f"_Q_{n}"))

    @property
    def A(self):
        return _FactorAccessor(self._A, transform=_positive if self.cfg.non_negative else None)

    @property
    def U(self):
        return _FactorAccessor(self._U, transform=_positive if self.cfg.non_negative else None)

    @property
    def S_n(self):
        return _FactorAccessor(self._S_n, transform=_positive if self.cfg.non_negative else None)

    @torch.no_grad()
    def finalize(self):
        if self.label_info and hasattr(self, "C"):
            coordinate_update = ConditionProcrustesLoss(k=1)
        else:
            coordinate_update = GlobalProcrustesLoss(k=1)
        coordinate_update.update_projections(self)

    def export_factors(self, history=None, loss_history=None):
        factors = [
            TensorFactors(
                A=self.A[n].detach().clone(),
                U=self.U[n].detach().clone(),
                Q=self.Q_n[n].detach().clone(),
                S=self.S_n[n].detach().clone(),
            )
            for n in range(len(self.tensors))
        ]
        return _AdaptationResult(
            factors,
            list(history or []),
            {name: list(values) for name, values in (loss_history or {}).items()},
        )


def partial_fit_factors(
    source: NeuralignFactors,
    tensors,
    trial_labels=None,
    *,
    device=None,
    config=None,
    **overrides,
) -> NeuralignFactors:
    """Return a copy of ``source`` with one or more fitted sessions appended."""
    cfg = _adaptation_config(source, config, overrides)
    source_device = source.w.device
    device = source_device if device is None else torch.device(device)
    tensors = [
        torch.as_tensor(tensor, dtype=source.w.dtype, device=device)
        for tensor in _as_sequence(tensors)
    ]

    if not tensors or any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("Expected one or more (neurons, time, trials) tensors.")
    if any(tensor.shape[1] != source.V.shape[0] for tensor in tensors):
        raise ValueError("New tensors must use the learned time axis.")

    labels = _label_sequence(trial_labels, len(tensors))
    vocabulary = _condition_vocabulary(source)
    label_info, ignored = _label_info(tensors, labels, vocabulary)
    if ignored:
        warnings.warn(
            f"Ignoring labels without frozen condition prototypes: {ignored!r}",
            UserWarning,
            stacklevel=2,
        )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    model = _AdaptationModel(source, tensors, cfg, label_info)
    callbacks = [EarlyStopping(cfg.patience, cfg.tol)]
    if cfg.progress:
        callbacks.append(ProgressBar())
    trainer = Trainer(
        model,
        _adaptation_losses(model, cfg),
        lr=cfg.lr,
        max_steps=cfg.max_iter,
        callbacks=callbacks,
    )
    fitted = trainer.fit()

    result = copy.deepcopy(source)
    new_factors = [factors.to(source_device) for factors in fitted.factors]
    result.tensor_factors.extend(new_factors)
    if result.condition_labels is None and source.C is not None:
        result.condition_labels = vocabulary

    result.metadata = result.metadata or {}
    result.metadata.setdefault("adaptations", []).append({
        "session_indices": list(range(len(source), len(result))),
        "loss": fitted.history,
        "losses": fitted.loss_history,
        "ignored_labels": ignored,
        "config": to_dict(cfg),
    })
    return result


def _initialize_tensor(tensor, w, V, H):
    S = torch.ones_like(w)
    template = H @ torch.diag(w) @ V.T
    Q = ops.solve_procrustes(tensor.mean(dim=2), template)
    A = ops.reconstruct_A(Q, H, S)
    U = initialize_trial_factors(tensor, w, A, V)
    return TensorFactors(A=A, U=U, Q=Q, S=S)


class _FactorAccessor:
    def __init__(self, raw, transform=None):
        self.raw = raw
        self.transform = transform

    def __getitem__(self, idx):
        value = self.raw[idx]
        return self.transform(value) if self.transform is not None else value

    def __len__(self):
        return len(self.raw)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _positive(x):
    return F.softplus(x) + 1e-12


def _adaptation_config(source, config, overrides):
    base_config = source.config if config is None else config
    if base_config is None:
        base_config = NeuralignConfig(rank=len(source.w))
    if isinstance(base_config, dict):
        config = from_dict(base_config)
    else:
        config = copy.deepcopy(base_config)
    config = override_config(config, overrides)
    if config.rank != len(source.w):
        raise ValueError("Adaptation rank must match the learned rank.")
    config.gauge_fix = False
    return config


def _adaptation_losses(model, cfg):
    losses = {}
    if cfg.lambda_tensor_reconstruction > 0:
        norms = [torch.linalg.norm(tensor) ** 2 for tensor in model.tensors]
        losses["Tensor"] = (
            TensorReconstructionLoss(norms), cfg.lambda_tensor_reconstruction
        )
    if cfg.lambda_global_gram > 0:
        losses["Geometry"] = (GlobalGramLoss(), cfg.lambda_global_gram)
    if cfg.lambda_procrustes > 0:
        if model.label_info and hasattr(model, "C"):
            coordinate_loss = ConditionProcrustesLoss(cfg.procrustes_every_k)
        else:
            coordinate_loss = GlobalProcrustesLoss(cfg.procrustes_every_k)
        losses["Coordinates"] = (
            coordinate_loss,
            cfg.lambda_procrustes,
        )
    if cfg.lambda_neuron_sparsity > 0:
        losses["Sparsity"] = (NeuronSparsityLoss(), cfg.lambda_neuron_sparsity)
    return losses


def _as_sequence(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return [value]
    if isinstance(value, (list, tuple)) and value and isinstance(
        value[0], (np.ndarray, torch.Tensor)
    ):
        return list(value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        array = None
    return [value] if array is not None and array.ndim == 3 else list(value)


def _label_sequence(labels, n_tensors):
    if labels is None:
        return [None] * n_tensors
    if n_tensors == 1:
        if len(labels) == 1 and isinstance(
            labels[0], (list, tuple, np.ndarray, torch.Tensor)
        ):
            return list(labels)
        return [labels]
    if len(labels) != n_tensors:
        raise ValueError("Provide one label sequence for each new tensor.")
    return list(labels)


def _condition_vocabulary(source):
    if source.C is None:
        return []
    if source.condition_labels is None:
        return list(range(source.C.shape[0]))
    return list(source.condition_labels)


def _label_info(tensors, labels, vocabulary):
    label_to_index = {label: n for n, label in enumerate(vocabulary)}
    label_info, ignored = [], []
    for n, (tensor, tensor_labels) in enumerate(zip(tensors, labels)):
        if tensor_labels is None:
            continue
        tensor_labels = np.asarray(tensor_labels)
        if len(tensor_labels) != tensor.shape[2]:
            raise ValueError("Trial labels must match the number of trials.")
        for label in np.unique(tensor_labels):
            label = label.item()
            if label not in label_to_index:
                if label not in ignored:
                    ignored.append(label)
                continue
            indices = torch.as_tensor(
                np.flatnonzero(tensor_labels == label),
                dtype=torch.long,
                device=tensor.device,
            )
            label_info.append((n, label_to_index[label], indices))
    return label_info, ignored
