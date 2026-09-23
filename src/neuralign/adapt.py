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
    AdaptationConditionCentroidLoss,
    AdaptationProcrustesLoss,
    GlobalGramLoss,
    NeuronSparsityLoss,
    TensorReconstructionLoss,
)
from neuralign.neuralign_factors import NeuralignFactors, TensorFactors
from neuralign.trainer import Trainer

DEFAULT_CONDITION_CENTROID_WEIGHT = 1e-2

@dataclass
class _AdaptationResult:
    factors: list[TensorFactors]
    history: list[float]
    loss_history: dict[str, list[float]]


class _AdaptationModel(torch.nn.Module):
    """Neuralign model with fixed global factors and new local factors."""

    def __init__(
        self, source, tensors, config, session_label_info, trial_masks
    ):
        super().__init__()
        self.cfg = config
        self._tensor_names = []
        for n, tensor in enumerate(tensors):
            name = f"_tensor_{n}"
            self.register_buffer(name, tensor, persistent=False)
            self._tensor_names.append(name)
        self.tensors = [getattr(self, name) for name in self._tensor_names]

        self._trial_mask_names = []
        self._alignment_index_names = []
        for n, mask in enumerate(trial_masks):
            mask_name = f"_trial_mask_{n}"
            index_name = f"_alignment_indices_{n}"
            self.register_buffer(mask_name, mask, persistent=False)
            self.register_buffer(
                index_name,
                torch.nonzero(~mask, as_tuple=True)[0],
                persistent=False,
            )
            self._trial_mask_names.append(mask_name)
            self._alignment_index_names.append(index_name)
        self.trial_masks = [
            getattr(self, name) for name in self._trial_mask_names
        ]
        self.alignment_indices = [
            getattr(self, name) for name in self._alignment_index_names
        ]

        for name in ("w", "V", "H"):
            value = getattr(source, name).detach().clone().to(tensors[0].device)
            self.register_buffer(name, value)
        if source.C is not None:
            self.register_buffer(
                "C", source.C.detach().clone().to(tensors[0].device)
            )

        initial = [
            _initialize_tensor(
                tensor,
                self.w,
                self.V,
                self.H,
                alignment_indices=self.alignment_indices[n],
                init=self.cfg.init,
            )
            for n, tensor in enumerate(tensors)
        ]
        self._A = torch.nn.ParameterList([
            torch.nn.Parameter(_parameter_value(factors.A, self.cfg.non_negative))
            for factors in initial
        ])
        self._U = torch.nn.ParameterList([
            torch.nn.Parameter(_parameter_value(factors.U, self.cfg.non_negative))
            for factors in initial
        ])
        self._S_n = torch.nn.ParameterList([
            torch.nn.Parameter(_parameter_value(factors.S, self.cfg.non_negative))
            for factors in initial
        ])
        self._Q_names = []
        for n, factors in enumerate(initial):
            name = f"_Q_{n}"
            self.register_buffer(name, factors.Q.detach().clone())
            self._Q_names.append(name)
        self.Q_n = [getattr(self, name) for name in self._Q_names]
        self.coordinate_loss = AdaptationProcrustesLoss(
            self, session_label_info, k=self.cfg.procrustes_every_k
        )

    @property
    def A(self):
        return _FactorAccessor(
            self._A, transform=_positive if self.cfg.non_negative else None
        )

    @property
    def U(self):
        return _FactorAccessor(
            self._U, transform=_positive if self.cfg.non_negative else None
        )

    @property
    def S_n(self):
        return _FactorAccessor(
            self._S_n, transform=_positive if self.cfg.non_negative else None
        )

    @torch.no_grad()
    def finalize(self, trainer):
        del trainer
        self.coordinate_loss.update_projections(self)

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
    trial_mask=None,
    *,
    device=None,
    config=None,
    **overrides,
) -> NeuralignFactors:
    """Return a copy of source with one or more fitted sessions appended."""
    cfg = _adaptation_config(source, config, overrides)
    source_device = source.w.device
    device = source_device if device is None else torch.device(device)

    tensors = [
        torch.as_tensor(value, dtype=source.w.dtype, device=device).detach()
        for value in _as_sequence(tensors)
    ]
    if not tensors:
        raise ValueError("Expected one or more (neurons, time, trials) tensors.")
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("Tensors must have shape (neurons, time, trials).")
    if any(tensor.shape[1] != source.V.shape[0] for tensor in tensors):
        raise ValueError("New tensors must use the learned time axis.")

    labels = _session_sequence(trial_labels, len(tensors), "label")
    masks = _trial_masks(tensors, trial_mask)
    vocabulary = _condition_vocabulary(source)
    session_label_info, ignored = _label_info(
        tensors, labels, masks, vocabulary
    )
    if cfg.lambda_condition_centroid is None:
        cfg.lambda_condition_centroid = (
            DEFAULT_CONDITION_CENTROID_WEIGHT
            if any(session_label_info)
            else 0.0
        )
    if ignored:
        warnings.warn(
            f"Ignoring labels without frozen condition prototypes: {ignored!r}",
            UserWarning,
            stacklevel=2,
        )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    model = _AdaptationModel(
        source, tensors, cfg, session_label_info, masks
    )
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
    result.tensor_factors = list(result.tensor_factors)
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


def _initialize_tensor(
    tensor, w, V, H, alignment_indices, init="svd"
):
    S = torch.ones_like(w)
    if init == "svd":
        template = H @ torch.diag(w) @ V.T
        tensor_slice = tensor.index_select(2, alignment_indices).mean(dim=2)
        Q = ops.solve_procrustes(tensor_slice, template)
    elif init == "random":
        random = torch.randn(
            tensor.shape[0], len(w), dtype=tensor.dtype, device=tensor.device
        )
        Q = torch.linalg.qr(random, mode="reduced").Q
    else:
        raise ValueError(f"Invalid initialization method: {init}")
    A = ops.reconstruct_A(Q, H, S)
    if init == "random":
        U = torch.randn(
            tensor.shape[2], len(w), dtype=tensor.dtype, device=tensor.device
        )
    else:
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


def _inverse_positive(value):
    target = value.abs().clamp_min(torch.finfo(value.dtype).eps)
    shifted = (target - 1e-12).clamp_min(torch.finfo(value.dtype).tiny)
    return shifted + torch.log(-torch.expm1(-shifted))


def _parameter_value(value, non_negative):
    value = value.detach().clone()
    return _inverse_positive(value) if non_negative else value


def _adaptation_config(source, config, overrides):
    base_config = source.config if config is None else config
    if base_config is None:
        base_config = NeuralignConfig(rank=len(source.w))
    if isinstance(base_config, dict):
        cfg = from_dict(base_config)
    elif isinstance(base_config, NeuralignConfig):
        cfg = copy.deepcopy(base_config)
    else:
        raise TypeError("config must be a NeuralignConfig, a dictionary, or None.")
    cfg = override_config(cfg, overrides)
    cfg.rank = len(source.w)
    cfg.gauge_fix = False
    return cfg


def _adaptation_losses(model, cfg):
    losses = {}
    if cfg.lambda_tensor_reconstruction > 0:
        floor = torch.finfo(model.tensors[0].dtype).tiny
        norms = [
            (torch.linalg.norm(tensor) ** 2).clamp_min(floor)
            for tensor in model.tensors
        ]
        losses["Tensor"] = (
            TensorReconstructionLoss(norms, cfg.loss_type),
            cfg.lambda_tensor_reconstruction,
        )
    if cfg.lambda_global_gram > 0:
        losses["Geometry"] = (GlobalGramLoss(), cfg.lambda_global_gram)
    if cfg.lambda_procrustes > 0:
        losses["Coordinates"] = (
            model.coordinate_loss, cfg.lambda_procrustes
        )
    if cfg.lambda_condition_centroid > 0:
        condition_loss = AdaptationConditionCentroidLoss(
            model.coordinate_loss.session_label_info
        )
        if not hasattr(model, "C") or not condition_loss.entries:
            raise ValueError(
                "lambda_condition_centroid requires frozen source condition "
                "prototypes and at least one recognized, unmasked trial label."
            )
        losses["Conditions"] = (
            condition_loss, cfg.lambda_condition_centroid
        )
    if cfg.lambda_neuron_sparsity > 0:
        losses["Sparsity"] = (
            NeuronSparsityLoss(), cfg.lambda_neuron_sparsity
        )
    return losses


def _as_sequence(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return [value]
    try:
        array = np.asarray(value)
    except ValueError:
        return list(value)
    if array.ndim == 3:
        return [value]
    return list(value)


def _session_sequence(values, n_tensors, name):
    if values is None:
        return [None] * n_tensors
    if n_tensors == 1:
        if len(values) == 1 and isinstance(
            values[0], (list, tuple, np.ndarray, torch.Tensor)
        ):
            return [values[0]]
        return [values]
    if len(values) != n_tensors:
        raise ValueError(f"Provide one {name} sequence for each new tensor.")
    return list(values)


def _trial_masks(tensors, trial_mask):
    values = _session_sequence(trial_mask, len(tensors), "trial-mask")
    masks = []
    for tensor, value in zip(tensors, values):
        if value is None:
            mask = torch.zeros(
                tensor.shape[2], dtype=torch.bool, device=tensor.device
            )
        else:
            mask = torch.as_tensor(
                value, dtype=torch.bool, device=tensor.device
            )
        if mask.ndim != 1 or len(mask) != tensor.shape[2]:
            raise ValueError(
                "Trial masks must match the number of trials in each tensor."
            )
        if mask.all():
            raise ValueError("Each tensor must contain an unmasked trial.")
        masks.append(mask)
    return masks


def _condition_vocabulary(source):
    if source.C is None:
        return []
    if source.condition_labels is None:
        return list(range(source.C.shape[0]))
    return [_python_scalar(label) for label in source.condition_labels]


def _label_info(tensors, labels, masks, vocabulary):
    label_to_index = {label: n for n, label in enumerate(vocabulary)}
    session_label_info = []
    ignored = []
    ignored_set = set()
    for n, (tensor, tensor_labels, mask) in enumerate(
        zip(tensors, labels, masks)
    ):
        if tensor_labels is None:
            session_label_info.append([])
            continue
        values = (
            tensor_labels.detach().cpu().numpy()
            if isinstance(tensor_labels, torch.Tensor)
            else np.asarray(tensor_labels)
        )
        if values.ndim != 1 or len(values) != tensor.shape[2]:
            raise ValueError(
                f"Trial labels for tensor {n} must match its number of trials."
            )
        positions = {}
        alignment_indices = torch.nonzero(~mask, as_tuple=True)[0].tolist()
        for index in alignment_indices:
            label = values[index]
            label = _python_scalar(label)
            condition = label_to_index.get(label)
            if condition is None:
                if label not in ignored_set:
                    ignored_set.add(label)
                    ignored.append(label)
                continue
            positions.setdefault(condition, []).append(index)
        condition_info = [
            (
                condition,
                torch.tensor(indices, dtype=torch.long, device=tensor.device),
            )
            for condition, indices in positions.items()
        ]
        session_label_info.append(condition_info)
    return session_label_info, ignored


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value
