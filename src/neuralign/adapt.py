"""Fit new sessions into an existing Neuralign decomposition."""

import copy
import math
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
    GlobalGramLoss,
    NeuronSparsityLoss,
    TensorReconstructionLoss,
    relative_fro_error_sq,
)
from neuralign.neuralign_factors import NeuralignFactors, TensorFactors
from neuralign.trainer import Trainer


_POSITIVE_FLOOR = 1e-12
_LOSS_WEIGHT_NAMES = (
    "lambda_tensor_reconstruction",
    "lambda_global_gram",
    "lambda_procrustes",
    "lambda_neuron_sparsity",
)


@dataclass
class _AdaptationResult:
    factors: list[TensorFactors]
    history: list[float]
    loss_history: dict[str, list[float]]


class _AdaptationProcrustesLoss(torch.nn.Module):
    """Align new sessions without changing learned condition prototypes."""

    def __init__(self, model, session_label_info, k=1, eps=1e-12):
        super().__init__()
        self.k = k
        self.iter = 0
        self.eps = eps
        self.condition_indices = []

        for n, condition_info in enumerate(session_label_info):
            condition_ids = tuple(cond for cond, _ in condition_info)
            self.condition_indices.append(condition_ids)
            if condition_info:
                blocks = [
                    model.tensors[n].index_select(2, indices).mean(dim=2)
                    for _, indices in condition_info
                ]
                observed = torch.cat(blocks, dim=1)
            else:
                observed = model.tensors[n].mean(dim=2)
            self.register_buffer(
                f"Observed_{n}", observed.detach().clone(), persistent=False
            )

    def _predicted_coordinates(self, model, n):
        condition_ids = self.condition_indices[n]
        if condition_ids:
            blocks = [
                ops.predicted_latent_trajectories(
                    model.w,
                    model.H,
                    model.S_n[n],
                    model.V,
                    model.C[cond].unsqueeze(0),
                ).squeeze(0)
                for cond in condition_ids
            ]
            return torch.cat(blocks, dim=0)
        return ops.predicted_latent_trajectories(
            model.w,
            model.H,
            model.S_n[n],
            model.V,
            model.U[n].mean(dim=0, keepdim=True),
        ).squeeze(0)

    @torch.no_grad()
    def update_projections(self, model):
        for n in range(len(model.tensors)):
            observed = getattr(self, f"Observed_{n}")
            predicted = self._predicted_coordinates(model, n)
            model.Q_n[n].copy_(ops.solve_procrustes(observed, predicted.T))

    def forward(self, model):
        if self.iter % self.k == 0:
            self.update_projections(model)
        losses = []
        for n in range(len(model.tensors)):
            aligned = model.Q_n[n] @ ops.scale_mixing(
                model.H, model.S_n[n]
            )
            losses.append(
                relative_fro_error_sq(
                    model.A[n],
                    aligned,
                    torch.linalg.norm(aligned) ** 2 + self.eps,
                )
            )
        self.iter += 1
        return torch.stack(losses).mean()


class _AdaptationModel(torch.nn.Module):
    """Neuralign model with fixed global factors and new local factors."""

    def __init__(self, source, tensors, config, session_label_info):
        super().__init__()
        self.cfg = config
        self._tensor_names = []
        for n, tensor in enumerate(tensors):
            name = f"_tensor_{n}"
            self.register_buffer(name, tensor, persistent=False)
            self._tensor_names.append(name)
        self.tensors = [getattr(self, name) for name in self._tensor_names]

        for name in ("w", "V", "H"):
            value = getattr(source, name).detach().clone().to(tensors[0].device)
            self.register_buffer(name, value)
        if source.C is not None:
            self.register_buffer(
                "C", source.C.detach().clone().to(tensors[0].device)
            )

        initial = [
            _initialize_tensor(
                tensor, self.w, self.V, self.H, init=self.cfg.init
            )
            for tensor in tensors
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
        self.coordinate_loss = _AdaptationProcrustesLoss(
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
    *,
    device=None,
    config=None,
    **overrides,
) -> NeuralignFactors:
    """Return a copy of source with one or more fitted sessions appended."""
    _validate_source(source)
    cfg = _adaptation_config(source, config, overrides)
    source_device = source.w.device
    device = source_device if device is None else torch.device(device)

    tensor_values = _as_sequence(tensors)
    if not tensor_values:
        raise ValueError("Expected one or more (neurons, time, trials) tensors.")
    converted = []
    for n, value in enumerate(tensor_values):
        try:
            tensor = torch.as_tensor(
                value, dtype=source.w.dtype, device=device
            ).detach()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Could not convert tensor {n} to a PyTorch tensor."
            ) from error
        converted.append(tensor)
    tensors = converted
    _validate_tensors(tensors, source, cfg)

    labels = _label_sequence(trial_labels, len(tensors))
    vocabulary = _condition_vocabulary(source)
    session_label_info, ignored = _label_info(tensors, labels, vocabulary)
    if ignored:
        warnings.warn(
            f"Ignoring labels without frozen condition prototypes: {ignored!r}",
            UserWarning,
            stacklevel=2,
        )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    model = _AdaptationModel(source, tensors, cfg, session_label_info)
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
    _validate_history(fitted)

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


def _initialize_tensor(tensor, w, V, H, init="svd"):
    S = torch.ones_like(w)
    if init == "svd":
        template = H @ torch.diag(w) @ V.T
        Q = ops.solve_procrustes(tensor.mean(dim=2), template)
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
    return F.softplus(x) + _POSITIVE_FLOOR


def _inverse_positive(value):
    target = value.abs().clamp_min(torch.finfo(value.dtype).eps)
    shifted = (target - _POSITIVE_FLOOR).clamp_min(torch.finfo(value.dtype).tiny)
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
    _validate_config(cfg, len(source.w))
    cfg.gauge_fix = False
    return cfg


def _validate_config(cfg, rank):
    if cfg.rank != rank:
        raise ValueError("Adaptation rank must match the learned rank.")
    if cfg.init not in {"svd", "random"}:
        raise ValueError("init must be 'svd' or 'random'.")
    if cfg.loss_type not in {"frobenius", "poisson", "poisson_log"}:
        raise ValueError(
            "loss_type must be 'frobenius', 'poisson', or 'poisson_log'."
        )
    if not _is_finite_number(cfg.lr) or cfg.lr <= 0:
        raise ValueError("lr must be a positive finite value.")
    if not isinstance(cfg.max_iter, int) or cfg.max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(cfg.patience, int) or cfg.patience < 1:
        raise ValueError("patience must be a positive integer.")
    if not _is_finite_number(cfg.tol) or cfg.tol < 0:
        raise ValueError("tol must be a finite non-negative value.")
    if not isinstance(cfg.procrustes_every_k, int) or cfg.procrustes_every_k < 1:
        raise ValueError("procrustes_every_k must be a positive integer.")
    if cfg.seed is not None and not isinstance(cfg.seed, int):
        raise ValueError("seed must be an integer or None.")

    weights = [getattr(cfg, name) for name in _LOSS_WEIGHT_NAMES]
    if any(
        not _is_finite_number(weight) or weight < 0
        for weight in weights
    ):
        raise ValueError("Adaptation loss weights must be finite and non-negative.")
    if not any(weight > 0 for weight in weights):
        raise ValueError("At least one adaptation loss weight must be positive.")
    if (
        cfg.lambda_tensor_reconstruction > 0
        and cfg.loss_type == "poisson"
        and not cfg.non_negative
    ):
        raise ValueError("Poisson loss requires non_negative=True.")


def _is_finite_number(value):
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _validate_source(source):
    if not isinstance(source, NeuralignFactors):
        raise TypeError("source must be a NeuralignFactors object.")
    for name in ("w", "V", "H"):
        if not isinstance(getattr(source, name, None), torch.Tensor):
            raise ValueError(f"source.{name} must be a PyTorch tensor.")
    if source.w.ndim != 1 or source.w.numel() == 0:
        raise ValueError("source.w must be a non-empty rank vector.")
    if not source.w.is_floating_point():
        raise ValueError("Source factors must use a floating-point dtype.")
    rank = len(source.w)
    if source.V.ndim != 2 or source.V.shape[1] != rank:
        raise ValueError("source.V must have shape (time, rank).")
    if source.H.shape != (rank, rank):
        raise ValueError("source.H must have shape (rank, rank).")
    for name in ("V", "H"):
        value = getattr(source, name)
        if not value.is_floating_point() or value.dtype != source.w.dtype:
            raise ValueError(
                "Source global factors must share one floating-point dtype."
            )
    if source.tensor_factors is None:
        raise ValueError("source.tensor_factors must be a sequence.")
    if source.metadata is not None and not isinstance(source.metadata, dict):
        raise ValueError("source.metadata must be a dictionary or None.")
    for name in ("w", "V", "H"):
        if not torch.isfinite(getattr(source, name)).all():
            raise ValueError(f"source.{name} contains non-finite values.")
    if source.C is not None:
        if not isinstance(source.C, torch.Tensor):
            raise ValueError("source.C must be a PyTorch tensor or None.")
        if source.C.ndim != 2 or source.C.shape[1] != rank:
            raise ValueError("source.C must have shape (conditions, rank).")
        if not source.C.is_floating_point() or source.C.dtype != source.w.dtype:
            raise ValueError(
                "source.C must use the same floating-point dtype as source.w."
            )
        if not torch.isfinite(source.C).all():
            raise ValueError("source.C contains non-finite values.")
        if (
            source.condition_labels is not None
            and len(source.condition_labels) != source.C.shape[0]
        ):
            raise ValueError("condition_labels must contain one label per row of C.")
    elif source.condition_labels is not None:
        raise ValueError("condition_labels cannot be set when source.C is None.")


def _validate_tensors(tensors, source, cfg):
    rank = len(source.w)
    if cfg.non_negative:
        frozen = [source.w, source.V, source.H]
        if source.C is not None:
            frozen.append(source.C)
        if any(torch.any(value < 0) for value in frozen):
            raise ValueError(
                "non_negative adaptation requires non-negative frozen factors."
            )
    for n, tensor in enumerate(tensors):
        if tensor.ndim != 3:
            raise ValueError(
                f"Tensor {n} must have shape (neurons, time, trials)."
            )
        if any(size == 0 for size in tensor.shape):
            raise ValueError(f"Tensor {n} cannot contain an empty dimension.")
        if tensor.shape[1] != source.V.shape[0]:
            raise ValueError("New tensors must use the learned time axis.")
        if tensor.shape[0] < rank:
            raise ValueError(
                f"Tensor {n} must contain at least rank={rank} neurons."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Tensor {n} contains non-finite values.")
        if cfg.non_negative and torch.any(tensor < 0):
            raise ValueError(
                f"Tensor {n} contains negative values but non_negative is True."
            )
        if (
            cfg.lambda_tensor_reconstruction > 0
            and cfg.loss_type in {"poisson", "poisson_log"}
            and torch.any(tensor < 0)
        ):
            raise ValueError(f"Tensor {n} contains negative Poisson targets.")


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
    if cfg.lambda_neuron_sparsity > 0:
        losses["Sparsity"] = (
            NeuronSparsityLoss(), cfg.lambda_neuron_sparsity
        )
    return losses


def _validate_history(result):
    values = list(result.history)
    for history in result.loss_history.values():
        values.extend(history)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            "Adaptation produced a non-finite loss; no sessions were appended."
        )
    for factors in result.factors:
        if any(
            not torch.isfinite(getattr(factors, name)).all()
            for name in ("A", "U", "Q", "S")
        ):
            raise RuntimeError(
                "Adaptation produced non-finite factors; no sessions were appended."
            )


def _as_sequence(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(
            isinstance(item, (np.ndarray, torch.Tensor)) and item.ndim == 3
            for item in value
        ):
            return list(value)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        return [value] if array is not None and array.ndim == 3 else list(value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim == 3:
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _label_sequence(labels, n_tensors):
    if labels is None:
        return [None] * n_tensors
    if n_tensors == 1:
        if isinstance(labels, (str, bytes)):
            return [labels]
        try:
            if len(labels) == 1 and isinstance(
                labels[0], (list, tuple, np.ndarray, torch.Tensor)
            ):
                return [labels[0]]
        except (TypeError, IndexError):
            pass
        return [labels]
    try:
        valid_length = len(labels) == n_tensors
    except TypeError:
        valid_length = False
    if isinstance(labels, (str, bytes)) or not valid_length:
        raise ValueError("Provide one label sequence for each new tensor.")
    return list(labels)


def _condition_vocabulary(source):
    if source.C is None:
        return []
    if source.condition_labels is None:
        return list(range(source.C.shape[0]))
    vocabulary = [_python_scalar(label) for label in source.condition_labels]
    try:
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("condition_labels must be unique.")
    except TypeError as error:
        raise ValueError("condition_labels must be hashable.") from error
    return vocabulary


def _label_info(tensors, labels, vocabulary):
    label_to_index = {label: n for n, label in enumerate(vocabulary)}
    session_label_info = []
    ignored = []
    ignored_set = set()
    for n, (tensor, tensor_labels) in enumerate(zip(tensors, labels)):
        if tensor_labels is None:
            session_label_info.append([])
            continue
        values = _label_values(tensor_labels, n)
        if len(values) != tensor.shape[2]:
            raise ValueError(
                f"Trial labels for tensor {n} must match its number of trials."
            )
        positions = {}
        for index, label in enumerate(values):
            label = _python_scalar(label)
            try:
                condition = label_to_index.get(label)
            except TypeError as error:
                raise ValueError(
                    "Trial labels must be hashable scalar values."
                ) from error
            if condition is None:
                try:
                    if label not in ignored_set:
                        ignored_set.add(label)
                        ignored.append(label)
                except TypeError as error:
                    raise ValueError(
                        "Trial labels must be hashable scalar values."
                    ) from error
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


def _label_values(labels, tensor_index):
    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1:
            raise ValueError(f"Trial labels for tensor {tensor_index} must be 1D.")
        return labels.detach().cpu().tolist()
    if isinstance(labels, (str, bytes)):
        raise ValueError(
            f"Trial labels for tensor {tensor_index} must be a 1D sequence."
        )
    try:
        array = np.asarray(labels)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Trial labels for tensor {tensor_index} must be a 1D sequence."
        ) from error
    if array.ndim != 1:
        raise ValueError(f"Trial labels for tensor {tensor_index} must be 1D.")
    return array.tolist()


def _python_scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, (float, complex)) and not bool(np.isfinite(value)):
        raise ValueError("Condition and trial labels must be finite.")
    return value
