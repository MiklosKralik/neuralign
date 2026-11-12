"""Neuralign fitting options."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class NeuralignConfig:
    """Configuration for fitting and adapting session tensors."""

    rank: int = 4
    init: str = "svd"
    gauge_fix: bool = True
    non_negative: bool = False
    loss_type: str = "frobenius"  # "frobenius" or "poisson" or "poisson_log"
    lr: float = 1e-2
    max_iter: int = 2000
    tol: float = 1e-6
    patience: int = 100
    progress: bool = True
    seed: int | None = None

    lambda_tensor_reconstruction: float = 1.0
    lambda_procrustes: float = 1.0 # coordinate alignment loss (can be condition-aware)
    # Adaptation penalty toward frozen condition means: None uses the default;
    # 0.0 disables it. Only recognized, unmasked conditions contribute.
    lambda_condition_centroid: float | None = None
    procrustes_every_k: int = 3 # how often to re-compute aligned coordinates (expensive)
    lambda_global_gram: float = 0.0 # global geometry loss (regularizer)
    lambda_neuron_sparsity: float = 0.0 # weight for the neuron sparsity loss (regularizer)


def to_dict(cfg: NeuralignConfig) -> Dict[str, Any]:
    return asdict(cfg)


def from_dict(d: Dict[str, Any]) -> NeuralignConfig:
    valid = {f.name for f in fields(NeuralignConfig)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return NeuralignConfig(**filtered)


def override_config(cfg: NeuralignConfig, overrides: Dict[str, Any]) -> NeuralignConfig:
    d = to_dict(cfg)
    for k, v in overrides.items():
        if v is None:
            continue
        if k in d:
            d[k] = v
        else:
            raise ValueError(f"Invalid config key: {k}")
    return from_dict(d)
