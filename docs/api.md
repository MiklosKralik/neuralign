# API overview

The stable top-level imports currently exposed by `neuralign` are:

```python
from neuralign import (
    NeuralignConfig,
    NeuralignFactors,
    NeuralignModel,
    TensorFactors,
)
```

## `NeuralignModel`

Constructs and fits a joint decomposition from a sequence of session tensors.
Use `fit(device)` to optimize the model and return exported factors.

## `NeuralignConfig`

A dataclass containing model rank, optimizer settings, loss weights,
non-negativity settings, progress behavior, and reproducibility controls.

## `NeuralignFactors`

Stores the global weights, temporal factor, shared geometry, optional condition
centroids, configuration, metadata, and all session factors. It supports
indexing, persistence, and fitting additional sessions against learned global
factors.

## `TensorFactors`

Stores the neuron, trial, projection, and scaling factors for one recording
session and provides access to that session's reconstructed tensor.
