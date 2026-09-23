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

`NeuralignModel(tensor_sequence, trial_labels=None, trial_mask=None,
config=None, **overrides)` accepts one Boolean trial mask per session. `True`
trials are used by reconstruction but excluded from unsupervised and
condition-aware alignment. In condition-aware fits they are also excluded from
the learned condition vocabulary and centroids. At least one unmasked trial is
required per session.

## `NeuralignConfig`

A dataclass containing model rank, optimizer settings, loss weights,
non-negativity settings, progress behavior, and reproducibility controls.

## `NeuralignFactors`

Stores the global weights, temporal factor, shared geometry, optional condition
centroids, configuration, metadata, and all session factors. It supports
indexing, persistence, and fitting additional sessions against learned global
factors.

`partial_fit(tensor, trial_labels=None, trial_mask=None, *, device=None,
config=None, **overrides)` returns a cloned factor collection with one or more
held-out sessions appended. It optimizes only the new sessions' neuron, trial,
scaling, and projection factors. `True` mask entries exclude trials from
alignment while retaining them for reconstruction. Masked labels are not used
as anchors and are not reported as unknown labels. Shared factors and condition
prototypes remain frozen; adaptation histories are appended to
`metadata["adaptations"]`.

During supervised `partial_fit`, `lambda_condition_centroid=None` resolves to a
small default weight of `1e-2`. The regularizer penalizes squared distance
between each new session's unmasked condition-mean trial factor and the
corresponding frozen source condition prototype. Pass `0.0` to disable it or a
positive value to override its strength. Explicitly enabling it requires at
least one recognized, unmasked trial label. Its history is recorded under the
`Conditions` loss name.

## `TensorFactors`

Stores the neuron, trial, projection, and scaling factors for one recording
session and provides access to that session's reconstructed tensor.
