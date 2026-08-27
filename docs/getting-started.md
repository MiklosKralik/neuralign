# Getting started

## Input data

Pass `NeuralignModel` a sequence containing one NumPy array, PyTorch tensor, or
array-like object per session. Each input must have shape
`(neurons, time, trials)`. Sessions may contain different numbers of neurons
and trials, but their time dimension must match.

Optional trial labels can be supplied alongside the tensors. Provide one label
sequence per session, with one label for each trial:

```python
from neuralign import NeuralignConfig, NeuralignModel

model = NeuralignModel(
    session_tensors,
    trial_labels=session_labels,
    config=NeuralignConfig(rank=6, seed=42),
)
factors = model.fit("cpu")
```

Pass `"cuda"` to `fit` when a compatible PyTorch CUDA installation is
available.

## Configuration

`NeuralignConfig` controls rank, initialization, optimization, loss selection,
regularization, non-negativity, progress display, and random seeding. Keyword
overrides can also be passed to `NeuralignModel`; invalid configuration names
raise `ValueError`.

The default loss uses Frobenius reconstruction error. Count-valued data can use
the supported Poisson modes through `loss_type`, optionally together with
`non_negative=True`.

## Results

`fit` returns a `NeuralignFactors` object containing shared factors and one
`TensorFactors` object per input session. Index the result to access a session,
or use its save and adaptation methods to persist a fit or add held-out
sessions.
