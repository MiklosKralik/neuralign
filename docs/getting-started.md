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

## Adapting to held-out sessions

Use `partial_fit` on fitted factors to estimate only the session-specific
factors for new recordings. The shared weights, temporal factors, geometry,
and optional condition prototypes remain frozen. The method returns a copy;
it does not modify the original fit.

```python
adapted = factors.partial_fit(
    held_out_tensor,
    trial_labels=held_out_labels,
    device="cpu",
    max_iter=500,
)

held_out_factors = adapted[-1]
adaptation_history = adapted.metadata["adaptations"][-1]
```

A sequence of tensors can be adapted together by supplying one label sequence
per tensor. Labels matching the fitted condition vocabulary are used for
condition-aware alignment. Unknown labels produce a warning and are excluded;
a session with no recognized labels falls back to trial-averaged alignment.
All new tensors must share the fitted time axis and contain at least as many
neurons as the fitted rank.
