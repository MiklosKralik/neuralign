# Getting started

## Install

```bash
pip install "neuralign[examples] @ git+https://github.com/MiklosKralik/neuralign.git"
```

Python 3.10 or later is required. The `examples` extra installs Matplotlib for plotting; the model itself requires NumPy and PyTorch. For a local checkout:

```bash
uv sync --extra examples
uv run python examples/quickstart.py
```

The example saves `examples/quickstart.png` and displays the figure. Use `MPLBACKEND=Agg` when running without a display.

## Prepare your recordings

Provide a list of floating-point arrays or tensors, each shaped **neurons × time × trials**. Neuron and trial counts may differ between sessions. Time bins must match, and the rank cannot exceed any session's neuron count. Choose preprocessing for your data; fit its statistics on training data when evaluating held-out observations.

## Fit and inspect

```python
from neuralign import NeuralignConfig, NeuralignModel
from neuralign.simulation import simulate_sessions

sessions, truth = simulate_sessions(seed=42)
factors = NeuralignModel(
    sessions,
    config=NeuralignConfig(rank=2, max_iter=500, seed=42, progress=False),
).fit("cpu")

session = factors.session(0)
print(factors.V.shape)                   # time, components
print(session.neuron_factors.shape)      # neurons, components
print(session.trial_factors.shape)       # trials, components
reconstructed = session.reconstructed_tensor()
```

Use `fit("cuda")` with a compatible CUDA-enabled PyTorch installation.

## Complete plotting example

[Download quickstart.py](assets/quickstart.py), or run it from the package's `examples` folder. Its small simulator uses the sigmoid/transient profiles and changing trial factors from the manuscript's identifiability example, with light Gaussian noise. It is a teaching example, not the full Poisson benchmark. Component order, sign and scale can differ between the generating and fitted factors.

{{example}}

## Optional condition labels

```python
factors = NeuralignModel(
    sessions, trial_labels=labels_by_session,
    config=NeuralignConfig(rank=2, seed=42),
).fit("cpu")
```

Supply one label per trial in each session. Labels orient alignment using condition means; they are not reconstruction targets. See the [API reference](api.html) for masks and adding new sessions.
