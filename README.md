# Neuralign

Jointly align and decompose neural recordings with different neurons and trial counts. Neuralign learns shared temporal components while keeping a factor for each recorded neuron and each trial.

## Install

```bash
pip install "neuralign[examples] @ git+https://github.com/MiklosKralik/neuralign.git"
```

Requires Python 3.10 or later. The optional `examples` extra adds Matplotlib.

## Fit a small simulation

```python
from neuralign import NeuralignConfig, NeuralignModel
from neuralign.simulation import simulate_sessions

sessions, truth = simulate_sessions(seed=42)
factors = NeuralignModel(
    sessions,
    config=NeuralignConfig(rank=2, max_iter=500, seed=42, progress=False),
).fit("cpu")

session = factors[0]
time_factors = factors.V                       # time × components
trial_factors = session.trial_factors            # trials × components
reconstructed = session.reconstructed_tensor()  # neurons × time × trials
```

Inputs use **neurons × time × trials**. Sessions may differ in neuron and trial counts, but their time bins must match. Use `fit("cuda")` for GPU fitting with a compatible PyTorch installation.

The [complete example](examples/quickstart.py) simulates its own data, fits the model and plots factors and reconstructions:

```bash
uv sync --extra examples
uv run python examples/quickstart.py
```

![Example fit](docs/assets/quickstart.png)

## Common tasks

```python
# Orient alignment with one label per trial in each session.
factors = NeuralignModel(sessions, trial_labels=labels_by_session, rank=2).fit("cpu")

# Compare measured activity with the model prediction.
measured = factors[0].projected_trajectories(sessions[0])
predicted = factors[0].latent_trajectories()

# Reconstruct one component; indices are zero-based.
component = factors[0].reconstructed_latent_tensor(components=0)
unit_weights = factors[0].reconstructed_latent_tensor(components=0, normalized=True)

# Save a fit or add another session with shared factors frozen.
factors.save("fit.pt")
adapted = factors.partial_fit(new_session, device="cpu")
```

Projection and reconstruction describe different quantities. `normalized=True` replaces global weights with ones; it does not normalise the output. [Read about the outputs](docs/model-and-outputs.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [Model and outputs](docs/model-and-outputs.md)
- [API reference](docs/api.md)
- [Reproducibility](docs/reproducibility.md)
- [Documentation website](https://mikloskralik.github.io/neuralign/)

The [companion repository](https://github.com/MiklosKralik/neural_tensors) contains the manuscript analyses and figure notebooks. This package's example needs no external recordings or caches.

For a local documentation preview:

```bash
uv sync --extra docs
uv run python docs/build.py
python -m http.server 8765 --directory docs
```

Neuralign is research software (version 0.1.0). Record the version used in an analysis. Citation metadata are in [CITATION.cff](CITATION.cff); manuscript and archive identifiers can be added when available.
