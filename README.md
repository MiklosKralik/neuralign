# <img src="https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-mark.png" width="36" alt="" /> Neuralign

Find shared activity patterns across neural recordings, with a factor for each neuron and each trial.

[![Neuralign model overview](https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-overview.png)](https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-overview.pdf)

## Quickstart

This example generates its own data:

```python
from neuralign import NeuralignModel
from neuralign.simulation import simulate_sessions

sessions, _ = simulate_sessions(seed=42)
factors = NeuralignModel(sessions, rank=2, max_iter=500, seed=42).fit("cpu")

time_factors = factors.V                       # time × components
neuron_factors = factors[0].neuron_factors       # neurons × components
trial_factors = factors[0].trial_factors         # trials × components
```

For your own recordings, pass a list of arrays in **neurons × time × trials** order. Sessions may have different neuron and trial counts, but must share time bins.

[Documentation](https://mikloskralik.github.io/neuralign/) · [Use your recordings](https://mikloskralik.github.io/neuralign/getting-started.html) · [Plotting example](https://mikloskralik.github.io/neuralign/plotting.html)

## Install

Python 3.10+; GPU fitting is recommended for larger datasets (use `fit("cuda")`).

```bash
python -m pip install neuralign
```

For CPU-only installation, install CPU PyTorch first:

```bash
python -m pip install "torch>=2.9.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install neuralign
```

## Paper

Neuralign is described in [*Identifying aligned tensor decompositions across heterogeneous neural datasets*](https://doi.org/10.64898/2026.11.12.766278) (bioRxiv, 2026).

Please cite the software and manuscript when using Neuralign in research. [Citation](https://github.com/MiklosKralik/neuralign/blob/main/CITATION.cff) · [MIT License](https://github.com/MiklosKralik/neuralign/blob/main/LICENSE).
