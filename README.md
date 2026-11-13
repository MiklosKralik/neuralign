# <img src="https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-mark.png" width="36" alt="" /> Neuralign

Find shared activity patterns across neural recordings, with a factor for each neuron and each trial.

[![Neuralign model overview](https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-overview.png)](https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/neuralign-overview.pdf)

## Install

Python 3.10 or later. Install from [PyPI](https://pypi.org/project/neuralign/).

For CPU-only use, install CPU PyTorch first, then Neuralign:

```bash
python -m pip install "torch>=2.9.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install neuralign
```

For CUDA on Linux with a compatible NVIDIA GPU and driver (recommended for larger datasets):

```bash
python -m pip install "neuralign[cuda]"
```

PyTorch's default PyPI wheels include CUDA dependencies on Linux, so plain `pip install neuralign` also downloads them unless CPU PyTorch is already installed. The `cuda` extra uses the same PyPI PyTorch dependency; it does not replace an existing CPU build. To switch builds, or install CUDA on Windows, follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/). CUDA is unavailable on macOS.

Or install Neuralign directly from GitHub after installing your preferred PyTorch build:

```bash
python -m pip install git+https://github.com/MiklosKralik/neuralign.git
```

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

[Documentation](https://mikloskralik.github.io/neuralign/) · [Use your recordings](https://mikloskralik.github.io/neuralign/getting-started.html) · [Plotting example](https://github.com/MiklosKralik/neuralign/blob/main/examples/quickstart.py)

## Paper

Neuralign is described in [*Identifying aligned tensor decompositions across heterogeneous neural datasets*](https://doi.org/10.64898/2026.11.12.766278) (bioRxiv, 2026).

Please cite the software and manuscript when using Neuralign in research. [Citation](https://github.com/MiklosKralik/neuralign/blob/main/CITATION.cff) · [MIT License](https://github.com/MiklosKralik/neuralign/blob/main/LICENSE).
