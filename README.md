# neuralign

Neuralign provides tensor decomposition methods for identifying shared neural
representations across recording sessions. The package jointly models a
sequence of three-dimensional neural data tensors while allowing each session
to retain session-specific neuron and trial factors.

> Neuralign is currently an early research package accompanying a paper in
> preparation. Its API may change before the first public release.

## Installation

Until the package is published, install it directly from GitHub:

```bash
pip install "neuralign @ git+https://github.com/MiklosKralik/neuralign.git"
```

For development, clone the repository and install it editably:

```bash
git clone https://github.com/MiklosKralik/neuralign.git
cd neuralign
pip install -e .
```

## Quick start

Neuralign expects one tensor per recording session. Each tensor must use the
axis order `(neurons, time, trials)` and all sessions must have the same number
of time samples.

```python
import torch

from neuralign import NeuralignConfig, NeuralignModel

sessions = [
    torch.randn(40, 100, 60),
    torch.randn(52, 100, 75),
]

config = NeuralignConfig(rank=4, max_iter=500, progress=True)
model = NeuralignModel(sessions, config=config)
factors = model.fit("cpu")
```

## Documentation website

A basic GitHub Pages site is available at:

- https://mikloskralik.github.io/neuralign/

It includes:

- [Getting Started](https://mikloskralik.github.io/neuralign/getting-started.html)
- [API Overview](https://mikloskralik.github.io/neuralign/api.html)

Local documentation sources remain in [`docs/`](docs/).
