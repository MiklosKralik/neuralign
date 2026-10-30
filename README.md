# Neuralign

Neuralign jointly aligns and factorizes trial-resolved neural recordings with non-matching neurons and trial counts. It combines a CP tensor reconstruction with a soft PARAFAC2-inspired alignment, producing shared temporal components and session-specific neuron factors, trial factors, component gains, and orthonormal neural projections.

Use Neuralign when the analysis needs all three of the following:

- correspondence across recordings without cell registration;
- explicit single-trial component expression;
- a linear map back to the neurons recorded in each session.

The package is research software accompanying the manuscript *Neuralign jointly aligns trial-resolved neural dynamics across recordings*. The API is usable and tested, but the project has not yet reached a stable 1.0 release.

## Installation

Install from GitHub:

```bash
pip install "neuralign @ git+https://github.com/MiklosKralik/neuralign.git"
```

For development with the locked environment:

```bash
git clone https://github.com/MiklosKralik/neuralign.git
cd neuralign
uv sync --dev
uv run pytest -q
```

Neuralign requires Python 3.10 or newer. Install a CUDA-enabled PyTorch build appropriate for the local driver before installing Neuralign when GPU fitting is required.

## Quick start

Every input session has axis order `(neurons, time, trials)`. Neuron and trial counts may vary; the time axis must match.

```python
import torch
from neuralign import NeuralignConfig, NeuralignModel

sessions = [
    torch.randn(40, 100, 60),
    torch.randn(52, 100, 75),
]

config = NeuralignConfig(rank=4, max_iter=500, progress=False, seed=42)
factors = NeuralignModel(sessions, config=config).fit("cpu")

shared_time = factors.V                         # (time, rank)
session_trials = factors[0].trial_factors       # (trials, rank)
session_neurons = factors[0].neuron_factors     # (neurons, rank)
predicted = factors[0].latent_trajectories()    # (trials, time, rank)
projected = factors[0].projected_trajectories(sessions[0])
reconstruction = factors[0].reconstructed_tensor()
component = factors[0].reconstructed_tensor(components=2)  # (neurons, time, trials)
latent = factors[0].reconstructed_latent_tensor()          # (rank, time, trials)
latent_subset = factors[0].reconstructed_latent_tensor(components=[0, 2])
trajectories = factors[0].latent_trajectories(components=[0, 2])
unit_weight_latent = factors[0].reconstructed_latent_tensor(components=2, normalized=True)
```

Component indices are zero-based. Omitting `components` reconstructs all components;
an empty sequence returns zeros. Selecting n components gives a rank-at-most-n
contribution, retaining all output neurons or latent coordinates. The latent
reconstruction uses `w, H, S, V, U`; the full reconstruction uses `w, A, V, U`.
Their magnitudes can differ because the fitted `A` is only softly matched to `Q H diag(S)`.
All three reconstruction methods accept `normalized=False` by default.
With `normalized=True`, global weights `w` are replaced by ones without modifying
stored factors. Session gains `S` and other factors are retained; this is not
unit-norm normalization. Gradients then bypass `w`.

Run the complete synthetic example, including held-out-session adaptation, with:

```bash
uv run python examples/quickstart.py
```

## Condition-aware alignment and held-out trials

Supply one trial-label sequence per session to orient the shared coordinates with condition means. Labels orient alignment; they are not targets in the tensor-reconstruction loss.

Set a Boolean `trial_mask` for each session when some trials should be reconstructed but excluded from alignment. `True` means alignment-excluded. This is useful for defining coordinates from training trials and evaluating errors or perturbations afterward.

After fitting, `factors.partial_fit(new_tensor, ...)` freezes the learned global factors and estimates session-specific factors for new recordings. Joint fitting and `partial_fit` answer different validation questions; evaluate held-out-session adaptation prospectively for the intended dataset.

## Documentation

- [Getting started](docs/getting-started.md)
- [Model and outputs](docs/model-and-outputs.md)
- [API overview](docs/api.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Documentation website](https://mikloskralik.github.io/neuralign/)

## Reproducing the manuscript

The manuscript analyses and processed-data loaders live in the companion repository:

```bash
git clone --recurse-submodules https://github.com/MiklosKralik/neural_tensors.git
cd neural_tensors
uv sync --group examples
```

The supplementary generator reads existing result caches and writes vector figures, LaTeX tables, and tidy CSV source data:

```bash
uv run python examples/publication/make_supplementary_materials.py
```

## Citation and release status

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Add the manuscript DOI and software archive DOI after acceptance/release. An explicit software license must also be selected by the copyright holders before broad redistribution.

