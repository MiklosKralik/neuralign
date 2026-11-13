# Your recordings

For installation and a runnable simulation, start with the [quickstart](index.md).

## Prepare the data

Use a list of floating-point NumPy arrays or PyTorch tensors, each shaped **neurons × time × trials**. Sessions may contain different neurons and trial counts. Their time bins must represent the same trial-aligned times.

Choose preprocessing appropriate for your recordings. When evaluating held-out data, estimate preprocessing statistics on the training data only.

## Fit and inspect

```python
from neuralign import NeuralignModel

sessions = [recording_a, recording_b]  # your neurons × time × trials arrays
factors = NeuralignModel(sessions, rank=4, seed=42).fit("cpu")

temporal = factors.V                      # time × components
neurons = factors[0].neuron_factors        # neurons × components
trials = factors[0].trial_factors          # trials × components
reconstructed = factors[0].reconstructed_tensor()
```

`rank` sets the number of components. With the default initialisation, it must not exceed any session's neuron count or the number of time bins. Use `fit("cuda")` with a CUDA-enabled PyTorch installation (see [installation](index.md#install)).

To guide alignment with known conditions, pass `trial_labels=labels_by_session`, with one label per trial in each session. Labels guide alignment using condition means; they are not reconstruction targets.

See [model outputs](model-and-outputs.md) to choose factors or trajectories, and the [API](api.md) for fitting options.

The [plotting example](plotting.md) also needs Matplotlib (`python -m pip install matplotlib`). It saves a figure next to the script.
