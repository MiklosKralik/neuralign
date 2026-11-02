# API reference

```python
from neuralign import NeuralignConfig, NeuralignModel, NeuralignFactors
```

## NeuralignConfig

`NeuralignConfig(**settings)` sets fitting options. Common settings are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `rank` | `4` | Number of components |
| `max_iter` | `2000` | Maximum optimisation steps |
| `lr` | `0.01` | Adam learning rate |
| `patience` | `100` | Early-stopping patience |
| `tol` | `1e-6` | Early-stopping improvement threshold |
| `seed` | `None` | Random seed; set explicitly for repeatable runs |
| `progress` | `True` | Display fitting progress |
| `lambda_tensor_reconstruction` | `1.0` | Reconstruction-loss weight |
| `lambda_procrustes` | `1.0` | Soft alignment-loss weight |
| `non_negative` | `False` | Nonnegative factor constraint |
| `loss_type` | `"frobenius"` | Reconstruction loss; also `"poisson"` or `"poisson_log"` |

The simulation uses continuous data and the default Frobenius loss. Count losses require appropriately prepared nonnegative count data; they are not for z-scored observations.

## NeuralignModel

```python
NeuralignModel(tensor_sequence, trial_labels=None, trial_mask=None,
               config=None, **overrides)
model.fit(device="cpu")
```

Construct the model from session tensors, then call `fit` to obtain `NeuralignFactors`. Configuration keywords can also be passed as overrides. Each session uses neurons × time × trials; all sessions share a time axis.

`trial_labels` supplies one sequence of labels per session. Without labels, alignment uses trial means; with labels, it uses condition means.

`trial_mask` supplies one Boolean array per session. **True excludes a trial from alignment only.** Masked trials still contribute to reconstruction and receive trial factors. A mask therefore does not fully hold trials out of model fitting. Every session needs at least one unmasked trial.

## NeuralignFactors

The fitted collection holds `w`, `V`, `H`, configuration, metadata and session factors.

| Method | Behaviour |
| --- | --- |
| `len(factors)` | Number of sessions |
| `factors.session(i)` or `factors[i]` | Session view with named factor properties |
| `factors.to(device)` | Move stored tensors to a device |
| `factors.save(path)` | Save factors, configuration and metadata |
| `NeuralignFactors.load(path, map_location="cpu")` | Load a saved fit |
| `factors.partial_fit(...)` | Fit new sessions with shared factors frozen |

```python
factors.save("fit.pt")
loaded = NeuralignFactors.load("fit.pt")
adapted = loaded.partial_fit(new_session, device="cpu", max_iter=500)
new_view = adapted[-1]
```

`partial_fit(tensor, trial_labels=None, trial_mask=None, *, device=None, config=None, **overrides)` returns a copy with new sessions appended. It freezes `w`, `V`, `H` and any condition prototypes, and fits only new session factors. A sequence of new sessions is also accepted. All must share the fitted time axis. With labels, recognised conditions anchor alignment; unknown labels are omitted from alignment. The original collection is unchanged.

## Session view

| Property | Shape |
| --- | --- |
| `neuron_factors` | neurons × components |
| `trial_factors` | trials × components |
| `projection_matrix` | neurons × latent coordinates |
| `scaling_matrix` | components |
| `weight_matrix` | components; shared w |
| `temporal_factors` | time × components; shared V |
| `mixing_matrix` | latent coordinates × components; shared H |

```python
session.reconstructed_tensor(components=None, *, normalized=False)
session.reconstructed_latent_tensor(components=None, *, normalized=False)
session.latent_trajectories(layout="ktr", *, components=None, normalized=False)
session.projected_trajectories(Z, layout="ktr")
```

The first three methods accept one integer or a one-dimensional sequence of distinct component indices. Negative, duplicate, Boolean, noninteger and out-of-range indices are rejected. Outputs preserve device, dtype and gradients. `normalized=True` uses unit global weights without changing stored factors. See [Model and outputs](model-and-outputs.html) for shapes and interpretation.

`predicted_neuron_factors()` returns `Q @ H @ diag(S)`; `scaled_mixing_matrix()` returns `H @ diag(S)`. `TensorFactors` is the lower-level storage class for a session's `A`, `U`, `Q` and `S`; use the session view for routine analysis.

## Simulation helper

```python
from neuralign.simulation import simulate_sessions
sessions, truth = simulate_sessions(n_sessions=4, n_time=64, noise_std=0.15, seed=42)
```

Generates rank-two recordings with different neuron/trial counts. `noise_std` is relative to each clean signal's standard deviation. `truth` stores shared temporal factors and per-session generating factors. The helper uses NumPy and PyTorch only.
