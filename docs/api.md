# API reference

```python
from neuralign import NeuralignConfig, NeuralignModel, NeuralignFactors
```

## Fitting settings

Pass settings through `NeuralignConfig` or directly to `NeuralignModel`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `rank` | `4` | Number of components |
| `max_iter` | `2000` | Maximum optimisation steps |
| `lr` | `0.01` | Learning rate |
| `patience` | `100` | Early-stopping patience |
| `tol` | `1e-6` | Early-stopping improvement threshold |
| `seed` | `None` | Random seed |
| `progress` | `True` | Display fitting progress |
| `lambda_tensor_reconstruction` | `1.0` | Reconstruction-loss weight |
| `lambda_procrustes` | `1.0` | Alignment-loss weight |
| `non_negative` | `False` | Nonnegative factors |
| `loss_type` | `"frobenius"` | Also `"poisson"` or `"poisson_log"` |

The default Frobenius loss fits continuous activity. Count losses require nonnegative counts: `"poisson"` models rates and `"poisson_log"` models log-rates. For log-Poisson fits, exponentiate `reconstructed_tensor()` to obtain rates.

## NeuralignModel

```python
NeuralignModel(tensor_sequence, trial_labels=None, trial_mask=None,
               config=None, **settings)
model.fit(device="cpu")
```

`fit` returns `NeuralignFactors`. Inputs use neurons × time × trials and share a time axis.

`trial_labels` provides one label sequence per session. Alignment uses trial means without labels and condition means with labels.

`trial_mask` provides one Boolean array per session. **True excludes a trial from alignment only:** it still contributes to reconstruction. Each session needs an unmasked trial.

## NeuralignFactors

| Method | Behaviour |
| --- | --- |
| `len(factors)` | Number of sessions |
| `factors[i]` or `factors.session(i)` | Session factors |
| `factors.to(device)` | Move tensors to a device |
| `factors.save(path)` | Save factors, configuration and metadata |
| `NeuralignFactors.load(path, map_location="cpu")` | Load a saved fit |
| `factors.partial_fit(...)` | Add sessions with shared factors frozen |

`partial_fit(new_session, device="cpu", **settings)` returns a copy with new sessions appended. It fits their session factors while freezing `w`, `V`, `H` and condition prototypes. New sessions must share the fitted time axis. Recognised labels anchor alignment; sessions without recognised labels use trial means. Supervised adaptation applies a small condition-centroid penalty; `lambda_condition_centroid=0.0` disables it.

## Session outputs

Access a session with `session = factors[i]`. See [model outputs](model-and-outputs.md) for its factor properties and shapes.

```python
session.reconstructed_tensor(components=None, *, normalized=False)
session.reconstructed_latent_tensor(components=None, *, normalized=False)
session.latent_trajectories(layout="ktr", *, components=None, normalized=False)
session.projected_trajectories(Z, layout="ktr")
```

For reconstruction methods, `components` accepts an index or a list of indices; `None` includes all components. `normalized=True` uses unit global weights.

## Simulation

```python
from neuralign.simulation import simulate_sessions
sessions, truth = simulate_sessions(n_sessions=4, n_time=64, noise_std=0.15, seed=42)
```

Generates rank-two recordings with different neuron/trial counts. `noise_std` is relative to the clean signal's standard deviation; `truth` contains the generating factors.
