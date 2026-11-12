# Model and outputs

Each component combines a shared temporal profile with a weight for each neuron and trial. Neuralign also learns a shared component geometry and a projection for each session.

## Factors

For `session = factors[0]`, the main outputs are:

| Output | Shape | Meaning |
| --- | --- | --- |
| `factors.V` | time × components | Shared temporal profiles |
| `session.neuron_factors` | neurons × components | Neuron weights |
| `session.trial_factors` | trials × components | Expression on each trial |
| `session.scaling_matrix` | components | Session gains |
| `session.projection_matrix` | neurons × latent coordinates | Session projection Q |
| `factors.H` | latent coordinates × components | Shared geometry H |

Latent coordinates and components both have size `rank`. Outputs are PyTorch tensors; use `value.detach().cpu().numpy()` for NumPy plotting.

## Reconstructions and trajectories

| Method | Meaning | Output shape |
| --- | --- | --- |
| `session.reconstructed_tensor()` | Model prediction using neuron factors | neurons × time × trials |
| `session.reconstructed_latent_tensor()` | Model prediction using shared geometry and session gains | latent coordinates × time × trials |
| `session.latent_trajectories()` | The same latent prediction, trials first | trials × time × latent coordinates |
| `session.projected_trajectories(Z)` | Measured activity projected with Q | trials × time × latent coordinates |

A projection shows measured activity in the shared space. A reconstruction shows the model prediction. The model softly couples neuron weights to the shared geometry (`A ≈ Q @ H @ diag(S)`), so projecting a reconstruction need not give the latent reconstruction.

## Select components

```python
one = session.reconstructed_tensor(components=0)
trajectories = session.latent_trajectories(components=[0, 1])
```

Component indices are **zero-based**. Selection keeps the output shape and includes only the selected components' contributions.

Reconstruction methods include global weights `w` by default. `normalized=True` replaces only these weights with ones; it does not normalise the output or remove session gains.

Trajectory methods also accept `layout="tkr"` for time × trials × latent coordinates.

Component order, sign and scale can differ across fits. Match components before comparing them.
