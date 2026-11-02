# Model and outputs

Each session is reconstructed as a sum of components:

```text
Z_hat[n, neuron, time, trial]
    = sum(w[c] * A[n][neuron,c] * V[time,c] * U[n][trial,c])
```

`w` and the temporal factors `V` are shared. Each session has neuron factors `A`, trial factors `U`, an orthonormal projection `Q` and gains `S`. The shared mixing matrix `H` links component directions across sessions through a soft constraint:

```text
A[n] ≈ Q[n] @ H @ diag(S[n])
```

## Access factors

```python
session = factors[0]                    # same as factors.session(0)
V = factors.V                          # time × components
U = session.trial_factors               # trials × components
A = session.neuron_factors              # neurons × components
Q = session.projection_matrix           # neurons × latent coordinates
S = session.scaling_matrix              # one gain per component
H = factors.H                          # latent coordinates × components
```

Factor arrays are PyTorch tensors. Use `value.detach().cpu().numpy()` for NumPy plotting.

## Choose a representation

| Method | What it represents | Output shape |
| --- | --- | --- |
| `session.reconstructed_tensor()` | Model prediction using fitted A | neurons × time × trials |
| `session.reconstructed_latent_tensor()` | Model prediction using H and S | latent coordinates × time × trials |
| `session.latent_trajectories()` | The same latent prediction, trials first | trials × time × latent coordinates |
| `session.projected_trajectories(Z)` | Measured activity projected with Q | trials × time × latent coordinates |

A projection retains a linear view of measured activity. A reconstruction restricts activity to the fitted component profiles. Because coupling is soft, projecting the full reconstruction need not equal the latent reconstruction.

## Reconstruct selected components

```python
one = session.reconstructed_tensor(components=0)
subset = session.reconstructed_latent_tensor(components=[0, 1])
trajectories = session.latent_trajectories(components=[0, 1])
unit_weights = session.reconstructed_latent_tensor(components=0, normalized=True)
```

Indices are **zero-based**. `None` uses all components; `[]` returns zeros. Selecting components restricts the contributing terms, retaining all output neurons or latent coordinates. A subset is not a new model fit.

The default `normalized=False` includes global weights `w`. Setting it to `True` replaces only `w` with ones; it does not normalise the output or remove session gains. Stored factors are unchanged.

`latent_trajectories` and `projected_trajectories` also accept `layout="tkr"` for time × trials × latent coordinates.

## Compare fits carefully

Components can change order, sign and scale across fits. Match them before comparing component numbers. A shared component does not establish a causal role, and a fitted temporal profile does not estimate arbitrary within-trial changes in that profile.
