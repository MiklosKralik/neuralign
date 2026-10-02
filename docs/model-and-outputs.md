# Model and outputs

Neuralign jointly models session tensors with shape `(neurons, time, trials)`. For session \(n\), the reconstruction is

\[
\hat Z^{(n)} = \sum_{r=1}^{R} w_r A^{(n)}_r \circ V_r \circ U^{(n)}_r.
\]

`V` and `w` are shared. Each session has its own neuron factors `A`, trial factors `U`, orthonormal projection `Q`, and component gains `S`. A soft penalty draws the neuron factors toward

\[
Q^{(n)} H \operatorname{diag}(S^{(n)}),
\]

where `H` is the shared mixing matrix. Reconstruction still uses `A`, so a session can retain structure that is not perfectly shared.

## Which representation should I use?

- `factors.V` contains shared temporal factors with shape `(time, rank)`.
- `factors[index].trial_factors` contains trial-level component expression with shape `(trials, rank)`.
- `factors[index].neuron_factors` maps components to recorded neurons with shape `(neurons, rank)`.
- `factors[index].latent_trajectories()` returns rank-constrained predicted trajectories with shape `(trials, time, rank)`.
- `factors[index].projected_trajectories(session_tensor)` linearly projects measured activity with `Q` and returns `(trials, time, rank)`.
- `factors[index].reconstructed_tensor()` returns the fitted tensor in the original `(neurons, time, trials)` layout.

Use trial factors to study trial-to-trial component expression. Use predicted trajectories when rank-constrained denoising is part of the analysis. Use projected trajectories when the scientific question requires a linear view of measured population activity or additive neuron/region contributions.

## Label-free and condition-aware alignment

Without `trial_labels`, the trial-averaged trajectory orients each session projection. With labels, condition means supply multiple alignment anchors. Labels do not enter the tensor reconstruction term and are not predicted by the model.

`trial_mask` uses `True` for alignment-excluded trials. Masked trials remain in the reconstruction and retain rows in `U`; this is useful for fitting a coordinate system on training conditions while evaluating perturbation or error trials afterward.

## Identifiability

Like other component models, factors are not identified without conventions for permutation, sign and scale. Neuralign applies gauge fixing during optimization, but comparisons across independent fits should still match components before interpreting component numbers. Similar reconstruction error does not guarantee identical factors. Inspect rank, initialization and alignment-weight sensitivity for each new application.

## Held-out-session adaptation

`NeuralignFactors.partial_fit` freezes `w`, `V`, `H` and any condition prototypes, then estimates session-specific factors for new tensors. It returns a copy and records adaptation history in `metadata["adaptations"]`. This API is appropriate for prospective calibration, but its performance must be validated for the target dataset; fitting every session jointly and adapting an unseen session answer different questions.
