# Reproducibility guide

## Deterministic inputs and seeds

Set `NeuralignConfig(seed=...)` and record the complete configuration. GPU operations can remain nondeterministic on some hardware, so publication analyses should report the device, PyTorch/CUDA versions and whether deterministic algorithms were enabled.

Run several initialization seeds. Seeds measure optimization stability; they are not substitutes for independent sessions, subjects or simulated datasets.

## Data conventions

- Every session must use `(neurons, time, trials)` axis order.
- Sessions may have different neuron and trial counts, but must share the time axis.
- Normalize using statistics fit only on the intended training data.
- Preserve the session identity of every trial when cross-validating downstream analyses.
- Use grouped folds when testing transfer to held-out sessions.
- Save trial masks and fold assignments alongside fitted factors.

## What to archive

For a reproducible analysis, save:

1. package and analysis-repository commit hashes;
2. the lockfile and Python/PyTorch/CUDA versions;
3. session inclusion and exclusion tables;
4. preprocessing parameters and trial masks;
5. the complete `NeuralignConfig` and every model seed;
6. fitted factors or a documented script that regenerates them;
7. downstream fold assignments and tidy plotted data;
8. vector figures generated from scripts without notebook-only state.

`NeuralignFactors.save` and `NeuralignFactors.load` persist exported factors, configuration and metadata. Only load files from trusted sources because persistence uses PyTorch serialization.

## Recommended validation

- Plot reconstruction and alignment losses across nearby ranks.
- Match components before comparing independent initializations.
- Compare observed results with label-preserving nulls appropriate to the experimental hierarchy.
- Separate representation fitting from downstream decoder validation in the methods and figures.
- Report unclipped regression scores, including negative values.
- Treat missing brain regions as unavailable rather than as zero activity.

The paper repository contains `examples/publication/make_supplementary_materials.py`, which converts cached result tables into the supplementary figures, LaTeX tables and tidy CSV source data used in the manuscript.
