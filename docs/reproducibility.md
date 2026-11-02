# Reproducibility

## Record a fit

Set a seed and save the fitted collection:

```python
config = NeuralignConfig(rank=6, seed=42)
factors = NeuralignModel(sessions, config=config).fit("cpu")
factors.save("fit.pt")
```

Keep the package version or commit, preprocessing settings, session order and trial labels alongside it. Save decoder splits separately. GPU results can vary across hardware even with a fixed seed. Load PyTorch files only from trusted sources.

## Separate fitting from evaluation

Fitting all sessions jointly, then holding sessions out of decoder training, measures decoder transfer within that shared fit. It does not hold sessions out of representation fitting.

Use `partial_fit` when adding recordings to a frozen shared representation. Its new session factors still use the supplied activity, so define the evaluation split for the scientific question. A `trial_mask` excludes trials from alignment, not reconstruction.

For new datasets, compare several ranks and seeds, inspect reconstruction and alignment losses, and match components before interpreting differences across fits. Repeated fits measure optimisation variability, not biological replication.

## Manuscript analyses

The [companion repository](https://github.com/MiklosKralik/neural_tensors) contains the experimental analyses, session lists and figure notebooks. The package's quickstart is independent of those recordings and cached fits.
