# Saving and evaluating fits

Set a seed and save the fitted factors:

```python
from neuralign import NeuralignFactors, NeuralignModel

factors = NeuralignModel(sessions, rank=4, seed=42).fit("cpu")
factors.save("fit.pt")
loaded = NeuralignFactors.load("fit.pt")
```

Keep the package version or commit, preprocessing settings, session order, labels and evaluation splits alongside the fit. GPU results can vary across hardware. Load saved fits only from trusted sources.

Choose rank by comparing several fits and inspect reconstruction quality. Match components before comparing fits: their order, sign and scale can vary.

Fitting all sessions jointly and holding some out of decoder training measures decoder transfer within the fitted cohort. For a frozen shared representation, use `partial_fit` to add sessions; their factors still use the supplied activity. A `trial_mask` excludes trials from alignment, not reconstruction.
