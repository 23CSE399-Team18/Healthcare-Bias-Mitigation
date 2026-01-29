# Installed API audit: HyperImpute and CTGAN

Checked 9 November 2026 in `C:/Users/ganap/envs/biasprep`, Python 3.11.18.

## HyperImpute 0.1.17

**The installed plugin is not a frozen inductive transformer.** In
`hyperimpute/plugins/imputers/plugin_hyperimpute.py`, `_fit` returns `self`
without fitting, while `_transform(X)` calls `self.model.fit_transform(X)`.
The internal `IterativeErrorCorrection.fit_transform` resets its setup on X,
fits the baseline imputer on X, optimizes column models and fits those models
on X. Thus `.fit(training).transform(validation)` refits on validation values.

The public `models()` accessor exposes models, but no public frozen `transform`
is implemented. A private `_impute_single_column(..., train=False)` exists;
using it safely would require a separately designed/tested adapter retaining
training encoders, limits, feature dependencies, initialization and iteration
state. It must not be silently substituted for the published plugin.

Recommended status: **unsupported for the primary inductive comparison**.
Optionally run the genuine plugin as a clearly labeled transductive masked
reconstruction benchmark wholly within training patients. That benchmark does
not qualify the plugin for applying the same frozen preprocessor to future
patients. Its reconstruction results must not be mixed with inductive
validation/test performance rankings.

Primary evidence: [official implementation](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_hyperimpute.py)
and [original HyperImpute paper](https://proceedings.mlr.press/v162/jarrett22a.html).
The local inspected plugin SHA256 is stored in `catalogue.json`.

## CTGAN 0.13.0

Official CTGAN imports successfully with PyTorch 2.6.0+cu124 and CUDA available
on NVIDIA GeForce RTX 3050 Laptop GPU. Installed constructor supports
`enable_gpu`; `cuda` is deprecated. Pass epochs to the constructor, not `fit`.

Suggested bounded genuine trial:

```python
model = CTGAN(
    epochs=50,
    embedding_dim=64,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    batch_size=500,
    pac=10,
    enable_gpu=torch.cuda.is_available(),
    verbose=False,
)
model.set_random_state(seed)
model.fit(train_table, discrete_columns=discrete_columns)
```

Use only training patients. A single joint generator can include the numeric
predictors, age/sex categories, all four binary task labels and race/ethnicity
as declared discrete columns. Do not include identifiers, future/discharge
variables or artificial hospital identities. Sensitive columns are retained as
generation/audit metadata, never silently introduced as downstream predictors.
Continuous NaNs are rejected by this implementation: apply a training-fitted
imputer first and record its identity. Fifty epochs is a limited comparison
budget, not a convergence or quality guarantee.

`sample(n, condition_column=..., condition_value=...)` **increases the chance**
of the condition and does not guarantee that every generated row satisfies it.
Filter by the requested class value instead of overwriting generated labels.
Validate categorical levels and original clinical min<=mean<=max and numerical
bounds in common original units. Prefer rejecting invalid rows to silently
sorting or clipping them. Bound the number of attempts and record requested,
generated, accepted, rejected and remaining counts. If no valid sample budget
is reached, report incomplete/failed generation rather than substituting a
different sampler under the CTGAN name.

Mark every accepted generated training record as synthetic and keep it separate
from original patients. Validation/test sets contain original records only.
Compare class-conditional feature distributions, demographic distribution,
constraint validity and downstream utility on unchanged held-out patients.

Primary evidence: [official implementation](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py)
and [original CTGAN paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html).
