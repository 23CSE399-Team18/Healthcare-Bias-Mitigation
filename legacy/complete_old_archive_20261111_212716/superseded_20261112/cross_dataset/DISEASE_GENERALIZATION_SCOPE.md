# Current four-condition generalization scope

The user requested heart, respiratory, diabetes and hypertension, and authorised source checks followed by preprocessing. They delegated the respiratory outcome choice for this additional study.

The additional hospital study uses four separate retrospective recorded-diagnosis classification targets. The original respiratory mortality application is preserved separately.

[Current datasets, phenotype rules, preprocessing and limitations](four_conditions/README.md).

Train/validate/test hospitals are separated in eICU; repeated patients crossing these partitions are purged. MIMIC is an external database from one institution, with its own patient-separated partitions for the reverse direction. Harmonised predictors and code families do not prove identical clinical label ascertainment. The CDC survey tasks remain population benchmarks, separate from these ICU classification tasks.

Preprocessing is complete. Predictive model fitting, four-condition cross-hospital evaluation and mitigation comparisons remain to be done.
