# Verified fixed preprocessing baseline

Open [the plots](../../plots/index.html) and [the audit report](../audit_current/REPORT.md).

For heart, respiratory, diabetes and hypertension, use the three
`<dataset>/<dataset>_<split>_preprocessed.csv` files (train, validation, test).
Predictors are all columns except `target`, `meta_*`, and `sens_*`.
Each folder also contains aligned `X_<split>.npz` / `y_<split>.npz`, feature names,
the fitted preprocessor, validation evidence, and the exact data behind its plot.

The fixed pipeline uses training medians, training missing indicators, training
standardization and one-hot encoding. Categorical missingness gets its own value;
unseen categories are ignored by the frozen encoder. All source rows are kept.
Numeric binary/ordinal source features are scaled alongside other numeric
predictors; text categories are one-hot encoded. No clinical values are relabeled,
no people are removed based on matching feature vectors, and no synthetic rows
are generated. Survey weights/design columns remain available in metadata.

Survey partitions are approximately 70% / 15% / 15%, stratified by target.
Respiratory partitions use the same approximate fractions while keeping all
stays for a patient together. They do not hold out whole hospitals.

MIMIC already has verified baseline preprocessing in
[`preprocessing/outputs/cross_dataset_baseline`](../cross_dataset_baseline).
Use `mimic_to_eicu` when fitting on MIMIC, or `eicu_to_mimic` when fitting on eICU.
Each `y_<split>.npz` has four named label arrays: `heart`, `respiratory`, `diabetes`,
and `hypertension`; `rows_<split>.csv` preserves their row identities.
The `mimic/` folder here contains complete CSV exports, their read-back evidence and plot data. Figures are in `preprocessing/plots/baseline/mimic/` at the project root. The source-fitted matrices and transformer remain in the cross-dataset baseline folder linked above.

This is verified baseline preprocessing. It does not constitute the eight-bias
mitigation experiment or demonstrate better accuracy or fairness. Historical notebooks and experimental outputs were moved outside this project.
See [the archive manifest](../../../dataset_audit/archive_manifest.json) for their locations.
Their test scores were used during selection and remain exploratory.

Rebuild and validate with `python preprocessing/code/audit_and_prepare.py` from the
project root using numpy, pandas, scikit-learn, joblib and matplotlib.
