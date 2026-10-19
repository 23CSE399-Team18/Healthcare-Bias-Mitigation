# Preprocessing code

Python scripts and notebooks live here. Fitted artifacts/complete CSVs are in `../outputs`; dashboard pages and figures are in `../plots`.

## Main scripts

- [confirm_saved_outputs.py](confirm_saved_outputs.py)
- [audit_saved_preprocessing.py](audit_saved_preprocessing.py)
- [audit_and_prepare.py](audit_and_prepare.py)
- [prepare_four_condition_baseline.py](prepare_four_condition_baseline.py)
- [publish_hospital_files.py](publish_hospital_files.py)
- [build_bias_dashboards.py](build_bias_dashboards.py)

Run verification from `D:/complete` using the full path, for example:

`python preprocessing/code/confirm_saved_outputs.py`

Training/rebuild scripts overwrite their own generated artifacts when explicitly run. A changed preprocessing or latent dataset requires rebuilding and revalidating its downstream models. Existing in-processing tasks are preserved; the validator detects changed input hashes.

Refresh navigation/dashboards with `python dataset_audit/publish_layout.py`, then save notebook outputs with `python dataset_audit/execute_current_notebooks.py --stage all`. This publisher reads existing results and does not retrain models.

Raw clinical extraction remains in [cross_dataset](../../cross_dataset/four_conditions/README.md), and original source-to-cohort scripts remain beside their source data. The active numerical preprocessing for both ICU directions is `prepare_four_condition_baseline.py` here.

Generate one file per bias/metric with `python preprocessing/code/build_bias_dashboards.py`. Shared rendering code and assets live in `dataset_audit` at the project root; plot data are saved under this stage's `outputs/bias_dashboards`. Install the dependencies listed in [requirements.txt](requirements.txt).
