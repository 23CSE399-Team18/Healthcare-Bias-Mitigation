# Latent code

Python scripts and notebooks live here. Fitted artifacts/complete CSVs are in `../outputs`; dashboard pages and figures are in `../plots`.

## Main scripts

- [validate_latent.py](validate_latent.py)
- [validate_hospital_latent.py](validate_hospital_latent.py)
- [check_handoff.py](check_handoff.py)
- [run_latent.py](run_latent.py)
- [build_hospital_latent.py](build_hospital_latent.py)
- [build_bias_dashboards.py](build_bias_dashboards.py)

Run verification from `D:/complete` using the full path, for example:

`python latent/code/validate_latent.py`

Training/rebuild scripts overwrite their own generated artifacts when explicitly run. A changed preprocessing or latent dataset requires rebuilding and revalidating its downstream models. Existing in-processing tasks are preserved; the validator detects changed input hashes.

Refresh navigation/dashboards with `python dataset_audit/publish_layout.py`, then save notebook outputs with `python dataset_audit/execute_current_notebooks.py --stage all`. This publisher reads existing results and does not retrain models.

Generate one file per bias/metric with `python latent/code/build_bias_dashboards.py`. Shared rendering code and assets live in `dataset_audit` at the project root; plot data are saved under this stage's `outputs/bias_dashboards`. Install the dependencies listed in [requirements.txt](requirements.txt).
