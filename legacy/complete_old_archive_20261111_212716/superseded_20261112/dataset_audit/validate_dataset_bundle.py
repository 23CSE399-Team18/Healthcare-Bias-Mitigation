"""Check the handoff's cohort counts, labels, feature exclusions and source paths."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(newline="", encoding="utf-8") as inp:
        yield from csv.DictReader(inp)


def check_cohort(path, target, keys, expected_rows, expected_positive):
    count = positive = 0
    seen = set()
    for row in rows(path):
        label = float(row[target])
        assert label in (0, 1), (path, "Nonbinary label")
        identity = tuple(row[key] for key in keys)
        assert identity not in seen, (path, "Duplicate source ID")
        seen.add(identity)
        count += 1
        positive += int(label)
    assert count == expected_rows, (path, count)
    assert positive == expected_positive, (path, positive)
    return {"path": path, "rows": count, "positive": positive,
            "negative": count - positive, "unique_source_ids": True, "binary_labels": True}


def main():
    checks = [
        check_cohort("Heart_correct/dataset_verified/heart_2022_baseline_raw.csv", "HeartDisease",
                     ["_STATE", "SEQNO"], 440111, 39751),
        check_cohort("diabetes_correct/diabetes_baseline_raw.csv", "diabetes_diagnosed",
                     ["SEQN"], 7876, 1073),
        check_cohort("hypertension_correct/nhis2023/derived/hypertension_nhis2023_raw_cohort.csv", "hypertension",
                     ["HHX"], 29471, 11094),
        check_cohort("resp_physionet/resp/dataset_verified/respiratory_cohort_24h_landmark.csv", "hospital_mortality",
                     ["patientunitstayid"], 29126, 3983),
    ]
    for row in rows("dataset_audit/dataset_sources.csv"):
        if row["local_path"]:
            assert (ROOT / row["local_path"]).exists(), row["local_path"]
    nhanes = json.loads((ROOT / "diabetes_correct/cohort_summary.json").read_text())
    assert not {"DIQ010", "LBXGH", "LBXGLU", "diabetes_diagnosed"}.intersection(nhanes["baseline_predictors"])
    nhis = json.loads((ROOT / "hypertension_correct/nhis2023/feature_schema.json").read_text())
    assert not {"HYPEV_A", "hypertension", "HYPMED_A", "WTFA_A", "HHX"}.intersection(nhis["candidate_baseline_predictors"])
    with (ROOT / checks[0]["path"]).open(newline="", encoding="utf-8") as inp:
        columns = csv.DictReader(inp).fieldnames
    assert not {"_MICHD", "CVDINFR4", "CVDCRHD4"}.intersection(columns)
    gz = list((ROOT / "resp_physionet/resp/official_verified").glob("*.gz"))
    assert len(gz) == 30
    assert all(p.stat().st_size < 2**32 for p in gz)
    assert not list((ROOT / "resp_physionet/resp").glob("*.verified_partial"))
    result = {"passed": True, "cohorts": checks, "source_paths_exist": True,
              "target_aliases_excluded_from_explicit_predictor_lists": True,
              "eicu_canonical_gzip_tables": len(gz), "eicu_sources_fit_fat32": True,
              "scope": "Dataset handoff integrity; no model evaluation or fairness guarantee"}
    (ROOT / "dataset_audit/bundle_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
