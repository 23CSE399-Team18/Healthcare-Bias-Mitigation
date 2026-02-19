"""Write the current approved dataset list after the user-authorized cleanup."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    current = ROOT / "cross_dataset/four_conditions/preprocessing_status.json"
    if current.exists() and json.loads(current.read_text(encoding="utf-8")).get("completed"):
        import sys
        sys.path.insert(0, str(ROOT / "cross_dataset"))
        from write_four_condition_report import main as write_current_report
        write_current_report()
        return
    approved = [
        {"application": "Heart", "source": "CDC BRFSS 2022", "rows": 440111,
         "cohort": "Heart_correct/dataset_verified/heart_2022_baseline_raw.csv",
         "raw": "Heart_correct/LLCP2022.XPT", "target": "HeartDisease",
         "meaning": "Self-reported history of heart attack or coronary heart disease/angina",
         "source_url": "https://www.cdc.gov/brfss/annual_data/annual_2022.html"},
        {"application": "Hypertension", "source": "CDC NHIS 2023 Sample Adult", "rows": 29471,
         "cohort": "hypertension_correct/nhis2023/derived/hypertension_nhis2023_raw_cohort.csv",
         "raw": "hypertension_correct/nhis2023/raw/adult23.csv", "target": "hypertension",
         "meaning": "Self-reported ever clinician-diagnosed hypertension",
         "source_url": "https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html"},
        {"application": "Diabetes", "source": "CDC NHANES August 2021–August 2023", "rows": 7876,
         "cohort": "diabetes_correct/diabetes_baseline_raw.csv", "raw": "diabetes_correct/raw",
         "target": "diabetes_diagnosed", "meaning": "Self-reported previously diagnosed diabetes",
         "source_url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023"},
        {"application": "Respiratory", "source": "PhysioNet full eICU-CRD v2.0", "rows": 29126,
         "cohort": "resp_physionet/resp/dataset_verified/respiratory_cohort_24h_landmark.csv",
         "raw": "resp_physionet/resp/official_verified", "target": "hospital_mortality",
         "meaning": "Subsequent in-hospital mortality in respiratory ICU patients eligible at 24 hours",
         "source_url": "https://physionet.org/content/eicu-crd/2.0/"},
    ]
    for item in approved:
        assert (ROOT / item["cohort"]).is_file(), item["cohort"]
        assert (ROOT / item["raw"]).exists(), item["raw"]
    removed = json.loads((ROOT / "dataset_audit/cleanup_completed.json").read_text())
    (ROOT / "dataset_audit/approved_datasets.json").write_text(json.dumps(approved, indent=2), encoding="utf-8")
    summary_path = ROOT / "dataset_audit/dataset_bundle_summary.json"
    summary = json.loads(summary_path.read_text())
    def optional_json(relative):
        path = ROOT / relative
        return json.loads(path.read_text()) if path.exists() else {}
    study = optional_json("cross_dataset/status.json")
    browser = optional_json("cross_dataset/browser_transfer_status.json")
    imported = optional_json("cross_dataset/browser_import_state.json")
    job = optional_json("cross_dataset/completion_job.json")
    eicu = optional_json("cross_dataset/prepared/eicu_summary.json")
    mimic_metadata = optional_json("cross_dataset/prepared/mimic_cohort_validation.json")
    source_audit = optional_json("dataset_audit/source_verification_status.json")
    job_display = "complete" if study.get("evaluation_completed") else job.get("status", "not started")
    mimic_feature_status = (
        "Both real clinical feature tables are prepared."
        if study.get("mimic_prepared")
        else "The MIMIC clinical feature table still requires the completed charted-events source."
    )
    if source_audit.get("mode") == "source_check_only_requested_by_user":
        mimic_status = "Processing stopped at user request; verifying sources for all four applications"
    elif study.get("evaluation_completed"):
        mimic_status = "MIMIC and eICU prepared; cross-source baseline evaluation completed"
    elif study.get("mimic_prepared"):
        mimic_status = "MIMIC cohort prepared; cross-source baseline evaluation pending"
    elif imported.get("status") == "complete":
        mimic_status = "Browser downloads verified and imported; MIMIC feature extraction pending"
    elif browser.get("access_confirmed_by_user"):
        mimic_status = "Website access confirmed; browser downloads and checksum verification in progress"
    else:
        mimic_status = "Awaiting MIMIC-IV project access"
    summary.update(cleanup_removed_files=len(removed), cleanup_removed_bytes=sum(r["bytes"] for r in removed),
                   cross_dataset_study_required=True, mimic_status=mimic_status,
                   mimic_downloaded=imported.get("status") == "complete",
                   mimic_verified_source_files=len(imported.get("files", [])),
                   mimic_metadata_cohort_rows=mimic_metadata.get("rows"),
                   cross_dataset_completion_job=job_display)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    source_index = ROOT / "dataset_audit/dataset_sources.csv"
    if source_index.exists():
        with source_index.open(newline="", encoding="utf-8") as inp:
            reader = csv.DictReader(inp)
            fieldnames, records = reader.fieldnames, list(reader)
        for record in records:
            if "MIMIC" in record.get("source", ""):
                record.update(application="required cross-dataset generalization",
                              local_path="cross_dataset/raw/mimiciv/3.1", acquisition_status=mimic_status)
        with source_index.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# Approved project datasets — current status\n",
        "**Yes: these datasets suit your project's multi-stage fairness study, with the defined tasks below.** "
        "They cover the four clinical applications and retain demographic fields for subgroup evaluation. "
        "They are four separate prediction tasks, not interchangeable outcomes.\n",
        "| Application | Approved source | Labelled records | Dataset to use |",
        "|---|---|---:|---|",
    ]
    for item in approved:
        lines.append(f"| {item['application']} | [{item['source']}]({item['source_url']}) | {item['rows']:,} | [Open cohort]({item['cohort']}) |")
    if source_audit.get("mode") == "source_check_only_requested_by_user":
        lines += [
            "\n## Source verification requested before further construction\n",
            "The downloaded raw sources are genuine. Existing cohort/feature CSVs were research-derived from those sources; "
            "they are not four prepackaged MIMIC disease datasets. Processing is currently stopped. "
            "**Four matching hospital-based disease datasets have not yet been verified.** "
            "The CDC survey benchmarks alone cannot establish hospital-held-out performance. "
            "MIMIC diagnosis tables still need source inspection, consistent phenotype rules, and timing review. "
            "[Source check](dataset_audit/FOUR_CONDITION_SOURCE_CHECK.md).\n",
        ]
    lines += [
        "\n## Correct outcomes\n",
        *[f"- **{item['application']}:** {item['meaning']}. Target `{item['target']}`." for item in approved],
        "\nRead each dictionary and explicit predictor list. Source-label aliases, identifiers, sample weights "
        "and sampling-design columns must not automatically become features. Unknown labels are excluded and audited. "
        "Missing predictors are retained for the preprocessing comparison. The respiratory file defines cohort membership; "
        "its clinical features still require extraction from the raw tables.\n",
        "## Cleanup completed\n",
        f"Removed **{len(removed)} wrong, incomplete or superseded dataset/cache files**, totalling "
        f"{sum(row['bytes'] for row in removed)/1e9:.2f} GB. "
        "This includes the hypertension BRFSS 2024 file, both truncated eICU CSVs, duplicate uncompressed copies, "
        "the unverified unused admissionDrug copy, and obsolete prepared cohorts/extraction caches. "
        "Correct source archives, fresh cohorts, dictionaries and manifests remain. "
        "No files outside D:/complete were deleted.\n",
        "Old notebooks, figures, tables and fitted-model caches are in "
        "[legacy_results_not_for_training](dataset_audit/legacy_results_not_for_training). "
        "They are historical project work, not current training data. "
        "[Exact removal list](dataset_audit/cleanup_completed.json).\n",
        "## Original source links\n",
        "- [Heart: BRFSS 2022 ZIP](https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip)",
        "- [Hypertension: NHIS 2023 adult CSV ZIP](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip)",
        "- [Diabetes: all six downloads and codebooks](diabetes_correct/DATASET_README.md)",
        "- [Respiratory: full eICU-CRD](https://physionet.org/content/eicu-crd/2.0/)",
        "\nThe existing heart archive was checked locally. Hypertension and diabetes were downloaded from CDC. "
        "Thirty complete eICU compressed tables were copied from your existing download and checked against its supplied manifest, "
        "with gzip CRC validation. An authenticated comparison with PhysioNet's live manifest remains separate.\n",
        "## Required MIMIC–eICU generalization study\n",
        "This is now part of the requested work. The additional study uses the **same adult ICU mortality task** in both sources: "
        "patients remain in ICU and hospital beyond 24 hours; predictors use the first 24 hours; "
        "the outcome is subsequent in-hospital death. Train on eICU and test on MIMIC, then reverse the direction, "
        "with patient-separated source splits and demographic performance evaluation.\n",
        "The original respiratory disease task remains eICU-only. MIMIC discharge diagnoses do not supply reliable by-24-hour "
        "diagnosis timing, so they cannot simply replace eICU's time-stamped respiratory cohort rule. "
        "The additional common ICU task avoids that mismatch.\n",
        f"The prepared eICU feature table contains **{eicu.get('rows', 0):,} stays**. "
        f"Verified MIMIC metadata defines **{mimic_metadata.get('rows', 0):,} eligible stays**. "
        f"{mimic_feature_status}\n",
        "[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) requires credentialing, required training and its project DUA. "
        "Your signed-in browser account is being used for website downloads; the large file is downloading in Microsoft Edge. "
        "No browser cookies or credentials are read by the import tool. "
        f"**Current status: {mimic_status}.**\n",
        "[Study setup](cross_dataset/README_STUDY.md) · [Preferred browser instructions](cross_dataset/README_BROWSER.md).\n",
        f"The automatic completion runner is **{job_display}**. "
        "It verifies/imports the finished browser file, prepares MIMIC features and runs both transfer directions. "
        "[Live local job status](cross_dataset/completion_job.json) · "
        "[Processing log](cross_dataset/completion_job.log). "
        "No result is complete while a download or feature-extraction stage is still pending.\n",
        "## Practical limits\n",
        "D uses FAT32. Read eICU `.csv.gz` files directly. Large MIMIC downloads are stored as verified segments. "
        "Dataset choice supports retrospective fairness experiments; it does not itself prove clinical safety, equitable new data "
        "collection, deployed monitoring or formal privacy protection. The suitability of each specific method in the images still "
        "needs evaluation on each task.\n",
        "[Project-slide correction](dataset_audit/PDF_DATASET_REPLACEMENT.md). Original PDF and reference images retained.\n",
    ]
    (ROOT / "DATASET_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"approved_applications": len(approved), "removed_files": len(removed), "mimic_status": summary["mimic_status"]}))


if __name__ == "__main__":
    main()
