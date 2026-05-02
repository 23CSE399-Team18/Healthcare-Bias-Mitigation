"""Refresh current reproducibility commands and summarize completed verification."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[1]

def main():
    migration=json.loads((ROOT/'dataset_audit/layout_migration.json').read_text())
    moves={r['old']:r['new'] for r in migration['file_moves'] if Path(r['old']).suffix in ('.py','.ipynb') or Path(r['old']).name=='environment.json'}
    documents=[p for s in ('preprocessing','latent','inprocessing','dataset_audit') for p in (ROOT/s).rglob('*.md')]
    documents.append(ROOT/'cross_dataset/four_conditions/README.md')
    updated=[]
    for path in documents:
        before=path.read_text(encoding='utf-8');after=before
        for old,new in sorted(moves.items(),key=lambda kv:-len(kv[0])):after=after.replace(old,new)
        after=after.replace('latent/hospital_outputs','latent/outputs/hospital').replace('inprocessing/REPORT.md','inprocessing/outputs/REPORT.md')
        after=after.replace('`cross_dataset/four_conditions/baseline_preprocessed`','`preprocessing/outputs/cross_dataset_baseline`')
        if path==ROOT/'cross_dataset/four_conditions/README.md':
            after=after.replace('`baseline_preprocessed/mimic_to_eicu/` and `baseline_preprocessed/eicu_to_mimic/`','[MIMIC-source matrices](../../preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu) and [eICU-source matrices](../../preprocessing/outputs/cross_dataset_baseline/eicu_to_mimic)')
        if path==ROOT/'preprocessing/outputs/verified_baseline/README.md':
            after=after.replace('The `mimic/` folder here adds plots and plot data without duplicating its matrices.','The `mimic/` folder here contains complete CSV exports, their read-back evidence and plot data. Figures are in `preprocessing/plots/baseline/mimic/` at the project root. The source-fitted matrices and transformer remain in the cross-dataset baseline folder linked above.')
        if path==ROOT/'dataset_audit/recheck_current/REPORT.md':
            note='> Current completion status: in-processing and hospital evaluation are now complete. This document describes the earlier preprocessing/latent review. See [the final project review](../FINAL_REVIEW.md) and [measured in-processing results](../../inprocessing/outputs/REPORT.md).\n\n'
            if not after.startswith(note):after=note+after
        if after!=before:path.write_text(after,encoding='utf-8');updated.append(path.relative_to(ROOT).as_posix())
    reports={
      'Original preprocessing':'preprocessing/outputs/audit_current/confirmation.json',
      'MIMIC/eICU preprocessing':'preprocessing/outputs/audit_current/mimic_audit.json',
      'Latent artifacts':'latent/outputs/validation.json',
      'Latent plot metrics':'dataset_audit/recheck_current/verification.json',
      'Latent loader':'latent/outputs/handoff_check.json',
      'Hospital preprocessing and latent':'latent/outputs/hospital/independent_validation.json',
      'In-processing models and metrics':'inprocessing/outputs/validation.json',
      'Final prediction CSVs':'inprocessing/outputs/export_validation.json',
      'All executed notebooks':'dataset_audit/notebook_execution_all.json',
      'Folder structure and links':'dataset_audit/layout_validation.json',
      'Fresh full pipeline recheck':'dataset_audit/active_recheck/checks.json',
      'Individual chart data and file inventory':'dataset_audit/active_recheck/dashboard_validation.json',
      'Individual dashboard control logic':'dataset_audit/active_recheck/dashboard_control_tests.json'}
    for path in reports.values():assert json.loads((ROOT/path).read_text())['passed'],path
    stage_counts={s:0 for s in ('preprocessing','latent','inprocessing')}
    catalogue=json.loads((ROOT/'DATASETS.json').read_text())
    for item in catalogue['complete_csv_inventory']:stage_counts[item['stage']]+=1
    lines=['# Final project review','','The requested **preprocessing -> latent -> in-processing** artifacts are present in `D:/complete`. Each stage uses `code`, `outputs` and `plots` folders. There are **144 individual bias/metric HTML pages**: 64 preprocessing, 32 latent and 48 in-processing. Each file covers one bias or latent metric for a current dataset/task.','',
      '[Open the individual dashboard catalogue](../BIAS_DASHBOARDS.md). [Read the code/method review](CODE_REVIEW.md), including the difference between the fixed preprocessing baseline and the algorithms shown in the supplied image. [Every current stage file is inventoried here](active_recheck/ALL_STAGE_FILES.csv).','',
      'This verification confirms saved transformations, data alignment, reproducible model predictions, metric calculations and local file integration. It does not certify strong predictive performance, statistical significance or universal hospital generalization.','',
      '| Stage | Complete CSV exports | Contents |','|---|---:|---|',
      f'| Preprocessing | {stage_counts["preprocessing"]} | Original five baselines plus hospital/external branches |',
      f'| Latent | {stage_counts["latent"]} | Selected representations for the five baselines and three hospital/transfer branches |',
      f'| In-processing | {stage_counts["inprocessing"]} | Selected predictions for six branches and 12 tasks |','',
      'The pipeline retains 191 fitted in-processing candidates/references, nine executed notebooks, 49 plot images and the original frozen selection records. Each complete CSV contains its feature/latent columns, labels and metadata; prediction exports also include task probabilities and decisions.','',
      '## Verification evidence','','| Check | Result | Evidence |','|---|---|---|']
    for title,path in reports.items():
        target=path.removeprefix('dataset_audit/') if path.startswith('dataset_audit/') else '../'+path
        lines.append(f'| {title} | Passed | [Saved report]({target}) |')
    lines+=['','Preprocessing statistics were fitted on training rows and checked against saved matrices/CSVs. Hospital partitions were checked for hospital/patient overlap. Encoders reproduce saved latent values, and all 45 latent method/task comparison rows were recomputed. Saved in-processing models reproduce sampled predictions; all 506 validation/test/external metric rows and the complete prediction CSVs were reconciled. Model and prediction artifacts were not retrained during this folder cleanup.','',
      '## Dataset and study scope','','The five sources are CDC BRFSS 2022, eICU-CRD v2.0, NHANES August 2021-August 2023, NHIS 2023 and MIMIC-IV v3.1. [The full catalogue](../DATASETS.md) records targets, cohort counts, hospital counts, feature widths, split sizes and local source paths. [The complete CSV inventory](../DATASET_FILES.csv) lists all 69 files.','',
      '- Respiratory mortality uses 29 held-out eICU test hospitals; the four recorded-diagnosis tasks use 31 held-out eICU test hospitals.',
      '- Source-fitted MIMIC/eICU external transfer is also evaluated. Anonymous hospital IDs cannot prove institutional non-overlap across the two databases.',
      '- Survey tasks have no hospital IDs. They remain separate population benchmarks.',
      '- Respiratory mortality and recorded respiratory diagnosis are different outcomes. ICU diagnosis labels overlap and reflect documentation.',
      '- Methods requiring adaptation are named in [the protocol](../inprocessing/code/PLAN.md). Performance and fairness effects are mixed. These are bounded-budget single-seed experiments.','',
      '## Remaining limitations','','A live browser visual review could not be completed: browser access is unavailable, and the prior native browser attempt was stopped because its URL could not be verified. Local HTML structure, tab-to-panel wiring, figures and links passed checks. This report does not claim browser screenshots or interaction testing.','',
      'The original MIMIC/eICU preprocessing pickle reports scikit-learn 1.9.1 while the Python 3.12 validation environment uses 1.8.0. Its training statistics and transformation outputs matched independent reconstruction. The numeric matrices used by latent/in-processing do not depend on loading that pickle. Preserve the recorded dependencies when reusing it.','',
      '## Open the results','','- [Start page](../START_HERE.html)', '- [Individual in-processing dashboards](../inprocessing/index.html)', '- [Preprocessing dashboards](../preprocessing/plots/index.html)', '- [Latent dashboards](../latent/plots/index.html)', '- [Measured in-processing results](../inprocessing/outputs/REPORT.md)','',
      'Old experiments remain outside this project in `D:/complete_old_archive_20261111_212716`. The original archive manifest, additional historical-audit archive record and layout migration hashes are retained in this audit folder.']
    (ROOT/'dataset_audit/FINAL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'dataset_audit/documentation_review.json').write_text(json.dumps({'passed':True,'updated_command_documents':updated,'complete_csv_counts':stage_counts,'verification_reports':reports,'live_browser_review_completed':False},indent=2),encoding='utf-8')
    print(json.dumps({'documentation_updated':updated,'complete_csv_counts':stage_counts,'all_referenced_checks_passed':True}),flush=True)

if __name__=='__main__':main()
