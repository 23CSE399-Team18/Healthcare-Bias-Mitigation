# Independent source and phenotype review

Reviewed 9 November 2026. This report concerns the additional four-condition
hospital study. The original respiratory ICU mortality task remains separate.

## Decision

MIMIC-IV v3.1 contains real recorded diagnoses for all four requested condition
families. It is suitable for constructing four **retrospective recorded-diagnosis
classification benchmarks**, subject to successful file-integrity verification
and the cohort checks below. This is an appropriate additional hospital-data
study for the project; it does not make the existing CDC surveys hospital data.

The new target is whether the selected condition family was documented in the
source diagnosis table; for eICU the final recommended primary label also
includes selected explicit chronic-condition past history, as detailed below.
A zero means no matching condition was documented in these selected sources,
not a clinically verified disease-free patient.
Neither new disease onset nor availability of a diagnosis by ICU hour 24 can be
established from MIMIC discharge billing codes. These are researcher-defined
cohorts from official records, not four separately published PhysioNet datasets.

## Explicit condition definitions

Codes are normalized by removing periods and uppercasing. Match the first three
characters, with ICD version respected. eICU's `icd9code` field can contain both
numeric ICD-9-CM and letter-prefixed ICD-10-CM codes separated by commas; parse
each token rather than treating that whole field as one ICD-9 code. Use the same
code-family rules in both hospital sources.

| Research task | ICD-9-CM family | ICD-10-CM family | Scope and exclusions |
|---|---|---|---|
| Heart: recorded ischemic heart disease | 410-414 | I20-I25 | Includes myocardial infarction, angina and coronary/ischemic heart disease. This is a specific heart-disease family, not every heart disorder. Do not add heart failure or arrhythmias to the target silently. |
| Respiratory: recorded respiratory-system disease | 460-519 | J00-J99 | Broad respiratory-system category, including infections, chronic disease and respiratory failure. It is not ARDS, COPD, or respiratory mortality alone. Pulmonary embolism and pulmonary hypertension are outside these chapter ranges. |
| Diabetes: recorded non-gestational diabetes | 249, 250 | E08, E09, E10, E11, E13 | Includes secondary/drug-induced, type 1, type 2 and other specified diabetes. Not type-2-only. Excludes isolated gestational codes, prediabetes and isolated hyperglycemia. E12 is not a US ICD-10-CM diabetes family here. |
| Hypertension: recorded systemic hypertension | 401-405 | I10, I11, I12, I13, I15 | Includes essential, secondary and hypertensive heart/renal disease. Excludes pulmonary hypertension, pregnancy-only hypertension and an isolated raised BP. I14 is not a US ICD-10-CM hypertension family here. Isolated hypertensive-crisis I16 is not included in this deliberately specified core family. |

These ranges define reproducible research targets; they are not presented as a
clinically validated case-finding algorithm. CDC's coronary-heart-disease
hospitalization definition additionally lists ICD-9-CM 429.2. The selected
410-414/I20-I25 definition is deliberately called **ischemic heart disease**;
it is a related hospital-record concept for BRFSS's self-reported MI/CHD/angina
target, not an assertion of identical label ascertainment or lifetime history.

The official [CDC Atlas source definitions](https://www.cdc.gov/heart-disease-stroke-atlas/data-sources/index.html)
support the coronary/ischemic and systemic-hypertension families and show why
all-heart-disease is a broader category. The [US National Center for Health
Statistics ICD-10-CM files](https://www.cdc.gov/nchs/icd/icd-10-cm/files.html)
are the government classification reference. The downloaded official MIMIC
diagnosis dictionary supplies the code-specific descriptions and ICD version.
The [MIT-LCP Charlson implementation](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/comorbidity/charlson.sql)
is additional evidence that MIMIC diagnosis coding supports MI, diabetes and
chronic pulmonary phenotypes, but its narrower comorbidity definitions are not
substituted for the four broader families above.

## Actual downloaded-table evidence

A read-only scan of `C:\Users\ganap\Downloads\d_icd_diagnoses.csv.gz` found
the following dictionary entries matching the exact ranges above. A separate
read-only full scan of `diagnoses_icd.csv.gz` read **6,364,488 diagnosis rows**
and confirmed that all four families actually occur in patient records.

| Family | Matching dictionary entries | Matching diagnosis rows | Distinct hospital admissions with a matching code |
|---|---:|---:|---:|
| Ischemic heart disease | 168 | 166,211 | 108,833 |
| Systemic hypertension | 52 | 279,107 | 277,795 |
| Non-gestational diabetes | 637 | 181,189 | 134,298 |
| Broad respiratory-system disease | 714 | 242,615 | 159,122 |

These are aggregate evidence of condition presence in the source, covering all
available admissions and diagnosis times. They are **not final eligible adult
ICU-cohort counts**, prevalence estimates or mutually exclusive groups. People
and admissions can belong to several tasks. The dictionary includes allowable
codes not necessarily used in the patient diagnosis table. Integrity results
are owned by the separate download verification report; these scans do not
replace publisher-checksum verification.

## Cohort and preprocessing requirements

- Restrict both sources to adults with an ICU stay extending beyond the same
  24-hour observation landmark, retaining documented eligibility flow counts.
- Use the first 24 hours of time-stamped measurements for common predictors.
  In MIMIC, check measurement `charttime` and available `storetime`; a measurement
  recorded after the landmark must not be treated as already available.
- Derive labels separately from diagnosis records. Never include diagnosis
  codes, condition flags, diagnoses text, discharge status, death time or future
  measurements in the predictor matrix. A shared observation window does not
  establish when a chronic condition began.
- Retain patient, stay and hospital identifiers for splitting and audit only.
  Split by patient; hospital-held-out eICU evaluation also needs whole hospitals
  separated and patients appearing across hospitals removed from split overlap.
- Leave missing clinical values explicit in exported raw feature tables.
  Imputation, scaling, feature selection, sampling and any mitigation must be
  fitted on training data only after the split.
- Cases have matching codes. Potential controls require other diagnosis
  documentation; a stay with no usable diagnosis documentation is unknown and
  should not become a negative automatically. Even among potential controls,
  zero remains a documentation label rather than proof of disease absence.
- Preserve demographics and missing/unknown categories for fairness auditing.
  Public survey sampling identifiers are not hospital IDs. Source demographic
  categories and age top-coding require explicit harmonization.

## eICU chronic-history correction

The active ICU diagnosis table alone omits many conditions clearly recorded in
structured past history. For the user's patient-has-condition objective, the
recommended eICU **primary** label is active ICD positive **or explicit positive
chronic-history evidence**, retaining the strict ICD-only flag for sensitivity
analysis. The four concepts remain distinct classification tasks. This makes
the ascertainment broader than active-code-only matching and must be reported
as such; it does not eliminate MIMIC/eICU documentation differences.

`cross_dataset/eicu_history_evidence.py` implements exact structured-branch and
picklist rules, with two boundary/unknown unit checks passing. It recognizes:

- Hypertension requiring treatment under Cardiovascular (R).
- Insulin-dependent diabetes and non-insulin-dependent diabetes (both medication
  dependent and non-medication dependent) under Endocrine (R). The structured
  diabetes branch, not medication use alone, supplies the evidence.
- Explicit myocardial-infarction and angina history under Cardiovascular (R).
- Explicit COPD, asthma and restrictive pulmonary disease under Pulmonary.

PCI/CABG alone, home oxygen, pulmonary-function tests, transplant, pulmonary
embolism, sarcoidosis, lung cancer and past respiratory-failure episodes are
excluded as proxies, different code families or potentially resolved history.
Unknown, Not Obtainable and No Health Problems items never generate a positive
or a verified disease-negative label. Histories may be documented at any offset;
these are retrospective label evidence, never baseline predictor inputs.

The canonical past-history archive passed its stored source SHA check and was
read completely to gzip EOF: 1,149,180 rows covering 176,258 stays. Detailed
aggregate evidence and every selected structured path/value are saved in
`dataset_audit/eicu_history_evidence_audit.json`.

| Condition | Explicit positive-history stays in all source history | Added positives within the 155,494 active-ICD-documented stays | ICD/history union positives within those stays |
|---|---:|---:|---:|
| Ischemic heart disease | 20,133 | 12,943 | 27,783 |
| Respiratory-system disease | 36,159 | 11,688 | 71,425 |
| Non-gestational diabetes | 52,914 | 40,895 | 47,895 |
| Systemic hypertension | 91,209 | 62,097 | 87,193 |

These are before the adult/24-hour feature eligibility filter and hospital
partition purge. To keep a common documented-control eligibility rule across
all four tasks, the primary common cohort still requires at least one valid
active diagnosis code. Positive history in stays without such documentation is
counted separately (heart 2,822; respiratory 2,677; diabetes 5,844; hypertension
8,918) and is not used to create controls for the other conditions. These
positive-only exclusions and incomplete history are limitations, not evidence
that the excluded patients are disease-free.

The official [eICU past-history documentation](https://eicu.mit.edu/eicutables/pasthistory/)
defines the structured fields and warns that detailed history is not common.
Source presence supports explicit recorded positives; absent history does not
establish a negative diagnosis.

## Cross-hospital meaning and remaining limits

[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) is from Beth Israel
Deaconess Medical Center. [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/)
contains multiple hospitals. Thus eICU can support a hospital-held-out study and
MIMIC can serve as an additional external hospital/data source. A random
patient split within MIMIC alone is not a cross-hospital experiment.

MIMIC [hospital diagnoses](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html)
are billing diagnoses assigned at discharge, with no availability timestamp in
that table. eICU [active diagnoses](https://eicu.mit.edu/eicutables/diagnosis/)
are entered by clinicians during ICU care and may differ from billing; resolved
or ruled-out status is not reliably supplied. These differences create label
and documentation shifts even after code-family harmonization. Cross-source
evaluation should explicitly discuss them and cannot certify clinical safety,
causal fairness or deployment readiness.

The original respiratory mortality cohort uses early respiratory membership
and a later mortality outcome, which is a different task. The current local
`prepare_respiratory_cohort.py` now checks matching APACHE `admissionDx` entry
offset <=1440 minutes, addressing the previously flagged timing route in code.
That script still requires its own rerun/validation before new cohort counts
are asserted. The new respiratory diagnosis-classification task above should
be saved separately so those mortality data and labels are not overwritten.

## Review scope

This independent review read the source documentation, current scripts and
downloaded dictionaries/diagnoses. It wrote this report only. No new patient
cohort, features or model was constructed by this reviewer.

## Independent implementation review

Reviewed `cross_dataset/prepare_four_condition_tables.py` after its metadata
merge and code-dictionary fixes. No remaining blocker was found for diagnosis
label construction: version-specific family matching is explicit, mixed eICU
comma-separated tokens are combined within their original rows, unknown
documentation is excluded, labels are attached through validated encounter
joins, and predictors use an explicit feature allowlist. eICU whole-hospital
partitions purge patients crossing partitions; MIMIC splits group by patient.
These checks concern construction correctness, not predictive performance.

An independent full eICU diagnosis-code audit found 1,689 unique nonempty
normalized tokens. Eleven tokens were absent from the exact MIMIC diagnosis
dictionary but were genuine ICD-9-CM category parents of its codes: `7872`,
`7895`, `2848`, `2554`, `2387`, `5233`, `3230`, `3238`, `2773`, `3234`, `9952`.
The revised implementation accepts dictionary-backed exact codes and category
parents, and counts them separately rather than treating any plausible string
as a valid code. Every one of these parent-containing source rows also contained
an exact recognized code. There were 2,278,713 rows with recognized diagnosis
documentation among 2,710,672 eICU diagnosis rows.

All tokens triggering the four positive families matched exact dictionary
entries; none required parent-code fallback. Independent counts for audit
comparison are:

| Family | Matching code tokens (both code versions can occur in one row) | Source diagnosis rows with at least one match |
|---|---:|---:|
| Ischemic heart disease | 101,428 | 51,369 |
| Systemic hypertension | 162,385 | 81,343 |
| Non-gestational diabetes | 37,892 | 18,996 |
| Broad respiratory-system disease | 826,820 | 444,381 |

These are source-row counts, not ICU-stay positives or final model-table sizes.
No invalid nonnumeric respiratory `J` category token occurred in the actual
source. The implementation nevertheless now requires the two numeric category
digits so an arbitrary lexicographically in-range token cannot trigger it.

## Final actual-export validation

`cross_dataset/validate_four_condition_tables.py` passed on all eight actual
exports. The aggregate report is
`cross_dataset/four_conditions/independent_validation.json`. It independently
recomputed history evidence from the verified original history gzip and checked
the MIMIC admission joins, eICU ICD/history union, strict ICD sensitivity flags,
documented-control eligibility, exact clinical eligibility population, expected
hash partitions, crossing-patient purge, metadata alignment and every baseline
feature against its clinical source table. It also confirmed identical feature
values and row membership across the four tasks within each source, binary
targets, no patient split overlap, and no eICU hospital split overlap. Only the
declared eleven baseline features enter the exported predictor allowlist.

The validated final populations contain **73,990 MIMIC ICU stays** and
**109,378 eICU ICU stays** after their eligibility checks and split purging.
This validates construction consistency, not clinical label equivalence or
cross-hospital predictive performance. No model was trained by this validator.
