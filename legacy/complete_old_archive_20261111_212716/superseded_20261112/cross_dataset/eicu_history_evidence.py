"""Explicit chronic-condition positives from eICU structured past history.

History supplies positive evidence only. Absence of a selected item, No Health
Problems, and Not Obtainable never create a disease-negative label. The caller
must define its own documented-control eligibility. This helper does not infer
disease from treatments, procedures, organ-system headings or free text.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "_data_reader"))
import pandas as pd

CONDITIONS = ("heart", "respiratory", "diabetes", "hypertension")
BASE = ("notes", "progress notes", "past history", "organ systems")


def normalized(value):
    return " ".join(str(value or "").strip().casefold().split())


# Exact structured branches and positive picklist values observed in the
# official source. Each selected history concept maps to the matching disease
# family, not a treatment proxy or a resolved nonspecific symptom.
RULES = {
    ("cardiovascular (r)", "hypertension requiring treatment"): (
        "hypertension", {"hypertension requiring treatment"}),
    ("endocrine (r)", "insulin dependent diabetes"): (
        "diabetes", {"insulin dependent diabetes"}),
    ("endocrine (r)", "non-insulin dependent diabetes"): (
        "diabetes", {"medication dependent", "non-medication dependent"}),
    ("cardiovascular (r)", "myocardial infarction"): (
        "heart", {"mi - date unknown", "mi - remote", "mi - within 6 months",
                  "mi - within 2 years", "mi - within 5 years", "multiple"}),
    ("cardiovascular (r)", "angina"): (
        "heart", {"angina", "angina - severity unknown", "angina - class i",
                  "angina - class ii", "angina - class iii", "angina - class iv"}),
    ("pulmonary", "copd"): (
        "respiratory", {"copd - moderate", "copd - no limitations", "copd - severe"}),
    ("pulmonary", "asthma"): ("respiratory", {"asthma"}),
    ("pulmonary", "restrictive disease"): (
        "respiratory", {"restrictive pulmonary disease"}),
}
UNKNOWN_VALUES = {"", "not obtainable", "unknown", "no health problems",
                  "not performed", "none", "no", "negative"}


def history_condition(path, value, value_text=""):
    """Return a selected condition for an exact positive structured item only."""
    parts = tuple(normalized(part) for part in str(path or "").split("/"))
    if len(parts) < 7 or parts[:4] != BASE:
        return None
    rule = RULES.get(parts[4:6])
    if rule is None:
        return None
    selected = normalized(value) or normalized(value_text)
    if selected in UNKNOWN_VALUES or normalized(value_text) in (UNKNOWN_VALUES - {""}):
        return None
    condition, positive_values = rule
    # Checking both the branch leaf and selected value prevents a negated or
    # unknown leaf under a disease heading from becoming positive.
    return condition if selected in positive_values and parts[-1] in positive_values else None


def load_eicu_history_evidence(path=None):
    """Return (stay-level positive-history DataFrame, aggregate audit).

    The DataFrame contains every stay represented in pastHistory, its
    `history_rows` count, and integer `history_<condition>` flags. A zero is
    absence of selected positive history evidence, never disease-negative
    ground truth. The default source is the verified canonical eICU table.
    """
    verified_default_source = path is None
    if path is None:
        sys.path.insert(0, str(PROJECT / "resp_physionet" / "resp"))
        from eicu_sources import get_eicu_table_path
        path = get_eicu_table_path("pastHistory", verify=True)
    path = Path(path)
    rows_by_stay = Counter()
    positives = {condition: set() for condition in CONDITIONS}
    positive_rows = Counter()
    selected_items = Counter()
    all_rows = 0
    invalid_stay_rows = 0
    unknown_value_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            all_rows += 1
            stay_text = str(row["patientunitstayid"]).strip()
            if not stay_text.isdigit() or int(stay_text) <= 0:
                invalid_stay_rows += 1
                continue
            stay = int(stay_text)
            rows_by_stay[stay] += 1
            if normalized(row.get("pasthistoryvalue")) in UNKNOWN_VALUES:
                unknown_value_rows += 1
            condition = history_condition(row.get("pasthistorypath"),
                                          row.get("pasthistoryvalue"),
                                          row.get("pasthistoryvaluetext"))
            if condition:
                positives[condition].add(stay)
                positive_rows[condition] += 1
                selected_items[(condition, row["pasthistorypath"], row["pasthistoryvalue"])] += 1
    frame = pd.DataFrame({"patientunitstayid": sorted(rows_by_stay)})
    frame["history_rows"] = frame.patientunitstayid.map(rows_by_stay).astype("int32")
    for condition in CONDITIONS:
        frame["history_" + condition] = frame.patientunitstayid.isin(positives[condition]).astype("int8")
    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {
        "source_file": str(path), "source_sha256": source_hash,
        "canonical_source_verified": verified_default_source,
        "source_gzip_read_to_eof": True, "source_rows": all_rows,
        "invalid_stay_identifier_rows": invalid_stay_rows,
        "stays_with_any_history": len(frame), "unknown_or_nonspecific_value_rows": unknown_value_rows,
        "positive_history_rows": {condition: positive_rows[condition] for condition in CONDITIONS},
        "positive_history_stays": {condition: len(positives[condition]) for condition in CONDITIONS},
        "selected_structured_items": [
            {"condition": condition, "path": path_name, "value": value, "rows": count}
            for (condition, path_name, value), count in sorted(selected_items.items())],
        "timing": "Structured chronic past-history evidence documented at any offset; retrospective label evidence only",
        "negative_labels_generated": False,
        "excluded_proxies": ["PCI/CABG alone", "Home oxygen", "Pulmonary function tests",
                             "Lung transplant", "Pulmonary embolism", "Sarcoidosis", "Lung cancer",
                             "Past respiratory-failure episodes"],
        "limitation": "Past-history documentation is incomplete; history flags identify explicit positives only",
    }
    return frame, audit
