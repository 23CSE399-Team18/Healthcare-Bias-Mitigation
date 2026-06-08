"""Load selected latent inputs without accidentally using labels or metadata as X."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'outputs'
DATASETS=('heart','respiratory','diabetes','hypertension','mimic')


def load_latent(dataset, split='train', task=None):
    if dataset not in DATASETS or split not in ('train','validation','test'):
        raise ValueError('Choose a known dataset and train/validation/test split.')
    manifest=json.loads((BASE/dataset/'manifest.json').read_text())
    if dataset=='mimic':
        if task not in ('heart','respiratory','diabetes','hypertension'):
            raise ValueError('MIMIC requires an explicit task: heart, respiratory, diabetes or hypertension.')
    elif task is not None and task!=dataset:
        raise ValueError('The original datasets each have one specific target.')
    else:
        task=dataset
    with np.load(BASE/dataset/'selected'/f'Z_{split}.npz',allow_pickle=False) as archive:
        x=archive['Z']
    with np.load(BASE/dataset/f'y_{split}.npz',allow_pickle=False) as archive:
        y=archive[task]
    row_path=BASE/dataset/f'rows_{split}.csv'
    cols=pd.read_csv(row_path,nrows=0).columns
    # Preserve identifier strings; weights remain numeric for explicit downstream use.
    id_columns=[c for c in cols if c in ('meta_record_id','meta_patientunitstayid',
        'meta_split_group_uniquepid','meta_patient_id','meta_stay_id')]
    rows=pd.read_csv(row_path,dtype={c:'string' for c in id_columns})
    metadata=rows[[c for c in rows if c.startswith(('meta_','sens_'))]].copy()
    assert len(x)==len(y)==len(metadata) and np.isfinite(x).all()
    assert x.shape[1]==manifest['selection']['latent_dimensions']
    assert metadata.meta_split.eq(split).all()
    return x,y,metadata
