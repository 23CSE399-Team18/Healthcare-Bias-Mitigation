"""Expose the existing verified preprocessing artifacts without retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'preprocessing/outputs/verified_baseline'
names = ('heart', 'respiratory', 'diabetes', 'hypertension', 'mimic')
splits = ('train', 'validation', 'test')
source = ROOT / 'preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu'
features = json.loads((source / 'feature_names.json').read_text())['output_features']
exports = {}
for split in splits:
    with np.load(source / f'X_{split}.npz') as a:
        x = a['X']
    cols = pd.read_csv(source / f'rows_{split}.csv', nrows=0).columns
    rows = pd.read_csv(source / f'rows_{split}.csv', dtype={c:'string' for c in cols if c.startswith('meta_')})
    frame = pd.concat([pd.DataFrame(x, columns=features), rows], axis=1)
    target = BASE / 'mimic' / f'mimic_{split}_preprocessed.csv'
    frame.to_csv(target, index=False)
    reread = pd.read_csv(target, dtype={c:'string' for c in cols if c.startswith('meta_')})
    np.testing.assert_allclose(reread[features], x, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(reread[list(rows)], rows, check_dtype=False)
    with np.load(source / f'y_{split}.npz') as labels:
        for task in labels.files:
            np.testing.assert_array_equal(reread['target_'+task], labels[task])
    exports[split] = {'rows':len(frame), 'features':len(features), 'csv':str(target.relative_to(ROOT)),
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(BASE/'mimic/csv_validation.json').write_text(json.dumps({'passed':True,'source':'existing verified mimic_to_eicu arrays','partitions':exports},indent=2),encoding='utf-8')

import sys
sys.path.insert(0,str(ROOT/'dataset_audit'))
from publish_layout import main as publish_layout
publish_layout()
