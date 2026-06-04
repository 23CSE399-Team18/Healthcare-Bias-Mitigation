from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0,str(ROOT/'dataset_audit'))
from publish_layout import main as publish_layout
publish_layout()
