"""Recompute eight separate bias dashboards for each current preprocessing branch."""
from pathlib import Path
import runpy, sys
if __name__=='__main__':
    sys.argv=[sys.argv[0],'--stage','preprocessing']
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'dataset_audit/build_bias_dashboards.py'),run_name='__main__')
