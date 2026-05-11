"""Run the active scientific validators without fitting replacement models."""
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
CHECKS=[
 ('preprocessing','preprocessing/code/confirm_saved_outputs.py'),
 ('mimic_eicu_preprocessing','preprocessing/code/audit_saved_preprocessing.py'),
 ('latent','latent/code/validate_latent.py'),
 ('hospital','latent/code/validate_hospital_latent.py'),
 ('latent_model_contracts','latent/code/test_latent.py'),
 ('inprocessing_model_contracts','inprocessing/code/test_core.py'),
 ('inprocessing','inprocessing/code/validate.py'),
 ('prediction_exports','inprocessing/code/check_exports.py'),
]

if __name__=='__main__':
    out=ROOT/'dataset_audit/active_recheck';out.mkdir(exist_ok=True)
    report={'passed':False,'started_utc':datetime.now(timezone.utc).isoformat(),'checks':[],'model_refitting':False}
    for name,script in CHECKS:
        print('RUN '+name,flush=True);start=time.monotonic()
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,script],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record={'name':name,'script':script,'passed':result.returncode==0,'seconds':round(time.monotonic()-start,2),'log':name+'.log'}
        report['checks'].append(record)
        (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(('PASS ' if record['passed'] else 'FAIL ')+name,flush=True)
        if not record['passed']:
            print((out/(name+'.log')).read_text()[-5000:],flush=True)
            raise SystemExit(result.returncode)
    report.update(passed=True,completed_utc=datetime.now(timezone.utc).isoformat())
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
