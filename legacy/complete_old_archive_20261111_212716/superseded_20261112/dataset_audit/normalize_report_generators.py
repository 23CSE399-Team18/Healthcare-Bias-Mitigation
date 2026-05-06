"""Keep report refreshes consistent with the organized stage folders."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
for name in ('latent/code/run_latent.py','preprocessing/code/audit_and_prepare.py','inprocessing/code/make_report.py'):
    p=ROOT/name;t=p.read_text(encoding='utf-8')
    lines=[]
    for line in t.splitlines():
        if line.strip().startswith(("cards=''.join(","cards = ''.join(","(HERE/'index.html').write_text(","(OUT / 'index.html').write_text(","links=[f'<img src=", "cards.append(f'<section>")):continue
        lines.append(line)
    t='\n'.join(lines)+'\n'
    if name.startswith('inprocessing'):
        t=t.replace('[plots](index.html) and [all method metrics](outputs/method_comparison.csv)','[plots](../plots/index.html) and [all method metrics](method_comparison.csv)')
    elif name.startswith('latent'):
        t=t.replace('[PLAN.md](PLAN.md)','[PLAN.md](../code/PLAN.md)')
        t=t.replace("    log('Latent stage completed;", "    sys.path.insert(0,str(ROOT/'dataset_audit'))\n    from publish_layout import main as publish_layout\n    publish_layout()\n    log('Latent stage completed;")
    else:
        t=t.replace('[Open current plots](../verified_baseline/index.html) and [the file guide](../verified_baseline/README.md).','[Open current plots](../../plots/index.html) and [the file guide](../README.md).')
        t=t.replace('[the archive manifest](../../dataset_audit/archive_manifest.json)','[the archive manifest](../../../dataset_audit/archive_manifest.json)')
        t=t.replace('`inprocessing/REPORT.md`','`inprocessing/outputs/REPORT.md`')
    ast.parse(t);p.write_text(t,encoding='utf-8')
p=ROOT/'dataset_audit/publish_current_overview.py'
p.write_text('"""Compatibility entry point for the current dashboard publisher."""\nfrom publish_layout import main\nif __name__=="__main__":main()\n',encoding='utf-8')
print('All report generators now preserve the code/outputs/plots layout.',flush=True)
