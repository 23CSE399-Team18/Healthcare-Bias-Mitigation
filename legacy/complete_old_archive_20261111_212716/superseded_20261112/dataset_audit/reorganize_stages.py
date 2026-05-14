"""One-time, hash-recorded file relocation; never retrain or rewrite frozen choices."""
from pathlib import Path
import hashlib,json,re,os

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'dataset_audit/layout_migration.json'
PREFIXES={
 'preprocessing/verified_baseline':'preprocessing/outputs/verified_baseline',
 'preprocessing/hospital_baseline':'preprocessing/outputs/hospital_baseline',
 'preprocessing/audit_current':'preprocessing/outputs/audit_current',
 'latent/hospital_outputs':'latent/outputs/hospital',
 'latent/work':'latent/code/references',
 'cross_dataset/four_conditions/baseline_preprocessed':'preprocessing/outputs/cross_dataset_baseline',
}

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def mapped(s):
    s=s.replace('\\','/')
    for old,new in sorted(PREFIXES.items(),key=lambda x:-len(x[0])):
        if s==old or s.startswith(old+'/'):return new+s[len(old):]
    return s

def main():
    if MANIFEST.exists():raise RuntimeError('Migration already recorded; do not run twice.')
    paths={}
    for stage in ('preprocessing','latent','inprocessing'):
        for p in (ROOT/stage).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            old=p.relative_to(ROOT).as_posix();new=mapped(old)
            if p.parent==ROOT/stage:
                if p.suffix in ('.py','.ipynb') or p.name in ('PLAN.md','requirements.txt'):new=f'{stage}/code/{p.name}'
                elif p.name in ('REPORT.md','environment.json','progress.log'):new=f'{stage}/outputs/{p.name}'
            if p.suffix=='.png' and '/work/' not in old:
                if old.startswith('preprocessing/verified_baseline/'):new=old.replace('preprocessing/verified_baseline/','preprocessing/plots/baseline/',1)
                elif old.startswith('preprocessing/hospital_baseline/'):new=old.replace('preprocessing/hospital_baseline/','preprocessing/plots/hospital/',1)
                elif old.startswith('latent/hospital_outputs/'):new=old.replace('latent/hospital_outputs/','latent/plots/hospital/',1)
                elif f'{stage}/outputs/' in old:new=old.replace(f'{stage}/outputs/',f'{stage}/plots/',1)
            if old=='preprocessing/verified_baseline/index.html':new='preprocessing/plots/index.html'
            if old=='latent/hospital_outputs/index.html':new='latent/plots/hospital/index.html'
            if new!=old:paths[old]=new
    for p in (ROOT/'cross_dataset/four_conditions/baseline_preprocessed').rglob('*'):
        if p.is_file():paths[p.relative_to(ROOT).as_posix()]=mapped(p.relative_to(ROOT).as_posix())
    for filename in ('prepare_four_condition_baseline.py','audit_saved_preprocessing.py'):
        paths['cross_dataset/'+filename]='preprocessing/code/'+filename
    for filename in ('build_hospital_latent.py','validate_hospital_latent.py'):
        paths['cross_dataset/'+filename]='latent/code/'+filename
    assert len(set(paths.values()))==len(paths)
    records=[]
    # Resolve every source and target before any move; all remain within D:/complete.
    for old,new in paths.items():
        source=(ROOT/old).resolve();dest=(ROOT/new).resolve()
        assert source.is_relative_to(ROOT) and dest.is_relative_to(ROOT)
        assert source.is_file() and not dest.exists(),(old,new)
    frozen={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'inprocessing/outputs').rglob('selection_frozen.json')}
    for old,new in paths.items():
        source=ROOT/old;dest=ROOT/new;before=digest(source)
        dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
        assert digest(dest)==before
        records.append({'old':old,'new':new,'sha256_before':before,'move_hash_verified':True})
    # Text references are relocated without touching numeric data, model files,
    # probabilities, or the immutable validation selection records.
    def replace_path(value):
        norm=value.replace('\\','/')
        absolute=norm.lower().startswith(ROOT.as_posix().lower()+'/')
        rel=norm[len(ROOT.as_posix())+1:] if absolute else norm
        dest=paths.get(rel,mapped(rel))
        if dest==rel:return value
        return str(ROOT/dest) if absolute else (dest.replace('/','\\') if '\\' in value else dest)
    def convert(obj):
        if isinstance(obj,dict):return {replace_path(k):convert(v) for k,v in obj.items()}
        if isinstance(obj,list):return [convert(v) for v in obj]
        if isinstance(obj,str):return replace_path(obj)
        return obj
    def code_text(text):
        for old,new in sorted(paths.items(),key=lambda x:-len(x[0])):
            if old.endswith(('.py','.ipynb','.png')):
                text=text.replace(old,new).replace(old.replace('/','\\\\'),new.replace('/','\\\\'))
        for old,new in sorted(PREFIXES.items(),key=lambda x:-len(x[0])):
            text=text.replace(old,new).replace(old.replace('/','\\\\'),new.replace('/','\\\\'))
        text=text.replace('(Path.cwd(), Path.cwd().parent)','(Path.cwd(), *Path.cwd().parents)')
        text=text.replace('(Path.cwd(),Path.cwd().parent)','(Path.cwd(),*Path.cwd().parents)')
        return text
    inverse={v:k for k,v in paths.items()}
    def links(text,current):
        old=ROOT/inverse.get(current.relative_to(ROOT).as_posix(),current.relative_to(ROOT).as_posix())
        def target(link):
            if re.match(r'^[a-zA-Z]+:|^#',link):return link
            try:rel=(old.parent/link).resolve().relative_to(ROOT).as_posix()
            except ValueError:return link
            new=ROOT/paths.get(rel,mapped(rel))
            return os.path.relpath(new,current.parent).replace('\\','/')
        text=re.sub(r'(?P<lead>(?:src|href)=")(?P<link>[^"]+)(?P<tail>")',lambda m:m['lead']+target(m['link'])+m['tail'],text)
        text=re.sub(r'(\]\()([^\)\n]+)(\))',lambda m:m[1]+target(m[2])+m[3],text)
        return text
    changes=[]
    scopes=[ROOT/s for s in ('preprocessing','latent','inprocessing','cross_dataset','dataset_audit')]
    candidates=[p for folder in scopes for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'README.md',ROOT/'DATASET_STATUS.md',ROOT/'START_HERE.html']
    for p in candidates:
        if p==Path(__file__) or p.name in ('selection_frozen.json','COMPLETE.json') or p.suffix not in ('.py','.md','.html','.json','.ipynb'):continue
        if 'archive' in p.name or p.name=='layout_migration.json':continue
        before=p.read_text(encoding='utf-8-sig');after=before
        if p.suffix=='.json':
            obj=json.loads(before);converted=convert(obj)
            if obj!=converted:after=json.dumps(converted,indent=2)
        elif p.suffix=='.ipynb':
            obj=json.loads(before)
            for cell in obj['cells']:
                source=''.join(cell['source']);updated=code_text(source) if cell['cell_type']=='code' else links(source,p)
                cell['source']=updated.splitlines(keepends=True)
                if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
            after=json.dumps(obj,indent=1,ensure_ascii=False)+'\n'
        elif p.suffix=='.py':
            after=code_text(before)
            if p.relative_to(ROOT).as_posix() in inverse:
                after=after.replace('Path(__file__).resolve().parents[1]','Path(__file__).resolve().parents[2]')
                if p.name in ('prepare_four_condition_baseline.py','audit_saved_preprocessing.py'):
                    after=after.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path(__file__).resolve().parents[2] / "cross_dataset"\nsys.path.insert(0, str(ROOT))')
        else:after=links(before,p)
        if after!=before:
            changes.append({'file':p.relative_to(ROOT).as_posix(),'before_sha256':digest(p)})
            p.write_text(after,encoding='utf-8');changes[-1]['after_sha256']=digest(p)
    for p,expected in frozen.items():assert digest(ROOT/p)==expected
    # Empty-directory cleanup only; no recursive deletion.
    for scope in scopes[:3]+[ROOT/'cross_dataset/four_conditions/baseline_preprocessed']:
        if scope.exists():
            for p in sorted(scope.rglob('*'),key=lambda x:len(x.parts),reverse=True):
                if p.is_dir() and not any(p.iterdir()):p.rmdir()
    MANIFEST.write_text(json.dumps({'passed':True,'file_moves':records,'text_reference_changes':changes,'frozen_selection_hashes':frozen,'frozen_choices_unchanged':True,'prefix_mapping':PREFIXES},indent=2),encoding='utf-8')
    print(f'Moved {len(records)} files with matching hashes; frozen selections unchanged.',flush=True)

if __name__=='__main__':main()
