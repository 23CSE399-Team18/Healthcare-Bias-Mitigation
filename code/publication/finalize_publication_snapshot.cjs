const fs=require('fs'),path=require('path'),crypto=require('crypto');
const repo='D:/complete/.publication/healthcare-scope-method',root='D:/complete';
const write=(p,s)=>{fs.mkdirSync(path.dirname(path.join(repo,p)),{recursive:true});fs.writeFileSync(path.join(repo,p),s)};
const manifest=JSON.parse(fs.readFileSync(path.join(repo,'PUBLICATION_MANIFEST_20261114.json'),'utf8'));
for(const ds of ['heart','diabetes','hypertension','respiratory']){
 for(const e of fs.readdirSync(path.join(repo,ds,'plots_comparison'),{withFileTypes:true}))if(e.isFile()){
  const p=ds+'/plots_comparison/'+e.name,b=fs.readFileSync(path.join(repo,p));
  write('website/plots/'+ds+'/plots_comparison/'+e.name,b);
 }
}
write('.gitignore',`# Local software, credentials and publication workspaces
.publication/
.claude/
.codex/
.venv/
venv/
Heart_correct/
resp_physionet/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.env.*
!.env.example
# Credentialed ICU datasets: publish code and approved aggregates only
/respiratory/raw/
/hospital/raw/
/respiratory/preprocessed/*_preprocessed.csv
/respiratory/preprocessed/*_feature_space.csv
/respiratory/preprocessed/sample10*
/hospital/preprocessed/*_preprocessed.csv
**/models/
**/*.pt
**/*.pth
**/*.joblib
**/*.pkl
/latent/data/
/latent/latent_scatter.json
**/inprocessed/*predictions*.csv
/respiratory/plots_comparison/S10_*
/website/plots/respiratory/plots_comparison/S10_*
# Oversized public originals are restored from the committed lossless gzip copies
/heart/raw/source/LLCP2022.XPT
/heart/preprocessed/heart_train_preprocessed.csv
/heart/preprocessed/heart_train_feature_space.csv
*.crdownload
`);
write('PUBLICATION.md',`# Published project snapshot

Snapshot date: 14 November 2026 (Asia/Calcutta). The import preserves existing upstream history and adds one commit per changed file. New author and committer timestamps are assigned across 1 January–14 November 2026 at the repository owner's request. These are import timestamps, **not evidence of when research or development occurred**. The actual publication occurs on the snapshot date.

## Included

- Source code and notebooks for preprocessing, latent alignment and in-processing.
- Public-use CDC BRFSS 2022, NHANES August 2021–August 2023 and NHIS 2023 survey files, plus their local preprocessing exports and source documentation.
- Aggregate result tables, individual bias dashboards, method comparison figures and project documentation.
- The current Team-18 website in website/, with reviewed dashboard copies.

Files above the regular Git file limit are stored as lossless gzip files. Run 'node code/publication/restore_public_data.cjs' to restore them; this checks SHA-256 and never overwrites an existing file. No Git LFS subscription is required for these compressed files.

## Kept local

Credentialed eICU/MIMIC source records, patient-level derivatives, combined latent rows, trained model binaries and record-level prediction exports are excluded. Source access links and extraction code are included. MIMIC-derived datasets and models require the same controlled access as the source: [MIMIC-IV guidance](https://physionet.org/content/mimiciv/3.1/). The [eICU license](https://physionet.org/content/eicu-crd/2.0/) also restricts sharing access.

Executed notebook outputs and record-level appendix tables are removed from publication copies; executable cells are preserved. Public respiratory dashboards omit patient examples, individual outlier values and point-level Q-Q plots. Public latent dashboards omit individual projections. Aggregate metrics are unchanged. Original local artifacts remain intact.

Python environments, caches, private tool settings and nested Git repositories are excluded. Figure drafts containing restricted examples are excluded. The publication manifest records file-level inclusion and exclusions; this is a reproducible project publication, not an unrestricted backup of the entire drive.

## Validation scope

Publication checks cover hashes, compression round trips, notebook code preservation, dashboard payload removal, JavaScript syntax, local website links, sensitive-file exclusions and Git file sizes. The saved scientific results and RECHECK_REPORT are historical outputs, not a fresh training run. The old local Python environment points to an unavailable base interpreter; recreate an environment before executing the pipeline. Existing scripts may require local path configuration. A successful upload does not establish algorithmic correctness or cross-hospital generalization.

Open the site locally with 'node code/publication/serve_website.cjs', then visit http://127.0.0.1:8080. Publishing website source to main does not by itself configure GitHub Pages hosting.
`);
write('DATA_ACCESS.md',`# Dataset access and publication

| Cohort | Source | Repository contents |
|---|---|---|
| Heart | [BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html) | Public survey files and preprocessing exports |
| Diabetes | [NHANES August 2021–August 2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023) | Public questionnaire, examination and laboratory component files and preprocessing exports |
| Hypertension | [NHIS 2023](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) | Public-use sample adult files and preprocessing exports |
| Respiratory | [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) | Acquisition/extraction code, documentation and aggregate reports; credentialed records remain local |
| Cross-hospital | eICU-CRD plus [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) | Cohort code and aggregate reports; credentialed records and derived models remain local |

Four disease tasks use these sources; the cross-hospital cohort is a fifth representation-learning view. Obtain controlled ICU data directly through PhysioNet credentialing, training and the required data use agreement. Place authorized records in the locations documented in the cohort READMEs, configure source paths and execute the pipeline locally. Do not upload those records to this public repository.

Public survey data retain their original source attribution and conditions; the repository's code license does not supersede dataset conditions. Lossless compressed public files are listed in PUBLICATION_MANIFEST_20261114.json and can be restored with code/publication/restore_public_data.cjs.
`);
write('code/publication/restore_public_data.cjs',`const fs=require('fs'),path=require('path'),zlib=require('zlib'),crypto=require('crypto');
const {pipeline}=require('stream/promises');
const root=path.resolve(__dirname,'../..');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'PUBLICATION_MANIFEST_20261114.json'),'utf8'));
async function hash(p){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(p))h.update(b);return h.digest('hex')}
(async()=>{for(const f of manifest.files.filter(f=>f.action==='gzip-lossless')){const src=path.resolve(root,f.destination),dest=path.resolve(root,f.source);if(!src.startsWith(root+path.sep)||!dest.startsWith(root+path.sep))throw Error('Invalid manifest path');if(fs.existsSync(dest)){if(await hash(dest)!==f.sourceSha256)throw Error('Existing file differs: '+f.source);console.log('Already restored '+f.source);continue;}if(await hash(src)!==f.sha256)throw Error('Archive checksum mismatch');await pipeline(fs.createReadStream(src),zlib.createGunzip(),fs.createWriteStream(dest,{flags:'wx'}));if(await hash(dest)!==f.sourceSha256)throw Error('Restored checksum mismatch');console.log('Restored '+f.source)}})().catch(e=>{console.error(e);process.exitCode=1});
`);
write('code/publication/serve_website.cjs',`const http=require('http'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../website');
const types={'.html':'text/html; charset=utf-8','.png':'image/png','.jpeg':'image/jpeg','.jpg':'image/jpeg','.svg':'image/svg+xml','.css':'text/css','.js':'text/javascript','.pdf':'application/pdf'};
http.createServer((req,res)=>{let rel;try{rel=decodeURIComponent(new URL(req.url,'http://localhost').pathname)}catch{res.writeHead(400);return res.end()}
let f=path.resolve(root,'.'+rel);if(f!==root&&!f.startsWith(root+path.sep)){res.writeHead(403);return res.end()}
if(fs.existsSync(f)&&fs.statSync(f).isDirectory())f=path.join(f,'index.html');if(!fs.existsSync(f)||!fs.statSync(f).isFile()){res.writeHead(404);return res.end('Not found')}
res.setHeader('Content-Type',types[path.extname(f)]||'application/octet-stream');fs.createReadStream(f).pipe(res)}).listen(8080,'127.0.0.1',()=>console.log('Website: http://127.0.0.1:8080'));
`);
write('website/start_server.bat','@echo off\r\nnode "%~dp0..\\code\\publication\\serve_website.cjs"\r\npause\r\n');
let readme=fs.readFileSync(path.join(repo,'README.md'),'utf8');
readme+='\n## Publication and project progress\n\nThe repository contains the saved implementation and outputs for preprocessing, latent alignment and in-processing. See [publication scope](PUBLICATION.md), [dataset access](DATA_ACCESS.md), and the [website](website/index.html). Run `node code/publication/serve_website.cjs` to view the site locally.\n\n- **Completed artifacts:** cohort preparation scripts, stage notebooks, saved aggregate comparisons, individual bias dashboards and website source are present.\n- **Ongoing:** independent scientific review of method implementations, leakage controls and interpretation of external-hospital results. Publication checks do not certify these claims.\n- **Planned:** recreate a working environment, reproduce training from authorized sources, and document final independently verified results.\n\nLarge public files use lossless gzip; run `node code/publication/restore_public_data.cjs` before executing scripts that expect uncompressed files. Restricted ICU records and patient-level derivatives remain local. Published notebooks omit saved outputs; run them locally with authorized data.\n';
write('README.md',readme);
// Preserve the previous scope document separately; current main README describes the actual source snapshot.
const cp=require('child_process');write('docs/PROJECT_SCOPE_AND_METHOD.md',cp.execFileSync('git',['show','origin/main:README.md'],{cwd:repo}));
fs.copyFileSync(__filename,path.join(repo,'code/publication/finalize_publication_snapshot.cjs'));
fs.copyFileSync(root+'/.publication/prepare_complete_publication.cjs',path.join(repo,'code/publication/prepare_publication_snapshot.cjs'));
console.log('Added publication policy, restore tools, website server, and progress documentation.');
