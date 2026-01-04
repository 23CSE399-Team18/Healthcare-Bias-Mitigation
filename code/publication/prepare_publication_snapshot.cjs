const fs = require('fs'), path = require('path'), zlib = require('zlib'), crypto = require('crypto');
const {pipeline} = require('stream/promises');
const ROOT='D:/complete', REPO=ROOT+'/.publication/healthcare-scope-method', SITE='D:/Team-18/Team-18';
const manifest=[], omissions=[];
const posix=s=>s.replaceAll('\\','/');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function write(rel,b){const dest=path.join(REPO,rel);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,b);}
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>['.git','.publication','.venv','.claude','__pycache__','.ipynb_checkpoints'].includes(e.name)?[]:e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]);}
function notebook(buf){const n=JSON.parse(buf);for(const c of n.cells){if(c.cell_type==='code'){c.outputs=[];c.execution_count=null;}delete c.attachments;if(c.cell_type==='markdown'){const t=c.source.join('');if(/^### Complete step \d+ details:|^## Raw records, final exported records and record survival/.test(t))c.source=[t.split('\n')[0]+'\n\nRecord-level saved tables remain local. Run with authorized datasets to view them.\n'];}if(c.metadata){delete c.metadata.execution;delete c.metadata.collapsed;delete c.metadata.scrolled;}}delete n.metadata.widgets;return Buffer.from(JSON.stringify(n,null,1)+'\n');}
const note='Public copy: individual clinical records and point-level projections are omitted; aggregate results are retained.';
function sanitizeDashboard(s,rel){
 if(rel.includes('respiratory/')&&!rel.includes('/inprocessing/')&&!rel.endsWith('/index.html')){
  const m=s.match(/const D = (.*);\r?\n/);if(!m)throw Error('Unknown respiratory payload '+rel);
  const d=JSON.parse(m[1]);
  for(const entry of Object.values(d.box||{}))for(const b of Object.values(entry))if(b&&typeof b==='object')b.out=[];
  for(const entry of Object.values(d.qq||{}))for(const k of Object.keys(entry))entry[k]=[];
  d.heat.rows=[];d.heat.empty='Record-level examples are available only in the authorized local version.';d.heat.note=note;
  d.footer=(d.footer||'')+' '+note;
  s=s.replace(m[1],JSON.stringify(d).replaceAll('</','<\\/'));
 }
 if(rel.includes('latent/')&&!rel.endsWith('/index.html')){
  const re=/(<script[^>]*id="payload"[^>]*>)([\s\S]*?)(<\/script>)/,m=s.match(re);if(!m)throw Error('Unknown latent payload '+rel);
  const d=JSON.parse(m[2]);for(const r of Object.values(d.results))r.scatter={x:[],y:[],dataset:[],label:[]};d.footer+=' '+note;
  s=s.replace(re,(_,a,b,c)=>a+JSON.stringify(d).replaceAll('</','<\\/')+c);
  s=s.replace('const xs=sc.x,ys=sc.y;',"const xs=sc.x,ys=sc.y;if(!xs.length){w.insertAdjacentHTML('beforeend','<p class=\"note\">Individual projections are available in the authorized local version.</p>');return;}");
 }
 if(s.includes('<body'))s=s.replace(/(<body[^>]*>)/,'$1<div style="padding:8px 16px;background:#eef1f7;color:#26334a;font:12px sans-serif">'+note+'</div>');
 return Buffer.from(s);
}
function classify(p){
 if(/^(Heart_correct|resp_physionet|\.claude)\//.test(p))return 'environment';
 if(p==='PER_FILE_COMMIT_REPORT.md')return 'historical-publication-report';
 if(p==='PPT_changed_in__Copy__2 (8).pdf')return 'copy';
 if(p.startsWith('output/')){
  if(/\/(01_heart|02_heart|05_hypertension|06_hypertension).*\.png$/.test(p)||p.endsWith('/build_specs.cjs'))return 'copy';
  if(p.includes('/source_data/')&&/\/(heart|hypertension|diabetes)_(sample10_complete_details\.csv|sample10_trace\.json|pipeline_log\.json|inprocessing_summary\.csv)$/.test(p))return 'copy';
  if(p.endsWith('/latent_method_comparison.csv')||/respiratory_(pipeline_log\.json|inprocessing_summary\.csv)$/.test(p))return 'copy';
  return 'restricted-or-mixed-figure-input';
 }
 if(/^latent\/(data\/|latent_scatter\.json)|\/(models\/|encoder\.pt$)|preprocessor\.joblib$/.test(p))return 'restricted-trained-artifact';
 if(/\/(hospital_predictions_|sample10_hospital_predictions|predictions_|sample10_test_predictions)/.test(p))return 'patient-level-model-output';
 if(/^(respiratory|hospital)\/raw\//.test(p)){
  if(/(README[^/]*\.md|LICENSE\.txt|SHA256SUMS\.txt|dictionary\.csv|phenotype_definitions\.json|preprocessing_status\.json|independent_validation\.json|canonical_file_manifest\.csv|respiratory_cohort_24h_flow\.json)$/.test(p))return 'copy';
  return 'restricted-source-records';
 }
 if(/^(respiratory|hospital)\/preprocessed\//.test(p)&&/(_preprocessed\.csv|_feature_space\.csv|sample10)/.test(p))return 'restricted-derived-records';
 if(/^respiratory\/plots_comparison\/S10_/.test(p))return 'restricted-record-image';
 if(p.endsWith('.ipynb'))return 'notebook';
 if(p.includes('/plots/')&&p.endsWith('.html'))return 'dashboard';
 return 'copy';
}
async function copy(src,rel,action='copy'){
 let b=fs.readFileSync(src),sourceHash=sha(b),dest=rel;
 if(action==='notebook')b=notebook(b);
 if(action==='dashboard')b=sanitizeDashboard(b.toString('utf8'),rel);
 if(b.length>99*1024*1024){
  dest=rel+'.gz';fs.mkdirSync(path.dirname(path.join(REPO,dest)),{recursive:true});
  await pipeline(fs.createReadStream(src),zlib.createGzip({level:6}),fs.createWriteStream(path.join(REPO,dest)));
  const gz=fs.readFileSync(path.join(REPO,dest));if(gz.length>99*1024*1024)throw Error('Compressed file still too large: '+dest);
  if(sha(zlib.gunzipSync(gz))!==sourceHash)throw Error('Compression mismatch '+dest);
  manifest.push({source:rel,destination:dest,action:'gzip-lossless',bytes:gz.length,sourceSha256:sourceHash,sha256:sha(gz)});
  console.log('Compressed '+rel+' -> '+Math.round(gz.length/1024/1024)+' MiB');return;
 }
 write(dest,b);manifest.push({source:rel,destination:dest,action,bytes:b.length,sourceSha256:sourceHash,sha256:sha(b)});
}
async function main(){
 for(const src of walk(ROOT)){
  const rel=posix(path.relative(ROOT,src)),action=classify(rel);
  if(!['copy','notebook','dashboard'].includes(action)){omissions.push({path:rel,reason:action});continue;}
  await copy(src,rel,action);
 }
 // The website receives reviewed copies of the same dashboards, never unchecked clinical exports.
 for(const src of walk(SITE)){
  const rel=posix(path.relative(SITE,src));
  if(rel.startsWith('plots/')){
   const m=rel.match(/^plots\/(heart|diabetes|hypertension|respiratory|latent)\/(.*)$/);
   if(m){const p=m[1]+'/plots/'+m[2],prepared=path.join(REPO,p);if(fs.existsSync(prepared)){await copy(prepared,'website/'+rel);continue;}}
   omissions.push({path:'website/'+rel,reason:'unreviewed-site-asset'});continue;
  }
  if(['index.html','start_server.bat','ppt_text.txt'].includes(rel))await copy(src,'website/'+rel);
 }
 // Link every static method-comparison figure referenced by the website.
 for(const ds of ['heart','diabetes','hypertension','respiratory'])for(const src of walk(path.join(REPO,ds,'plots_comparison'))){const rel=posix(path.relative(path.join(REPO,ds,'plots_comparison'),src));await copy(src,'website/plots/'+ds+'/plots_comparison/'+rel);}
 write('PUBLICATION_MANIFEST_20261114.json',JSON.stringify({snapshotDate:'2026-11-14',sourceRoot:'D:/complete',websiteSource:SITE,files:manifest,omissions,excludedDirectories:['.publication','.claude','.venv','__pycache__','.ipynb_checkpoints'],note:'Original files remain local. New commit timestamps are owner-assigned import dates, not evidence of development dates.'},null,2)+'\n');
 console.log(JSON.stringify({included:manifest.length,omitted:omissions.length,bytes:manifest.reduce((s,f)=>s+f.bytes,0)}));
}
main().catch(e=>{console.error(e);process.exit(1)});
