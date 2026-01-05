const fs=require('fs'),path=require('path'),zlib=require('zlib'),crypto=require('crypto');
const {pipeline}=require('stream/promises');
const root=path.resolve(__dirname,'../..');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'PUBLICATION_MANIFEST_20261114.json'),'utf8'));
async function hash(p){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(p))h.update(b);return h.digest('hex')}
(async()=>{for(const f of manifest.files.filter(f=>f.action==='gzip-lossless')){const src=path.resolve(root,f.destination),dest=path.resolve(root,f.source);if(!src.startsWith(root+path.sep)||!dest.startsWith(root+path.sep))throw Error('Invalid manifest path');if(fs.existsSync(dest)){if(await hash(dest)!==f.sourceSha256)throw Error('Existing file differs: '+f.source);console.log('Already restored '+f.source);continue;}if(await hash(src)!==f.sha256)throw Error('Archive checksum mismatch');await pipeline(fs.createReadStream(src),zlib.createGunzip(),fs.createWriteStream(dest,{flags:'wx'}));if(await hash(dest)!==f.sourceSha256)throw Error('Restored checksum mismatch');console.log('Restored '+f.source)}})().catch(e=>{console.error(e);process.exitCode=1});
