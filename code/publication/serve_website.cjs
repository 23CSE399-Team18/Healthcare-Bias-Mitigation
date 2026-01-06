const http=require('http'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../website');
const types={'.html':'text/html; charset=utf-8','.png':'image/png','.jpeg':'image/jpeg','.jpg':'image/jpeg','.svg':'image/svg+xml','.css':'text/css','.js':'text/javascript','.pdf':'application/pdf'};
http.createServer((req,res)=>{let rel;try{rel=decodeURIComponent(new URL(req.url,'http://localhost').pathname)}catch{res.writeHead(400);return res.end()}
let f=path.resolve(root,'.'+rel);if(f!==root&&!f.startsWith(root+path.sep)){res.writeHead(403);return res.end()}
if(fs.existsSync(f)&&fs.statSync(f).isDirectory())f=path.join(f,'index.html');if(!fs.existsSync(f)||!fs.statSync(f).isFile()){res.writeHead(404);return res.end('Not found')}
res.setHeader('Content-Type',types[path.extname(f)]||'application/octet-stream');fs.createReadStream(f).pipe(res)}).listen(8080,'127.0.0.1',()=>console.log('Website: http://127.0.0.1:8080'));
