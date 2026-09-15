import {chromium} from '@playwright/test';import fs from 'node:fs';import http from 'node:http';
const label=process.argv[2],query=process.argv[3];if(!label||!query)throw Error('label and query required');
// Keep large validation arrays out of browser DOM/layout and CDP snapshots.
const frames=[];const server=http.createServer((req,res)=>{res.setHeader('Access-Control-Allow-Origin','http://127.0.0.1:5173');if(req.method==='OPTIONS'){res.end();return;}if(req.method!=='POST'||req.url!=='/frame'){res.writeHead(404);res.end();return;}const chunks=[];req.on('data',x=>chunks.push(x));req.on('end',()=>{frames.push(JSON.parse(Buffer.concat(chunks)));res.end('ok');});});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const collector='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({headless:true,args:['--use-angle=metal']});
try{
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:5173/gpu-bench.html?'+query+'&collector='+encodeURIComponent(collector));await page.locator('body[data-done],body[data-error]').waitFor({timeout:240000});
 const raw=await page.locator('#result').textContent();if(await page.locator('body').getAttribute('data-error'))throw Error(raw);
 const r=JSON.parse(raw);r.sequence=frames;fs.writeFileSync('.cache/large-control/'+label+'.json',JSON.stringify(r));console.log(JSON.stringify({label,errors,frames:r.sequence.length,meanMse:r.sequence.reduce((s,x)=>s+x.mse,0)/r.sequence.length,meanCompute:r.timings.reduce((a,b)=>a+b)/r.timings.length,p95:r.timings.toSorted((a,b)=>a-b)[Math.floor(r.timings.length*.95)],spikes:r.spikes,maximumDisconnected:r.maximumOutput,diagnostics:r.diagnostics}));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
