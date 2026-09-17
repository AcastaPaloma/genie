export {};
// Diagnostic: does moving chunk transport + SHA-256 into workers beat the
// current main-thread serial loop? Integrity is still checked on every chunk.
const out=document.getElementById('result')!,query=new URL(location.href).searchParams;
const directory='/data/full-brain-783/';
const n=Number(query.get('chunks')??60);
const source=`self.onmessage=async e=>{
  for(const f of e.data.files){const raw=await (await fetch(e.data.dir+f)).arrayBuffer();await crypto.subtle.digest('SHA-256',raw);}
  self.postMessage('done');};`;
const blob=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
try{
  const meta=await (await fetch(directory+'brain.json')).json();
  const files=meta.chunks.slice(0,n).map((c:{file:string})=>c.file);
  const scale=meta.chunks.length/files.length/1000;
  const results:Record<string,number>={};
  for(const workers of [1,2,4,8]){
    const t=performance.now();
    await Promise.all(Array.from({length:workers},(_,w)=>new Promise<void>((resolve,reject)=>{
      const worker=new Worker(blob);
      worker.onmessage=()=>{worker.terminate();resolve();};
      worker.onerror=e=>reject(Error(e.message));
      worker.postMessage({dir:new URL(directory,location.href).href,files:files.filter((_:string,i:number)=>i%workers===w)});
    })));
    results[`workers${workers}`]=+((performance.now()-t)*scale).toFixed(1);
    out.textContent=JSON.stringify(results,null,1);
  }
  out.textContent=JSON.stringify({note:'projected seconds for all 270 chunks: fetch + SHA-256 only, no GPU upload',mainThreadSerialSeconds:12.1,...results},null,1);
  document.body.dataset.done='true';
}catch(error){out.textContent=String(error);document.body.dataset.error='true';}
