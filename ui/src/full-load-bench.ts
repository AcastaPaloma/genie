// Diagnostic: splits the complete-anatomy load into transport, integrity and GPU cost.
// It loads a prefix of the released chunks only, so it never claims a complete upload.
import {FullBrainView,type FullBrainMetadata} from './full-brain';
const out=document.getElementById('result')!,query=new URL(location.href).searchParams;
const directory='/data/full-brain-783/';
try{
  const chunkCount=Number(query.get('chunks')??20);
  const meta:FullBrainMetadata=await (await fetch(directory+'brain.json')).json();
  const chunks=meta.chunks.slice(0,chunkCount);
  // Pass A: transport + integrity only, no GPU work.
  let fetchMs=0,hashMs=0,bytes=0;
  for(const chunk of chunks){
    let t=performance.now();
    const raw=await (await fetch(directory+chunk.file)).arrayBuffer();
    fetchMs+=performance.now()-t;bytes+=raw.byteLength;
    t=performance.now();
    await crypto.subtle.digest('SHA-256',raw);
    hashMs+=performance.now()-t;
  }
  // Pass B: the real load path over the same prefix (fetch + hash + build + upload + draw).
  const view=new FullBrainView(document.getElementById('brain')!,{...meta,chunks});
  const marks:number[]=[];const started=performance.now();
  await view.load(directory,()=>marks.push(performance.now())).catch((error:Error)=>{
    if(!/counts do not match/.test(error.message))throw error;  // expected: this is a prefix, not the whole brain
  });
  const totalMs=performance.now()-started;
  let cacheMs=0;
  if(query.has('cache')){const t=performance.now();await view.loadLightCache(directory);cacheMs=performance.now()-t;}
  const project=(v:number)=>+(v/chunks.length*meta.chunks.length/1000).toFixed(1);
  // FullBrainView.load now fetches and verifies in workers while the main thread
  // uploads, so the pooled total can no longer be split into the serial phases.
  out.textContent=JSON.stringify({
    sampledChunks:chunks.length,totalChunks:meta.chunks.length,mbPerChunk:+(bytes/chunks.length/1e6).toFixed(1),
    projectedSeconds:{serialMainThreadFetchAndVerify:project(fetchMs+hashMs),pooledLoadIncludingGpuUpload:project(totalMs)},
    lightCacheSeconds:+(cacheMs/1000).toFixed(1),
    firstChunkMs:+(marks[0]-started).toFixed(1),lastChunkMs:+(marks.at(-1)!-marks.at(-2)!).toFixed(1),
  },null,1);
  view.dispose();
  document.body.dataset.done='true';
}catch(error){out.textContent=String(error);document.body.dataset.error='true';}
