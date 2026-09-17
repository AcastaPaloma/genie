// Diagnostic: times the startup phases that are NOT the skeleton geometry.
import {FlyBodyView} from './body';
const out=document.getElementById('result')!,query=new URL(location.href).searchParams;
const directory='/data/full-brain-783/';
const controlWidth=Number(query.get('controlWidth')??640);
const log:Record<string,number>={};
const time=async<T>(name:string,fn:()=>Promise<T>):Promise<T>=>{const t=performance.now();const v=await fn();log[name]=+((performance.now()-t)/1000).toFixed(2);out.textContent=JSON.stringify(log,null,1);return v;};
try{
  // How much of the neural worker's init is pure transport + integrity?
  const grab=async(file:string)=>{const raw=await (await fetch(directory+file)).arrayBuffer();await crypto.subtle.digest('SHA-256',raw);return raw.byteLength;};
  await time('arborFetchAndHash',()=>grab(`whole-arbor-${controlWidth}.bin`));
  await time('connectomeFetchAndHash',()=>grab('connectome.bin.gz'));
  await time('brainJsonFetchParse',async()=>{const raw=await (await fetch(directory+'brain.json')).arrayBuffer();await crypto.subtle.digest('SHA-256',raw);JSON.parse(new TextDecoder().decode(raw));});
  // The real neural worker init, end to end.
  await time('neuralWorkerInit',()=>new Promise<void>((resolve,reject)=>{
    const worker=new Worker(new URL('./full-neural.worker.ts',import.meta.url),{type:'module'});
    worker.onmessage=e=>{if(e.data.type==='error')reject(Error(e.data.message));else if(e.data.type==='ready'){resolve();worker.terminate();}};
    worker.onerror=e=>reject(Error(e.message));
    worker.postMessage({type:'init',base:directory,controlWidth});
  }));
  // The body/physics model.
  await time('bodyLoad',async()=>{const body=new FlyBodyView(document.getElementById('body')!);await body.load(()=>{});});
  out.textContent=JSON.stringify({...log,note:'seconds; arbor/connectome/brainJson rows are transport+integrity only and are included inside neuralWorkerInit'},null,1);
  document.body.dataset.done='true';
}catch(error){out.textContent=JSON.stringify({...log,error:String(error)},null,1);document.body.dataset.error='true';}
