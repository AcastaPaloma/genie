// Numerical MuJoCo comparison, no browser or UI automation.
import fs from 'node:fs';
import path from 'node:path';
import {Controller} from '../src/vendor/flygym-controller.js';
await import('../public/body/runtime.js');
const mj=await globalThis.neuroframeMujoco,assets=path.resolve('public/body/assets');
const meta=JSON.parse(fs.readFileSync(assets+'/model_meta.json','utf8'));meta.timestep=.00025;
const xml=fs.readFileSync(assets+'/model/fly.xml','utf8');
mj.FS.mkdir('/work');mj.FS.writeFile('/work/fly.xml',xml);
for(const file of new Set([...xml.matchAll(/<mesh[^>]*\bfile="([^"]+)"/g)].map(m=>m[1])))mj.FS.writeFile('/work/'+file,fs.readFileSync(assets+'/model/'+file));
const model=mj.MjModel.from_xml_path('/work/fly.xml'),data=new mj.MjData(model),options=model.opt;options.timestep=.00025;model.opt=options;
mj.mj_resetDataKeyframe(model,data,0);mj.mj_forward(model,data);
const request=async(route,body)=>{const r=await fetch('http://127.0.0.1:8769'+route,{method:'POST',headers:{Origin:'http://127.0.0.1:5173','Content-Type':typeof body==='string'?'application/json':'application/octet-stream'},body});if(!r.ok)throw Error(await r.text());return r;};
const info=await (await request('/session',JSON.stringify({timestep:.00025}))).json();
if(info.version!==mj.mj_versionString()||info.nq!==model.nq||info.ngeom!==model.ngeom||info.nu!==model.nu)throw Error('Physics engine or body structure differs.');
const controller=new Controller(meta),frames=[];
for(let i=0;i<400;i++){
 const phase=i%100,left=i>=300?0:phase<30?1:phase<50?-.7:phase<75?1.2:.15,right=i>=300?0:phase<30?1:phase<50?-.7:phase<75?.15:1.2;
 const controls=new Float64Array(400*model.nu),start=performance.now();
 for(let k=0;k<400;k++){controller.stepCPG(controls.subarray(k*model.nu,(k+1)*model.nu),left,right);data.ctrl.set(controls.subarray(k*model.nu,(k+1)*model.nu));mj.mj_step(model,data);}
 const wasmMs=performance.now()-start,reply=await request('/session/'+info.session+'/step',Buffer.from(controls.buffer));
 const native=new Float64Array(await reply.arrayBuffer()),qpos=Array.from(data.qpos),other=Array.from(native.slice(4,4+model.nq));
 if(Math.abs(native[0]-data.time)>1e-6||!other.every(Number.isFinite))throw Error('Native model time or stability differs.');
 frames.push({frame:i,wasmMs,nativeMs:native[1],wasmZ:qpos[2],nativeZ:other[2],maximumPoseDifference:Math.max(...qpos.map((x,j)=>Math.abs(x-other[j])))});
 if(i%100===99)console.log(JSON.stringify({frames:i+1,wasmMs,nativeMs:native[1],wasmZ:qpos[2],nativeZ:other[2]}));
}
const stats=key=>({min:Math.min(...frames.map(f=>f[key])),max:Math.max(...frames.map(f=>f[key])),mean:frames.reduce((s,f)=>s+f[key],0)/frames.length});
const result={version:info.version,timestep:.00025,frames:frames.length,firstFramePoseDifference:frames[0].maximumPoseDifference,wasmMs:stats('wasmMs'),nativeMs:stats('nativeMs'),wasmZ:stats('wasmZ'),nativeZ:stats('nativeZ'),measurements:frames};
fs.writeFileSync('.cache/full-brain/validation/native-body-comparison.json',JSON.stringify(result));console.log(JSON.stringify({...result,measurements:undefined}));
