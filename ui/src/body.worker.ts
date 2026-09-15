import {loadScene} from './vendor/flygym-scene';
import {Controller} from './vendor/flygym-controller';

let physics:Awaited<ReturnType<typeof loadScene>>,controller:Controller;
let busy=false;
let native:{base:string;session:string;nu:number;timestep:number}|null=null;
async function hash(raw:ArrayBuffer){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),v=>v.toString(16).padStart(2,'0')).join('');}
async function nativeState(action:string,data:ArrayBuffer|undefined,epoch:number,type:string,start:number){
  const response=await fetch(`${native!.base}/session/${native!.session}/${action}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:data});
  if(!response.ok)throw Error(`Native body failed: ${await response.text()}`);
  const values=new Float64Array(await response.arrayBuffer()),[time,physicsMs,nq,ngeom]=values;
  if(values.length!==4+nq+12*ngeom)throw Error('Native body dimensions differ.');
  const qpos=values.slice(4,4+nq),positions=values.slice(4+nq,4+nq+3*ngeom),rotations=values.slice(4+nq+3*ngeom);
  self.postMessage({type,epoch,time,qpos,positions,rotations,physicsMs,computeMs:performance.now()-start},{transfer:[qpos.buffer,positions.buffer,rotations.buffer]});
}
self.onmessage=async event=>{
  const m=event.data;
  try{
    if(m.type==='init'){
      if(m.native){
        let info:any=null;
        try{const r=await fetch(m.native+'/info',{signal:AbortSignal.timeout(1500)});if(r.ok)info=await r.json();}catch{/* The static application can still run its explicit WASM backend. */}
        if(info){
          const [xml,raw]=await Promise.all([fetch(m.assets+'/model/fly.xml').then(r=>r.arrayBuffer()),fetch(m.assets+'/model_meta.json').then(r=>r.arrayBuffer())]);
          if(info.version!=='3.9.0'||info.timestep!==m.timestep||await hash(xml)!==info.modelSha256||await hash(raw)!==info.metadataSha256)throw Error('Native physics version, model or timestep differs from the browser model.');
          const meta=JSON.parse(new TextDecoder().decode(raw));meta.timestep=m.timestep;controller=new Controller(meta);
          const r=await fetch(m.native+'/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({timestep:m.timestep})});if(!r.ok)throw Error('Could not create an independent native body session.');
          const session=await r.json();native={base:m.native,session:session.session,nu:session.nu,timestep:m.timestep};
          self.postMessage({type:'ready',backend:'Native MuJoCo 3.9.0 · 0.25 ms timestep'});return;
        }
      }
      await import(/* @vite-ignore */m.runtime);
      physics=await loadScene({assetsDir:m.assets});
      const options=physics.model.opt;options.timestep=m.timestep;physics.model.opt=options;
      physics.meta.timestep=m.timestep;controller=new Controller(physics.meta);
      self.postMessage({type:'ready',backend:`MuJoCo ${physics.mj.mj_versionString()} WASM · ${m.timestep*1000} ms timestep`});
    }else if(m.type==='reset'){
      if(native){controller.reset();await nativeState('reset',undefined,m.epoch,'reset',performance.now());return;}
      physics.mj.mj_resetDataKeyframe(physics.model,physics.data,0);physics.mj.mj_forward(physics.model,physics.data);controller.reset();sendState('reset',m.epoch,0);
    }else if(m.type==='step'){
      if(busy)throw Error('Body work cannot overlap.');busy=true;
      if(native){
        if(m.ms!==100)throw Error('Native body frames must span exactly 100 ms.');
        const start=performance.now(),steps=Math.round(m.ms/(native.timestep*1000)),controls=new Float64Array(steps*native.nu);
        for(let i=0;i<steps;i++)controller.stepCPG(controls.subarray(i*native.nu,(i+1)*native.nu),m.left,m.right);
        await nativeState('step',controls.buffer,m.epoch,'state',start);busy=false;return;
      }
      const start=performance.now(),before=physics.data.time;
      const steps=Math.round(m.ms/(physics.meta.timestep*1000));
      for(let i=0;i<steps;i++){controller.stepCPG(physics.data.ctrl,m.left,m.right);physics.mj.mj_step(physics.model,physics.data);}
      if(Math.abs(physics.data.time-before-m.ms/1000)>1e-6)throw Error('Body timestep did not advance the requested time.');
      if(!Array.from(physics.data.qpos).every(x=>Number.isFinite(x)))throw Error('The body physics became unstable.');
      busy=false;sendState('state',m.epoch,performance.now()-start);
    }
  }catch(error){busy=false;self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
function sendState(type:string,epoch:number,computeMs:number){
  const qpos=Float64Array.from(physics.data.qpos),positions=Float64Array.from(physics.data.geom_xpos),rotations=Float64Array.from(physics.data.geom_xmat);
  self.postMessage({type,epoch,computeMs,time:physics.data.time,qpos,positions,rotations},{transfer:[qpos.buffer,positions.buffer,rotations.buffer]});
}
