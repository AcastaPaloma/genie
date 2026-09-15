/** Calibrate a bounded local inverse controller using actual LIF responses.
 * Run with Node >= 22.18: node ui/scripts/calibrate_controller.ts
 * This is an experimental system-identification controller, not a published
 * fly visual model. Its output is stimulation rates, never renderer colors.
 */
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {SpikingCircuit} from '../src/neural.ts';

const dir=new URL('../public/data/',import.meta.url);
const meta=JSON.parse(fs.readFileSync(new URL('connectome.json',dir),'utf8'));
const bytes=gunzipSync(fs.readFileSync(new URL('connectome.bin.gz',dir)));
const brain=new SpikingCircuit(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const ob=fs.readFileSync(new URL('image-operator.bin',dir));
const buffer=ob.buffer.slice(ob.byteOffset,ob.byteOffset+ob.byteLength);
const [P,N,E]=new Uint32Array(buffer,0,3);
const pixel=new Uint16Array(buffer,12,E), neuron=new Uint16Array(buffer,12+2*E,E), weight=new Float32Array(buffer,12+4*E,E);
if(N!==meta.renderMap.length)throw Error('Operator and anatomy differ.');
const project=(a:Float32Array)=>{
  const y=new Float64Array(P);
  for(let e=0;e<E;e++)y[pixel[e]]+=weight[e]*a[neuron[e]];
  return y;
};
const groups=meta.inputs.map((n:{u:number;v:number})=>Math.min(7,Math.floor(n.u*8))+8*Math.min(5,Math.floor(n.v*6)));
const unique:number[]=[...new Set<number>(groups)].sort((a,b)=>a-b);
const groupIndex=groups.map((g:number)=>unique.indexOf(g));
const K=unique.length;
const ids=Int32Array.from(meta.inputs,(n:{index:number})=>n.index);
const rates=new Float32Array(ids.length).fill(80);
const run=(ms:number)=>{for(let i=0;i<Math.round(ms/brain.dt);i++)brain.step(ids,rates);};
run(300);
const initial=brain.snapshot();
run(100);
const baseline=project(brain.renderActivity(meta.renderMap));
const response:number[][]=[];
for(let k=0;k<K;k++){
  brain.restore(initial);rates.fill(80);
  for(let i=0;i<rates.length;i++)if(groupIndex[i]===k)rates[i]=160;
  run(100);
  const y=project(brain.renderActivity(meta.renderMap));
  response.push(Array.from(y,(v,p)=>v-baseline[p]));
  console.log(`Calibrated stimulation group ${k+1}/${K}`);
}
const covariance=Array.from({length:K},()=>new Array<number>(K).fill(0));
const ridge=0.00025;
for(let i=0;i<K;i++)for(let j=0;j<K;j++){
  let sum=0;for(let p=0;p<P;p++)sum+=response[i][p]*response[j][p]/P;
  covariance[i][j]=sum+(i===j?ridge:0);
}
const step=1/Math.max(...covariance.map(row=>row.reduce((s,x)=>s+Math.abs(x),0)));
function solve(target:number[]){
  const rhs=response.map(col=>col.reduce((s,v,p)=>s+v*(target[p]-baseline[p])/P,0));
  let u=new Float64Array(K);
  for(let iter=0;iter<250;iter++){
    const next=u.slice();
    for(let k=0;k<K;k++){
      let grad=-rhs[k];for(let j=0;j<K;j++)grad+=covariance[k][j]*u[j];
      next[k]=Math.max(-1,Math.min(1,u[k]-step*grad));
    }
    u=next;
  }
  return u;
}
const targets=JSON.parse(fs.readFileSync(new URL('controller-targets.json',dir),'utf8'));
const evaluation=[];
for(const target of targets){
  const controls=solve(target.pixels);brain.restore(initial);
  for(let i=0;i<rates.length;i++)rates[i]=80*(1+controls[groupIndex[i]]);
  run(100);
  const actual=project(brain.renderActivity(meta.renderMap));
  const mse=actual.reduce((s,x,p)=>s+(x-target.pixels[p])**2,0)/P;
  const baselineMse=baseline.reduce((s,x,p)=>s+(x-target.pixels[p])**2,0)/P;
  const prediction=Array.from(baseline,(x,p)=>x+response.reduce((s,col,k)=>s+col[p]*controls[k],0));
  const record={label:target.label,mse,baselineMse,psnr:-10*Math.log10(mse),controls:Array.from(controls),actual:Array.from(actual),prediction};
  evaluation.push(record);console.log(JSON.stringify({label:target.label,mse,baselineMse,psnr:record.psnr}));
}
fs.writeFileSync(new URL('stimulation-controller.json',dir),JSON.stringify({
  kind:'Local finite-difference inverse stimulation controller',controlCount:K,inputGroup:groupIndex,
  baseRateHz:80,deltaRateHz:80,resolution:[40,30],calibrationWarmupMs:300,calibrationHorizonMs:100,
  baseline:Array.from(baseline),response,covariance,step,ridge,
  limitation:'Calibrated around one neural state. Full nonlinear LIF validation below is authoritative; predictions may be inaccurate. Optical input and DN-body adapters are engineered.',
  evaluation,
}));
