/** Bounded experiment: presynaptic feedback through the unmodified full graph. */
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {SpikingCircuit} from '../src/neural.ts';

const root=new URL('../',import.meta.url),dir=new URL('.cache/control-goal/',root);
const meta=JSON.parse(fs.readFileSync(new URL(`presynaptic-controller${process.env.CONTROL_INPUTS_PER_SIGN?'-'+process.env.CONTROL_INPUTS_PER_SIGN:''}.json`,dir),'utf8'));
const targets=JSON.parse(fs.readFileSync(new URL('activity-targets.json',dir),'utf8'));
const raw=gunzipSync(fs.readFileSync(new URL('public/data/connectome.bin.gz',root)));
const c=new SpikingCircuit(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength),Number(process.env.CONTROL_TRACE_MS??50));
c.respectOpticalRefractory=process.env.CONTROL_REFRACTORY==='1';
const input=Int32Array.from(meta.inputIds), output=Int32Array.from(meta.outputIds),rates=new Float32Array(input.length);
if(new Set([...input,...output]).size!==input.length+output.length)throw Error('Input and displayed populations overlap.');
const rows=Int32Array.from(meta.row),cols=Int32Array.from(meta.column),weights=Float64Array.from(meta.weight);
const q=new Float64Array(input.length), residual=new Float64Array(output.length),gradient=new Float64Array(input.length);
const demand=new Float64Array(output.length),integral=new Float64Array(output.length);
const maxRate=Number(process.env.CONTROL_MAX_HZ??4000), rateScale=Number(process.env.CONTROL_OUTPUT_HZ??60);
const byInput=Array.from({length:input.length},()=>[] as number[]);
for(let e=0;e<cols.length;e++)byInput[cols[e]].push(e);
const norm=Float64Array.from(byInput,es=>es.reduce((v,e)=>v+weights[e]**2,0)+.0001);
function allocate(iterations:number){
  for(let j=0;j<residual.length;j++)residual[j]=-demand[j];
  for(let e=0;e<rows.length;e++)residual[rows[e]]+=weights[e]*q[cols[e]];
  for(let iter=0;iter<iterations;iter++){
    for(let i=0;i<q.length;i++){
      let grad=.0001*q[i];for(const e of byInput[i])grad+=weights[e]*residual[rows[e]];
      const value=Math.max(0,Math.min(maxRate*.001375*meta.strength[i],q[i]-grad/norm[i]));
      const delta=value-q[i];q[i]=value;
      for(const e of byInput[i])residual[rows[e]]+=weights[e]*delta;
    }
  }
  for(let i=0;i<rates.length;i++){const r=q[i]/(.001375*meta.strength[i]);rates[i]=c.respectOpticalRefractory?r/Math.max(.02,1-r*.002):r;}
}
const results=[];
for(const target of targets.slice(1)){
  c.reset();q.fill(0);integral.fill(0);rates.fill(0);
  const begin=performance.now();let lastSpikes=0;
  for(let step=0;step<3000;step++){
    if(step%25===0){
      for(let j=0;j<output.length;j++){
        const desired=rateScale*target.desiredActivity[j],actual=c.rateHz(output[j]);
        const time=desired>0?Math.max(.1,1000/desired-2.2):1e6;
        const feedforward=desired>0?7/(1-20/15*Math.exp(-time/20)+5/15*Math.exp(-time/5)):0;
        const error=desired-actual;
        integral[j]=Math.max(-250,Math.min(250,integral[j]+error*.005*4));
        demand[j]=Math.max(-300,Math.min(200,feedforward+integral[j]+.5*error));
      }
      allocate(step===0?30:3);
    }
    c.step(input,rates);
    if(step%500===499){console.log(`${target.label} ${c.timeMs} ms: ${c.spikes-lastSpikes} spikes, ${(performance.now()-begin).toFixed()} ms wall`);lastSpikes=c.spikes;}
  }
  const activity=Array.from(output,id=>Math.min(1,c.rateHz(id)/rateScale));
  const activityMse=activity.reduce((s,x,j)=>s+(x-target.desiredActivity[j])**2,0)/output.length;
  const record={label:target.label,pixels:target.pixels,desiredActivity:target.desiredActivity,activity,activityMse,spikes:c.spikes,deliveries:c.deliveries,maximumInputRateHz:Math.max(...rates),meanInputRateHz:rates.reduce((s,r)=>s+r,0)/rates.length,wallMs:performance.now()-begin};
  results.push(record);console.log(JSON.stringify({label:target.label,activityMse,spikes:c.spikes,wallMs:record.wallMs}));
  fs.writeFileSync(new URL(`feedback-coordinate-${maxRate}-${rateScale}-trace${c.traceMs}-k${process.env.CONTROL_INPUTS_PER_SIGN??3}-ref${c.respectOpticalRefractory}.json`,dir),JSON.stringify(results));
}
