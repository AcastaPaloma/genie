/** Validate the same controller used by the browser on persistent sequences. */
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {SpikingCircuit} from '../src/neural.ts';
import {SynapticImageController} from '../src/synaptic-controller.ts';

const root=new URL('../',import.meta.url),cache=new URL('.cache/control-goal/',root);
const data=new URL('public/data/',root);
const meta=JSON.parse(fs.readFileSync(new URL('presynaptic-controller.json',data),'utf8'));
const circuitMeta=JSON.parse(fs.readFileSync(new URL('connectome.json',data),'utf8'));
const bytes=gunzipSync(fs.readFileSync(new URL('connectome.bin.gz',data)));
const circuit=new SpikingCircuit(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),100);
const operator=fs.readFileSync(new URL('synaptic-image-operator.bin',data));
const ob=operator.buffer.slice(operator.byteOffset,operator.byteOffset+operator.byteLength);
const controller=new SynapticImageController(meta,ob,circuitMeta.outputs.map((x:{index:number})=>x.index));
const [P,,E]=new Uint32Array(ob,0,3),pixel=new Uint16Array(ob,12,E),neuron=new Uint16Array(ob,12+2*E,E),weight=new Float32Array(ob,12+4*E,E);
const project=(activity:Float32Array)=>{
  const y=new Float64Array(P);for(let e=0;e<E;e++)y[pixel[e]]+=weight[e]*activity[neuron[e]];return y;
};
const run=(ms:number)=>{for(let i=0;i<ms/circuit.dt;i++){controller.update(circuit);circuit.step(controller.inputIds,controller.inputRates);}};
const small=(luma:number[])=>{
  const y=new Float64Array(1200);for(let j=0;j<30;j++)for(let i=0;i<40;i++){const p=j*160+i*2;y[j*40+i]=(luma[p]+luma[p+1]+luma[p+80]+luma[p+81])/4;}return y;
};
const clips=JSON.parse(fs.readFileSync(new URL('validation-clips.json',cache),'utf8'));
const results=[];
for(const clip of clips){
  circuit.reset();controller.reset();controller.setFrame(clip.frames[0].luminance);run(300);
  const frames=[];const start=performance.now();
  for(const frame of clip.frames){
    controller.setFrame(frame.luminance);run(100);
    const observed=project(controller.measuredActivity(circuit)),target=small(frame.luminance);
    const mean=target.reduce((s,x)=>s+x,0)/P;
    const mse=observed.reduce((s,x,i)=>s+(x-target[i])**2,0)/P;
    const flatMse=target.reduce((s,x)=>s+(x-mean)**2,0)/P;
    frames.push({index:frame.index,mse,psnr:-10*Math.log10(mse),flatMse,fitMse:controller.fitMse,target:Array.from(target),observed:Array.from(observed)});
    if(frames.length%4===0)console.log(`${clip.clip} ${frames.length}/16 frames, ${frames.at(-1)!.psnr.toFixed(2)} dB, ${((performance.now()-start)/1000).toFixed(1)} s wall`);
  }
  results.push({clip:clip.clip,frames,wallMs:performance.now()-start});
  fs.writeFileSync(new URL('sequence-validation.json',cache),JSON.stringify(results));
}
// A reset and disconnected graph must leave every displayed neuron silent,
// despite a nonzero movie target and continued presynaptic stimulation.
circuit.reset();controller.reset();circuit.synapsesEnabled=false;
controller.setFrame(clips[0].frames[0].luminance);run(300);
const maximumDisconnected=Math.max(...controller.measuredActivity(circuit));
if(maximumDisconnected!==0)throw Error('Ablation failed: displayed neurons still fire without connections.');
fs.writeFileSync(new URL('sequence-ablation.json',cache),JSON.stringify({maximumDisconnected,deliveries:circuit.deliveries,inputSpikes:circuit.spikes}));
console.log('Ablation passed:',{maximumDisconnected,deliveries:circuit.deliveries,inputSpikes:circuit.spikes});
