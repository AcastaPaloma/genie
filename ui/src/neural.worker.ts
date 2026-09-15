import { SpikingCircuit, motorReadout, type CircuitMetadata } from './neural';
import {SynapticImageController,type SynapticControlMetadata} from './synaptic-controller';

let circuit: SpikingCircuit;
let metadata: CircuitMetadata;
let inputIds: Int32Array;
let inputRates: Float32Array;
let maxRate=180;
let muted=false;
let luminance: Float32Array;
let pulseUntil=0;
let encoder='synaptic';
let synapticController:SynapticImageController;
let visualInputIds:Int32Array;
let visualInputRates:Float32Array;
let controller:{controlCount:number;inputGroup:number[];baseline:number[];response:number[][];covariance:number[][];step:number}|null=null;
let controls:Float64Array;

function encode() {
  if (!metadata || !luminance || !inputRates) return;
  if(encoder==='synaptic'){
    synapticController.setFrame(luminance);
    return;
  }
  if(encoder==='inverse'&&controller){
    const c=controller, target=new Float32Array(1200);
    for(let y=0;y<30;y++)for(let x=0;x<40;x++){
      const p=y*2*80+x*2;target[y*40+x]=(luminance[p]+luminance[p+1]+luminance[p+80]+luminance[p+81])/4;
    }
    const rhs=c.response.map(col=>col.reduce((sum,value,p)=>sum+value*(target[p]-c.baseline[p])/1200,0));
    for(let iter=0;iter<60;iter++){
      const next=controls.slice();
      for(let k=0;k<c.controlCount;k++){
        let grad=-rhs[k];for(let j=0;j<c.controlCount;j++)grad+=c.covariance[k][j]*controls[j];
        next[k]=Math.max(-1,Math.min(1,controls[k]-c.step*grad));
      }
      controls=next;
    }
    for(let i=0;i<inputRates.length;i++)inputRates[i]=muted?0:80*(1+controls[c.inputGroup[i]]);
    return;
  }
  for (let i=0;i<metadata.inputs.length;i++) {
    const n=metadata.inputs[i];
    const x=Math.min(79,Math.floor(n.u*80)), y=Math.min(59,Math.floor(n.v*60));
    inputRates[i]=muted?0:maxRate*luminance[y*80+x];
  }
}
function report() {
  const values=encoder==='synaptic'?synapticController.measuredActivity(circuit):circuit.renderActivity(metadata.renderMap);
  self.postMessage({type:'state', timeMs:circuit.timeMs, spikes:circuit.spikes, deliveries:circuit.deliveries,
    active:circuit.activeCount, motor:motorReadout(metadata,circuit), values,encoder,traceMs:circuit.traceMs,
    fitMse:synapticController.fitMse},{transfer:[values.buffer]});
}
self.onmessage=async (event:MessageEvent)=>{
  try {
    const message=event.data;
    if (message.type==='init') {
      metadata=message.metadata;
      controller=message.controller;controls=new Float64Array(controller?.controlCount??0);
      const response=await fetch(message.url);
      if (!response.ok || !response.body) throw new Error(`Connectome download failed (${response.status}).`);
      // Vite serves *.gz with Content-Encoding:gzip, which fetch transparently
      // decodes. Other static hosts return the compressed bytes themselves.
      // Inspect the bytes so both serving conventions work without decoding twice.
      const raw=await response.arrayBuffer();
      const signature=new Uint8Array(raw,0,Math.min(2,raw.byteLength));
      const buffer=signature[0]===0x1f&&signature[1]===0x8b
        ? await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():raw;
      const [controlResponse,operatorResponse]=await Promise.all([fetch(message.controlUrl),fetch(message.operatorUrl)]);
      if(!controlResponse.ok||!operatorResponse.ok)throw Error('The synaptic image controller could not be loaded.');
      const [controlMeta,operator]=await Promise.all([controlResponse.json() as Promise<SynapticControlMetadata>,operatorResponse.arrayBuffer()]);
      synapticController=new SynapticImageController(controlMeta,operator,metadata.outputs.map(n=>n.index));
      if(controlMeta.outputIds.some((id,i)=>id!==metadata.renderMap[i]))throw Error('Controller and displayed neuron identities differ.');
      circuit=new SpikingCircuit(buffer,100);
      if (circuit.n!==metadata.neuronCount || circuit.edgeCount!==metadata.edgeCount) throw new Error('Connectome and metadata versions do not match.');
      visualInputIds=Int32Array.from(metadata.inputs,n=>n.index);
      visualInputRates=new Float32Array(visualInputIds.length);
      inputIds=synapticController.inputIds;inputRates=synapticController.inputRates;
      luminance=new Float32Array(80*60);
      self.postMessage({type:'ready'}); report();
    } else if (message.type==='frame') { luminance=message.values; encode(); }
    else if (message.type==='config') {
      maxRate=message.maxRate; muted=message.muted;encoder=message.encoder; circuit.synapsesEnabled=message.synapses;
      inputIds=encoder==='synaptic'?synapticController.inputIds:visualInputIds;
      inputRates=encoder==='synaptic'?synapticController.inputRates:visualInputRates;
      circuit.setOpticalTargets(inputIds);encode();
    } else if (message.type==='pulse') { pulseUntil=circuit.timeMs+150; }
    else if (message.type==='reset') { circuit.reset();synapticController.reset();pulseUntil=0;controls.fill(0);encode();report(); }
    else if (message.type==='step') {
      const ticks=Math.round(message.ms/circuit.dt);
      const pulse=metadata.outputs.filter(n=>n.cellType==='DNp09').map(n=>n.index);
      for (let i=0;i<ticks;i++) {
        if(encoder==='synaptic')synapticController.update(circuit,muted);
        circuit.step(inputIds,inputRates,circuit.timeMs<pulseUntil && Math.round(circuit.timeMs/circuit.dt)%25===0 ? pulse:[]);
      }
      report();
    }
  } catch(error) { self.postMessage({type:'error', message:error instanceof Error?error.message:String(error)}); }
};
