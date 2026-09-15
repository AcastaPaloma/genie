import {GpuCircuit} from './gpu-circuit';
import {GpuWholeBrainController} from './gpu-whole-brain-controller';
import {GpuSynapticGateController} from './gpu-synaptic-gate-controller';
import {GpuArborInverseController} from './gpu-arbor-inverse-controller';
import {motorReadout,type CircuitMetadata} from './neural';
import type {FullBrainMetadata} from './full-brain';
import {fetchJson} from './types';

let circuit:GpuCircuit,controller:GpuWholeBrainController|GpuArborInverseController|GpuSynapticGateController,metadata:CircuitMetadata;
let busy=false,epoch=0,inputMask:Uint8Array,seedMask:Uint8Array|undefined,visualMask:Uint8Array|undefined,calibration:Float32Array|null=null;
async function anatomyRecord(url:string){
  const response=await fetch(url);if(!response.ok)throw Error('Complete anatomy manifest is unavailable.');
  const raw=await response.arrayBuffer();
  const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),n=>n.toString(16).padStart(2,'0')).join('');
  return {value:JSON.parse(new TextDecoder().decode(raw)) as FullBrainMetadata,sha256};
}
async function verified(url:string,expected:string|string[]){
  const response=await fetch(url);if(!response.ok)throw Error(`Could not load ${url}: ${response.status}`);
  const raw=await response.arrayBuffer();
  const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),n=>n.toString(16).padStart(2,'0')).join('');
  if(!(Array.isArray(expected)?expected:[expected]).includes(actual))throw Error(`Data checksum differs: ${url}`);return raw;
}
self.onmessage=async event=>{
  const m=event.data;
  try{
    if(m.type==='init'){
      const controlWidth=m.controlWidth??320;
      if(controlWidth!==320&&controlWidth!==640)throw Error('Control width must be 320 or 640.');
      const [meta,record,basis]=await Promise.all([
        fetchJson<CircuitMetadata&{sha256:string;uncompressedSha256:string;wholeRelease:boolean;neuronIds:string[];visualSensoryIds:number[]}>(m.base+'connectome.json'),
        anatomyRecord(m.base+'brain.json'),
        fetchJson<{operatorSha256:string;sourceAnatomySha256:string;resolution:number[];neurons:number}>(m.base+`whole-arbor-${controlWidth}.json`)]);
      const anatomy=record.value;
      if(record.sha256!==basis.sourceAnatomySha256)throw Error('The observation operator belongs to a different anatomy manifest.');
      if(!meta.wholeRelease||!anatomy.complete||anatomy.missingRootIds.length||anatomy.neuronCount!==139255||basis.neurons!==anatomy.neuronCount)
        throw Error('Playback requires the complete verified FlyWire release.');
      if(meta.neuronIds.length!==anatomy.neuronCount||anatomy.neurons.some(cell=>meta.neuronIds[cell.modelIndex]!==cell.rootId))
        throw Error('Anatomical and simulated neuron identities differ.');
      metadata=meta;
      // Some static servers apply Content-Encoding: gzip to .gz files, so fetch
      // already returns the decoded bytes. Verify either pinned representation.
      const [packed,operator]=await Promise.all([verified(m.base+'connectome.bin.gz',[meta.sha256,meta.uncompressedSha256]),verified(m.base+`whole-arbor-${controlWidth}.bin`,basis.operatorSha256)]);
      const raw=new Uint8Array(packed)[0]===31?await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():packed;
      const [version,rows,columns,entries]=new Uint32Array(operator,0,4);
      if(version!==784||columns!==anatomy.neuronCount||basis.resolution[0]!==controlWidth||rows!==basis.resolution[0]*basis.resolution[1]||operator.byteLength!==16+entries*12)
        throw Error('The whole-anatomy observation matrix differs from its manifest.');
      circuit=await GpuCircuit.create(raw,{traceMs:m.parameters?.traceMs??40,gateBias:m.parameters?.gateBias??6.8,largeBufferBytes:Math.max(entries*8,new Uint32Array(raw,0,3)[2]*8)});
      if(circuit.n!==139255)throw Error('Not every released neuron is simulated.');
      const motorIds=metadata.outputs.map(n=>n.index);
      const parameters=m.parameters??{depth:0,feedbackTicks:100,step:20};
      if(parameters.useVisualInputs){if(parameters.controller!=='gate')throw Error('Visual-input control requires the gate experiment.');if(!Array.isArray(meta.visualSensoryIds)||meta.visualSensoryIds.length!==11391||new Set(meta.visualSensoryIds).size!==11391)throw Error('The complete published visual-input annotation is required.');parameters.visualInputIds=meta.visualSensoryIds;}
      const Control=parameters.controller==='gate'?GpuSynapticGateController:parameters.controller==='inverse'?GpuArborInverseController:GpuWholeBrainController;
      controller=new Control(circuit,{rows,columns,imageWidth:basis.resolution[0],row:new Uint32Array(operator,16,entries),column:new Uint32Array(operator,16+4*entries,entries),weight:new Float32Array(operator,16+8*entries,entries)},motorIds,parameters);
      inputMask=new Uint8Array(circuit.n);controller.inputIds.forEach(id=>inputMask[id]=1);if(controller instanceof GpuSynapticGateController){seedMask=new Uint8Array(circuit.n);controller.seedIds.forEach(id=>seedMask![id]=1);visualMask=new Uint8Array(circuit.n);controller.visualInputIds.forEach(id=>visualMask![id]=1);}
      if(parameters.calibrate){
        if(!(controller instanceof GpuWholeBrainController))throw Error('Calibration is supported only by image feedback.');
        // A uniform white calibration is independent of the provided clips.
        // Only counted spikes determine each neuron's fixed brightness scale.
        controller.setFrame(new Float32Array(rows).fill(1));
        const before=(await circuit.advance(1000,true,(pass,tick)=>controller.encode(pass,tick)))!;
        const after=(await circuit.advance(1000,true,(pass,tick)=>controller.encode(pass,tick)))!;
        controller.setRateCalibration(Float32Array.from(after.counts,(count,id)=>Math.max(10,count-before.counts[id])));
        calibration=controller.rateCalibration;
      }
      controller.setFrame(new Float32Array(rows));await circuit.advance(1,true,(pass,tick)=>controller.encode(pass,tick),controller instanceof GpuSynapticGateController?controller.simulationBatchTicks:9);circuit.reset();controller.reset();
      self.postMessage({type:'ready',neurons:circuit.n,edges:circuit.edgeCount,inputs:controller.inputIds.length,seeds:controller instanceof GpuSynapticGateController?controller.seedIds.length:undefined,inputIds:controller.inputIds.slice(),rateScale:controller.rateScale,
        seedIds:controller instanceof GpuSynapticGateController?controller.seedIds.slice():undefined,visualInputIds:controller instanceof GpuSynapticGateController?controller.visualInputIds.slice():undefined,
        adapter:circuit.device.adapterInfo.description||circuit.device.adapterInfo.architecture||'WebGPU'});
    }else if(m.type==='config'){
      circuit.synapsesEnabled=m.synapses;controller.setMuted(m.muted);
    }else if(m.type==='reset'){
      if(busy)throw Error('Cannot reset while a frame is in flight.');
      epoch=m.epoch;circuit.reset();controller.reset();self.postMessage({type:'reset',epoch});
    }else if(m.type==='step'){
      if(busy)throw Error('A second neural frame was queued.');busy=true;
      const start=performance.now();controller.setFrame(m.values);
      const observed=(await circuit.advance(100,true,(pass,tick)=>controller.encode(pass,tick),controller instanceof GpuSynapticGateController?controller.simulationBatchTicks:9))!;
      const values=new Float32Array(circuit.n);let inputSpikes=0,downstreamSpikes=0,maxInput=0,maxDownstream=0,seedSpikes=0,visualInputSpikes=0,synapseDependentSpikes=0;
      for(let id=0;id<circuit.n;id++){
        const activity=Math.max(0,Math.min(1,(observed.cells[id*5+2]*1000/circuit.traceMs/(calibration?.[id]??1)-controller.rateFloor)/controller.rateScale));values[id]=activity;
        if(seedMask){if(seedMask[id])seedSpikes+=observed.counts[id];else if(visualMask?.[id])visualInputSpikes+=observed.counts[id];else synapseDependentSpikes+=observed.counts[id];}
        if(inputMask[id]){inputSpikes+=observed.counts[id];maxInput=Math.max(maxInput,activity);}
        else{downstreamSpikes+=observed.counts[id];maxDownstream=Math.max(maxDownstream,activity);}
      }
      const motor=motorReadout(metadata,{rateHz:id=>observed.cells[id*5+2]*1000/circuit.traceMs});
      const attribution=controller instanceof GpuSynapticGateController?controller.lightAttribution(values):{};
      const computeMs=performance.now()-start,auditStart=performance.now();
      const audit=m.audit&&controller instanceof GpuSynapticGateController?await controller.audit(observed.cells):{};
      const auditMs=performance.now()-auditStart;
      busy=false;self.postMessage({type:'state',...audit,...attribution,epoch,timeMs:circuit.timeMs,values,motor,spikes:observed.spikes,deliveries:observed.deliveries,
        inputSpikes,downstreamSpikes,maxInput,maxDownstream,seedSpikes,visualInputSpikes,synapseDependentSpikes,computeMs,auditMs,sourceTime:m.sourceTime,capturedAt:m.capturedAt,decodedAt:m.decodedAt,frameId:m.frameId},{transfer:[values.buffer]});
    }
  }catch(error){busy=false;self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
self.onclose=()=>{controller?.destroy();circuit?.destroy();};
