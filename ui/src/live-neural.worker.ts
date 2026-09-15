import type {LiveProfile} from './live-profile';
import {GpuCircuit} from './gpu-circuit';
import {GpuPixelController} from './gpu-pixel-controller';
import {motorReadout,type CircuitMetadata} from './neural';
import {fetchJson} from './types';
import type {SynapticControlMetadata} from './synaptic-controller';

let circuit:GpuCircuit,controller:GpuPixelController,metadata:CircuitMetadata;
let busy=false,epoch=0;
self.onmessage=async event=>{
  const m=event.data;
  try{
    if(m.type==='init'){
      const profile=m.profile as LiveProfile;
      const [raw,meta,control,operator]=await Promise.all([
        fetch(m.base+'connectome.bin.gz').then(r=>r.arrayBuffer()),fetchJson<CircuitMetadata>(m.base+'connectome.json'),
        fetchJson<SynapticControlMetadata>(m.base+profile.controller),fetch(m.base+profile.operator).then(r=>r.arrayBuffer())]);
      metadata=meta;
      const decompressed=new Uint8Array(raw)[0]===31?await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():raw;
      const [version,rows,columns,entries]=new Uint32Array(operator,0,4);if(version!==784)throw Error('Unknown image basis.');
      circuit=await GpuCircuit.create(decompressed,{traceMs:profile.parameters.traceMs});
      controller=new GpuPixelController(circuit,control,{rows,columns,row:new Uint32Array(operator,16,entries),column:new Uint32Array(operator,16+4*entries,entries),weight:new Float32Array(operator,16+8*entries,entries)},metadata.outputs.map(n=>n.index),profile.parameters);
      // Compile all pipelines before starting the media clock; reset the warm-up.
      controller.setFrame(new Float32Array(rows));await circuit.advance(1,true,(pass,tick)=>controller.encode(pass,tick));circuit.reset();controller.reset();
      self.postMessage({type:'ready',neurons:circuit.n,edges:circuit.edgeCount,displayed:columns,inputs:control.inputIds.length,rateScale:controller.rateScale,adapter:circuit.device.adapterInfo.description||circuit.device.adapterInfo.architecture||'WebGPU'});
    }else if(m.type==='config'){
      circuit.synapsesEnabled=m.synapses;controller.setMuted(m.muted);
    }else if(m.type==='reset'){
      // Main thread waits for outstanding work before requesting reset.
      if(busy)throw Error('Cannot reset while a frame is in flight.');
      epoch=m.epoch;circuit.reset();controller.reset();self.postMessage({type:'reset',epoch});
    }else if(m.type==='step'){
      if(busy)throw Error('A second neural frame was queued.');busy=true;
      const start=performance.now();controller.setFrame(m.values);
      const observed=(await circuit.advance(100,true,(pass,tick)=>controller.encode(pass,tick)))!;
      const values=Float32Array.from(controller.outputIds,id=>Math.min(1,observed.cells[id*5+2]*1000/circuit.traceMs/controller.rateScale));
      const motor=motorReadout(metadata,{rateHz:id=>observed.cells[id*5+2]*1000/circuit.traceMs});
      busy=false;self.postMessage({type:'state',epoch,timeMs:circuit.timeMs,values,motor,spikes:observed.spikes,deliveries:observed.deliveries,computeMs:performance.now()-start,sourceTime:m.sourceTime,capturedAt:m.capturedAt,decodedAt:m.decodedAt,frameId:m.frameId},{transfer:[values.buffer]});
    }
  }catch(error){busy=false;self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
self.onclose=()=>{controller?.destroy();circuit?.destroy();};
