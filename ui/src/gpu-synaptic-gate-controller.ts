import {GpuCircuit} from './gpu-circuit';
import {GpuSparseSolver,type SparseTriplets} from './gpu-solver';
import {coverageSeeds} from './seed-coverage';
import {GpuRateSupport} from './gpu-rate-support';

const code=`
struct Params {neurons:u32,muted:u32,traceToHz:f32,rateScale:f32,seconds:f32,kp:f32,ki:f32,seedDrive:f32,minDrive:f32,gateMode:u32,shunt:f32,traceTrigger:u32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<storage,read> wanted:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> role:array<u32>;
@group(0) @binding(4) var<storage,read_write> integral:array<f32>;
@group(0) @binding(5) var<storage,read_write> drive:array<vec2<f32>>;
@compute @workgroup_size(128)
fn feedback(@builtin(global_invocation_id) g:vec3<u32>){
 let id=g.x;if(id>=p.neurons||role[id]==0u){return;}
 if(role[id]==2u){drive[id]=vec2(select(p.seedDrive,0.0,p.muted!=0u),3.0);return;}
 let error=wanted[id].x*p.rateScale-cells[id].trace*p.traceToHz;
 if(role[id]==3u){
  // Optional anatomical visual-input experiment. Apply a bounded external
  // current; the normal membrane, incoming currents and refractory period
  // still determine every spike. These cells are counted as direct inputs.
  let rate=wanted[id].x*p.rateScale;
  let interval=max(.0001,1000.0/max(.001,rate)-2.2);
  let feedforward=select(7.0/(1.0-exp(-interval/20.0)),0.0,rate<=0.0);
  let current=clamp(feedforward+p.kp*error,p.minDrive,p.seedDrive);
  drive[id]=vec2(select(current,0.0,p.muted!=0u),3.0);return;
 }
 if(p.gateMode>=2u){
  var trigger=wanted[id].x*p.rateScale;
  if(p.traceTrigger!=0u){
   // For a periodic train with mean r, the trace immediately before a spike is
   // A/(exp(A/r)-1), where A=1000/traceMs is one spike's trace increment in Hz.
   // Comparing directly with r instead creates a substantial positive bias,
   // especially when the requested intensity is low.
   trigger=select(p.traceToHz/(exp(min(80.0,p.traceToHz/max(.0001,trigger)))-1.0),0.0,trigger<=0.0);
  }
  let excitationReady=p.gateMode!=4u||cells[id].current>0.0;
  let conductance=select(p.shunt,0.0,cells[id].trace*p.traceToHz<trigger&&excitationReady);
  drive[id]=select(vec2(conductance,select(4.0,5.0,p.gateMode>=3u)),vec2(0.0,3.0),p.muted!=0u);return;
 }
 if(p.gateMode==1u){
  // Release inhibition as soon as another spike is needed. While the measured
  // trace is above its reference, counter the *observed* decaying current just
  // enough to approach rest over the next feedback interval. This avoids the
  // deep hyperpolarization and slow recovery caused by integral windup.
  // Future incoming events are not canceled and still undergo normal LIF
  // integration. This only changes the external tonic-current input.
  let decay=exp(-p.seconds/.02);
  let synaptic=cells[id].current/3.0*(decay-exp(-p.seconds/.005));
  let hold=(-(cells[id].voltage+52.0)*decay-synaptic)/max(.0001,1.0-decay);
  let value=select(clamp(hold,p.minDrive,0.0),6.8,error>0.0);
  drive[id]=vec2(select(value,0.0,p.muted!=0u),3.0);return;
 }
 if(p.muted==0u){integral[id]=clamp(integral[id]+error*p.seconds*p.ki,p.minDrive,6.8);}
 // 6.8 mV is strictly below the 7 mV rest-to-threshold difference.
 // With no incoming events these cells cannot generate spikes, regardless of
 // the requested image, controller error or number of iterations.
 let value=clamp(integral[id]+p.kp*error,p.minDrive,6.8);
 drive[id]=vec2(select(value,0.0,p.muted!=0u),3.0);
}
`;

/** Experimental excitability control. All light comes from own-neuron threshold
 * events. Constant seeds and optional, explicitly classified visual afferents
 * can fire without incoming events. Every input arbor stays visible.
 */
export class GpuSynapticGateController {
 readonly inputIds:Uint32Array;
 readonly seedIds:Uint32Array;
 readonly visualInputIds:Uint32Array;
 readonly rateScale:number;
 readonly simulationBatchTicks:number;
 readonly rateFloor=0;
 private inverse:GpuSparseSolver;
 private buffers:GPUBuffer[]=[];
 private parameters:GPUBuffer;
 private integral:GPUBuffer;
 private feedback:GPUComputePipeline;
 private group:GPUBindGroup;
 private fresh=true;
 private cold=true;
 private imageFeedback:number;
 private interval:number;
 private iterations:number;
 private importance:Float64Array;
 private rateSupport?:GpuRateSupport;
 constructor(private circuit:GpuCircuit,h:SparseTriplets,motorIds:number[],options:{rateScale?:number;feedbackTicks?:number;imageIterations?:number;seedCount?:number;kp?:number;ki?:number;seedDrive?:number;minDrive?:number;gateMode?:number;shunt?:number;traceTrigger?:number;imageFeedback?:number;imageAnchor?:number;seedCoverage?:number;simulationBatchTicks?:number;supportGain?:number;edgeWeight?:number;visualInputIds?:number[]}={}){
  if(h.columns!==circuit.n)throw Error('Every neuron must contribute its complete anatomy.');
  this.rateScale=options.rateScale??60;this.interval=options.feedbackTicks??25;this.iterations=options.imageIterations??16;
  this.simulationBatchTicks=options.simulationBatchTicks??9;
  this.imageFeedback=options.imageFeedback??0;
  const motor=new Set(motorIds),importance=new Float64Array(circuit.n),candidates:{id:number;score:number}[]=[];
  this.importance=importance;
  for(let e=0;e<h.row.length;e++)importance[h.column[e]]+=h.weight[e];
  for(let id=0;id<circuit.n;id++){
   if(motor.has(id))continue;
   let positive=0,negative=0;
   for(let e=circuit.offsets[id];e<circuit.offsets[id+1];e++){positive+=Math.max(0,circuit.weights[e]);negative+=Math.max(0,-circuit.weights[e]);}
   if(positive>0&&negative===0)candidates.push({id,score:Math.sqrt(positive)/Math.max(.05,importance[id])});
  }
  candidates.sort((a,b)=>b.score-a.score||a.id-b.id);
  this.seedIds=options.seedCoverage?coverageSeeds(circuit,candidates.map(x=>x.id),importance,options.seedCount??700):Uint32Array.from(candidates.slice(0,Math.min(options.seedCount??700,candidates.length)),x=>x.id);
  const roles=new Uint32Array(circuit.n).fill(1);for(const id of motor)roles[id]=0;for(const id of this.seedIds)roles[id]=2;
  const visual=new Set(options.visualInputIds??[]);
  for(const id of visual){if(!Number.isInteger(id)||id<0||id>=circuit.n||motor.has(id))throw Error('Visual inputs must be valid non-motor neurons.');if(roles[id]!==2)roles[id]=3;}
  this.visualInputIds=Uint32Array.from(Array.from(visual).filter(id=>roles[id]===3));
  this.inputIds=Uint32Array.from(Array.from(roles,(_,id)=>id).filter(id=>roles[id]>0));
  const device=circuit.device,create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
   const b=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});
   if(typeof data!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();}this.buffers.push(b);return b;
  };
  const upper=new Float32Array(circuit.n).fill(1);
  this.inverse=new GpuSparseSolver(device,h,upper,{observations:circuit.cells,observationScale:1000/circuit.traceMs/this.rateScale,observationStep:this.imageFeedback,observationAnchor:options.imageAnchor??0,spatialWeight:options.edgeWeight??0});
  if(options.supportGain)this.rateSupport=new GpuRateSupport(circuit,this.inverse.states,roles,importance,this.rateScale,options.supportGain);
  const params=new Uint32Array([circuit.n,0,0,0,0,0,0,0,0,0,0,0]),f=new Float32Array(params.buffer);
  f[2]=1000/circuit.traceMs;f[3]=this.rateScale;f[4]=this.interval*.0002;f[5]=options.kp??.3;f[6]=options.ki??4;f[7]=options.seedDrive??50;f[8]=options.minDrive??-50;params[9]=options.gateMode??0;f[10]=options.shunt??256;params[11]=options.traceTrigger??0;
  this.parameters=create(params,GPUBufferUsage.UNIFORM);this.integral=create(circuit.n*4);
  const module=device.createShaderModule({code});this.feedback=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'feedback'}});
  const bound=[this.parameters,circuit.cells,this.inverse.states,create(roles),this.integral,circuit.inputs];
  this.group=device.createBindGroup({layout:this.feedback.getBindGroupLayout(0),entries:bound.map((buffer,binding)=>({binding,resource:{buffer}}))});this.reset();
 }
 setFrame(values:Float32Array<ArrayBuffer>){this.inverse.setTarget(values);this.fresh=true;}
 setMuted(value:boolean){this.circuit.device.queue.writeBuffer(this.parameters,4,new Uint32Array([value?1:0]));}
 encode(pass:GPUComputePassEncoder,tick:number){
  if(tick%this.interval)return;if(this.fresh){if(!this.imageFeedback||this.cold)this.inverse.encode(pass,this.iterations);this.fresh=false;this.cold=false;}
  if(this.imageFeedback&&tick%100===0){this.inverse.encodeObservedCorrection(pass);this.rateSupport?.encode(pass,this.circuit.n);}
  pass.setPipeline(this.feedback);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(Math.ceil(this.circuit.n/128));
 }
 reset(){this.inverse.reset();this.fresh=true;this.cold=true;this.circuit.device.queue.writeBuffer(this.integral,0,new Float32Array(this.circuit.n).fill(6.8));this.circuit.setTonicInputs(this.inputIds,new Float32Array(this.inputIds.length));}
 async audit(cells:Float32Array){
  const device=this.circuit.device,n=this.circuit.n,readback=device.createBuffer({size:n*24,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),encoder=device.createCommandEncoder();
  encoder.copyBufferToBuffer(this.inverse.states,0,readback,0,n*16);encoder.copyBufferToBuffer(this.circuit.inputs,0,readback,n*16,n*8);device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);const copy=readback.getMappedRange().slice(0);readback.unmap();readback.destroy();
  const raw=new Float32Array(copy),rates=new Float32Array(n),measuredRates=new Float32Array(n),seeds=new Set(this.seedIds),worst=[];
  for(let id=0;id<n;id++){
   rates[id]=raw[id*4]*this.rateScale;const actual=cells[id*5+2]*1000/this.circuit.traceMs;measuredRates[id]=actual;
   const error=Math.min(1,actual/this.rateScale)-Math.min(1,rates[id]/this.rateScale);
   worst.push({id,seed:seeds.has(id),desiredHz:rates[id],actualHz:actual,drive:raw[n*4+id*2],voltage:cells[id*5],synapticCurrent:cells[id*5+1],importance:this.importance[id],score:this.importance[id]*error*error});
  }
  worst.sort((a,b)=>b.score-a.score);return {referenceRates:rates,measuredRates,worstCells:worst.slice(0,40)};
 }
 lightAttribution(values:Float32Array){
  let total=0,seed=0,visual=0;for(let id=0;id<values.length;id++)total+=this.importance[id]*values[id];
  for(const id of this.seedIds)seed+=this.importance[id]*values[id];for(const id of this.visualInputIds)visual+=this.importance[id]*values[id];
  return {seedLightFraction:total?seed/total:0,visualInputLightFraction:total?visual/total:0,synapseDependentLightFraction:total?(total-seed-visual)/total:0};
 }
 destroy(){this.rateSupport?.destroy();this.inverse.destroy();this.buffers.forEach(b=>b.destroy());}
}
