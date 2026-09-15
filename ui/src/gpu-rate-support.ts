import {GpuCircuit} from './gpu-circuit';

const code=`
struct Params {neurons:u32,gain:f32,traceScale:f32,currentScale:f32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<storage,read_write> requests:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> roles:array<u32>;
@group(0) @binding(4) var<storage,read> importance:array<vec2<f32>>;
@group(0) @binding(5) var<storage,read_write> deficit:array<f32>;
@group(0) @binding(6) var<storage,read> offsets:array<u32>;
@group(0) @binding(7) var<storage,read> edges:array<vec2<u32>>;
@compute @workgroup_size(128)
fn measure(@builtin(global_invocation_id) g:vec3<u32>){
 let id=g.x;if(id>=p.neurons){return;}
 let missing=max(0.0,requests[id].x-clamp(cells[id].trace*p.traceScale,0.0,1.0));
 let scarce=clamp(1.0-max(0.0,cells[id].current)/p.currentScale,0.0,1.0);
 // A stale, unreachable rate request is not evidence that the current image
 // needs more light. Only help where the measured-image gradient requests it.
 deficit[id]=select(0.0,missing*scarce*importance[id].y,roles[id]==1u&&requests[id].y<0.0);
}
@compute @workgroup_size(128)
fn support(@builtin(global_invocation_id) g:vec3<u32>){
 let pre=g.x;if(pre>=p.neurons||(roles[pre]!=1u&&roles[pre]!=3u)){return;}
 var signal=0.0;
 for(var e=offsets[pre];e<offsets[pre+1u];e++){
  let item=edges[e];let w=bitcast<f32>(item.y);
  signal+=w*deficit[item.x];
 }
 let old=requests[pre];let value=clamp(old.x+p.gain*signal/importance[pre].x,0.0,old.z);
 requests[pre]=vec4(value,old.y,old.z,old.w);
}
`;

/** Heuristic upstream support for requested but underactive gated neurons.
 * Signed, measured connections transfer their rate deficit to presynaptic rate
 * requests. The ordinary optical feedback still charges every presynaptic arbor
 * for its visible light. This changes external control references only; the
 * actual graph, membrane, incoming currents and spike traces are never written.
 */
export class GpuRateSupport {
 private buffers:GPUBuffer[]=[];
 private measure:GPUComputePipeline;
 private support:GPUComputePipeline;
 private measureGroup:GPUBindGroup;
 private supportGroup:GPUBindGroup;
 private subgroup:boolean;
 constructor(circuit:GpuCircuit,requests:GPUBuffer,roles:Uint32Array,importance:ArrayLike<number>,rateScale:number,gain:number){
  if(roles.length!==circuit.n||importance.length!==circuit.n||!Number.isFinite(gain)||gain<0||!(rateScale>0))throw Error('Invalid signed rate-support parameters.');
  const device=circuit.device,n=circuit.n;
  this.subgroup=device.features.has('subgroups')&&device.adapterInfo.subgroupMinSize===32&&device.adapterInfo.subgroupMaxSize===32;
  const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
   const buffer=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage,mappedAtCreation:typeof data!=='number'});
   if(typeof data!=='number'){new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));buffer.unmap();}this.buffers.push(buffer);return buffer;
  };
  const incoming=new Float64Array(n);
  for(let e=0;e<circuit.edgeCount;e++)incoming[circuit.targets[e]]+=Math.abs(circuit.weights[e]);
  const scales=new Float32Array(n*2);
  for(let id=0;id<n;id++){scales[id*2]=importance[id];scales[id*2+1]=importance[id]/Math.max(1,incoming[id]);}
  // This normalization depends only on the fixed anatomy/connectome. Computing
  // it once avoids another irregular 15-million-edge gather every feedback step.
  // Every measured edge still participates in both this sum and live feedback.
  for(let id=0;id<n;id++){
   let downstream=0;for(let e=circuit.offsets[id];e<circuit.offsets[id+1];e++)downstream+=Math.abs(circuit.weights[e])*scales[circuit.targets[e]*2+1];
   scales[id*2]=Math.max(.000001,scales[id*2],downstream);
  }
  const entries=new Uint32Array(circuit.edgeCount*2),floats=new Float32Array(entries.buffer);
  for(let e=0;e<circuit.edgeCount;e++){entries[e*2]=circuit.targets[e];floats[e*2+1]=circuit.weights[e];}
  const raw=new ArrayBuffer(16),u=new Uint32Array(raw),f=new Float32Array(raw);u[0]=n;f[1]=gain;f[2]=1000/circuit.traceMs/rateScale;f[3]=2;
  const parameters=create(new Uint8Array(raw),GPUBufferUsage.UNIFORM),roleBuffer=create(roles),importanceBuffer=create(scales),deficit=create(n*4),offsets=create(circuit.offsets),edges=create(entries);
  let shader=code;
  if(this.subgroup){
   shader='enable subgroups;\n'+shader;
   shader=shader.replace('@compute @workgroup_size(128)\nfn support(@builtin(global_invocation_id) g:vec3<u32>)','@compute @workgroup_size(32)\nfn support(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32)');
   shader=shader.replace('let pre=g.x;', 'let pre=g.x+g.y*65535u;');
   shader=shader.replace('e=offsets[pre];e<offsets[pre+1u];e++','e=offsets[pre]+lane;e<offsets[pre+1u];e+=32u');
   shader=shader.replace('let old=requests[pre];','signal=subgroupAdd(signal);if(lane!=0u){return;}let old=requests[pre];');
  }
  const module=device.createShaderModule({code:shader});
  this.measure=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'measure'}});
  this.support=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'support'}});
  const bind=(pipeline:GPUComputePipeline,buffers:[number,GPUBuffer][])=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map(([binding,buffer])=>({binding,resource:{buffer}}))});
  this.measureGroup=bind(this.measure,[[0,parameters],[1,circuit.cells],[2,requests],[3,roleBuffer],[4,importanceBuffer],[5,deficit]]);
  this.supportGroup=bind(this.support,[[0,parameters],[2,requests],[3,roleBuffer],[4,importanceBuffer],[5,deficit],[6,offsets],[7,edges]]);
 }
 encode(pass:GPUComputePassEncoder,n:number){
  pass.setPipeline(this.measure);pass.setBindGroup(0,this.measureGroup);pass.dispatchWorkgroups(Math.ceil(n/128));
  pass.setPipeline(this.support);pass.setBindGroup(0,this.supportGroup);pass.dispatchWorkgroups(this.subgroup?Math.min(n,65535):Math.ceil(n/128),this.subgroup?Math.ceil(n/65535):1);
 }
 destroy(){this.buffers.forEach(buffer=>buffer.destroy());}
}
