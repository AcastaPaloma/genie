import {GpuCircuit} from './gpu-circuit';
import {GpuSparseSolver,type SparseTriplets} from './gpu-solver';

const code=`
struct Params {neurons:u32,inputs:u32,muted:u32,pad:u32,traceToHz:f32,rateScale:f32,seconds:f32,maxDrive:f32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> desired:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> cells:array<Cell>;
@group(0) @binding(3) var<storage,read> inputMap:array<i32>;
@group(0) @binding(4) var<storage,read> pointer:array<u32>;
@group(0) @binding(5) var<storage,read> edges:array<vec2<u32>>;
@group(0) @binding(6) var<storage,read> importance:array<f32>;
@group(0) @binding(7) var<storage,read_write> currentTarget:array<vec2<f32>>;
@group(0) @binding(8) var<storage,read_write> integral:array<f32>;
@group(0) @binding(9) var<storage,read> rates:array<vec4<f32>>;
@group(0) @binding(10) var<storage,read> inputIds:array<u32>;
@group(0) @binding(11) var<storage,read_write> drive:array<vec2<f32>>;
fn synapticCurrent(rate:f32)->f32{
 let ticks=max(1.0,(1000.0/max(rate,.001)-2.0)/.2);
 let a=.9900498337491681;let b=.9607894391523232;
 let am=pow(a,ticks);let bm=pow(b,ticks);
 let response=.009753464865615247*.04/(1.0-b)*((1.0-am)/(1.0-a)+(10.0*(1.0-b)-b)*(am-bm)/(a-b));
 return select(0.0,7.0/response,rate>.01);
}
@compute @workgroup_size(128)
fn feedback(@builtin(global_invocation_id) g:vec3<u32>){
 let id=g.x;if(id>=p.neurons){return;}
 let wanted=desired[id].x*p.rateScale;let actual=cells[id].trace*p.traceToHz;
 if(inputMap[id]>=0){currentTarget[id].x=importance[id]*wanted*20.0/p.rateScale;return;}
 var background=0.0;
 for(var e=pointer[id];e<pointer[id+1u];e++){
  let edge=edges[e];if(inputMap[edge.x]<0){background+=bitcast<f32>(edge.y)*cells[edge.x].trace*p.traceToHz;}
 }
 let error=wanted-actual;
 if(p.muted==0u){integral[id]=clamp(integral[id]+error*p.seconds*.5,-12.0,12.0);}
 currentTarget[id].x=importance[id]*clamp(synapticCurrent(wanted)+.05*error+integral[id]-background,-100.0,100.0);
}
@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.inputs){return;}let id=inputIds[i];let wanted=rates[i].x;
 let actual=cells[id].trace*p.traceToHz;let error=wanted-actual;
 var background=0.0;
 for(var e=pointer[id];e<pointer[id+1u];e++){
  let edge=edges[e];background+=bitcast<f32>(edge.y)*cells[edge.x].trace*p.traceToHz;
 }
 // Feedforward is the tonic LIF response. Feedback compensates only at the
 // explicitly selected input cells; all incoming events are still integrated.
 let freeMs=max(.2,1000.0/max(wanted,.001)-2.2);
 let ff=select(0.0,7.0/(1.0-exp(-freeMs/20.0)),wanted>.01);
 if(p.muted==0u){integral[id]=clamp(integral[id]+error*p.seconds*1.5,-20.0,20.0);}
 let current=clamp(ff-background+.1*error+integral[id],-p.maxDrive,p.maxDrive);
 drive[id]=vec2(select(current,0.0,p.muted!=0u),3.0);
}
`;

/** Two bounded inverses generate rate REFERENCES, then input currents. Only
 * GpuCircuit's threshold events generate the activity passed to the renderer.
 * Every neuron's complete arbor participates in the image loss, including
 * input cells. The current inverse includes measured connections only.
 */
export class GpuArborInverseController {
 readonly inputIds:Uint32Array;
 readonly rateScale:number;
 readonly rateFloor=0;
 private anatomy:GpuSparseSolver;
 private currents:GpuSparseSolver;
 private buffers:GPUBuffer[]=[];
 private parameters:GPUBuffer;
 private integral:GPUBuffer;
 private feedback:GPUComputePipeline;
 private apply:GPUComputePipeline;
 private feedbackGroup:GPUBindGroup;
 private applyGroup:GPUBindGroup;
 private fresh=true;
 private interval:number;
 private imageIterations:number;
 private currentIterations:number;
 constructor(private circuit:GpuCircuit,h:SparseTriplets,motorIds:number[],options:{rateScale?:number;feedbackTicks?:number;imageIterations?:number;currentIterations?:number;maxDrive?:number}={}){
  if(h.columns!==circuit.n)throw Error('Every neuron must participate in the anatomical inverse.');
  this.rateScale=options.rateScale??120;this.interval=options.feedbackTicks??50;
  this.imageIterations=options.imageIterations??32;this.currentIterations=options.currentIterations??12;
  const motor=new Set(motorIds),ids:number[]=[],map=new Int32Array(circuit.n).fill(-1);
  for(let id=0;id<circuit.n;id++)if(!motor.has(id)&&((Math.imul(id+1,2654435761)>>>0)<858993459)){map[id]=ids.length;ids.push(id);}
  this.inputIds=Uint32Array.from(ids);
  const device=circuit.device;
  const create=(value:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
   const b=device.createBuffer({size:Math.max(4,typeof value==='number'?value:value.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof value!=='number'});
   if(typeof value!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));b.unmap();}this.buffers.push(b);return b;
  };
  this.anatomy=new GpuSparseSolver(device,h,new Float32Array(circuit.n).fill(1));
  const importance=new Float32Array(circuit.n);for(let e=0;e<h.row.length;e++)importance[h.column[e]]+=h.weight[e];
  for(let id=0;id<circuit.n;id++)importance[id]=Math.sqrt(importance[id]);
  // CSR incoming contacts for causal background estimation. No graph pruning.
  const pointer=new Uint32Array(circuit.n+1);
  for(const post of circuit.targets)pointer[post+1]++;
  for(let id=0;id<circuit.n;id++)pointer[id+1]+=pointer[id];
  const cursor=pointer.slice(),edges=new Uint32Array(circuit.edgeCount*2),edgeWeights=new Float32Array(edges.buffer);
  let count=ids.length;
  for(const pre of ids)for(let e=circuit.offsets[pre];e<circuit.offsets[pre+1];e++)if(map[circuit.targets[e]]<0&&circuit.weights[e])count++;
  const row=new Uint32Array(count),column=new Uint32Array(count),weight=new Float32Array(count);let k=0;
  for(let pre=0;pre<circuit.n;pre++)for(let e=circuit.offsets[pre];e<circuit.offsets[pre+1];e++){
   const post=circuit.targets[e],at=cursor[post]++;edges[2*at]=pre;edgeWeights[2*at+1]=circuit.weights[e]*.001375;
   if(map[pre]>=0&&map[post]<0&&circuit.weights[e]){row[k]=post;column[k]=map[pre];weight[k++]=circuit.weights[e]*.001375*importance[post];}
  }
  for(let i=0;i<ids.length;i++){row[k]=ids[i];column[k]=i;weight[k++]=importance[ids[i]]*20/this.rateScale;}
  this.currents=new GpuSparseSolver(device,{rows:circuit.n,columns:ids.length,row,column,weight},new Float32Array(ids.length).fill(250));
  this.integral=create(circuit.n*4);
  const params=new Uint32Array([circuit.n,ids.length,0,0,0,0,0,0]),f=new Float32Array(params.buffer);
  f[4]=1000/circuit.traceMs;f[5]=this.rateScale;f[6]=this.interval*.0002;f[7]=options.maxDrive??50;
  this.parameters=create(params,GPUBufferUsage.UNIFORM);
  const module=device.createShaderModule({code}),pipeline=(entryPoint:string)=>device.createComputePipeline({layout:'auto',compute:{module,entryPoint}});
  this.feedback=pipeline('feedback');this.apply=pipeline('apply');
  const bind=(p:GPUComputePipeline,entries:Array<[number,GPUBuffer]>)=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))});
  const incomingPointer=create(pointer),incomingEdges=create(edges);
  this.feedbackGroup=bind(this.feedback,[[0,this.parameters],[1,this.anatomy.states],[2,circuit.cells],[3,create(map)],[4,incomingPointer],[5,incomingEdges],[6,create(importance)],[7,this.currents.residual],[8,this.integral]]);
  this.applyGroup=bind(this.apply,[[0,this.parameters],[2,circuit.cells],[4,incomingPointer],[5,incomingEdges],[8,this.integral],[9,this.currents.states],[10,create(this.inputIds)],[11,circuit.inputs]]);
  this.reset();
 }
 setFrame(values:Float32Array<ArrayBuffer>){this.anatomy.setTarget(values);this.fresh=true;}
 setMuted(value:boolean){this.circuit.device.queue.writeBuffer(this.parameters,8,new Uint32Array([value?1:0]));}
 encode(pass:GPUComputePassEncoder,tick:number){
  if(tick%this.interval)return;
  if(this.fresh){this.anatomy.encode(pass,this.imageIterations);this.fresh=false;}
  pass.setPipeline(this.feedback);pass.setBindGroup(0,this.feedbackGroup);pass.dispatchWorkgroups(Math.ceil(this.circuit.n/128));
  this.currents.encode(pass,this.currentIterations);
  pass.setPipeline(this.apply);pass.setBindGroup(0,this.applyGroup);pass.dispatchWorkgroups(Math.ceil(this.inputIds.length/128));
 }
 reset(){this.anatomy.reset();this.currents.reset();this.fresh=true;this.circuit.device.queue.writeBuffer(this.integral,0,new Float32Array(this.circuit.n));this.circuit.setTonicInputs(this.inputIds,new Float32Array(this.inputIds.length));}
 destroy(){this.anatomy.destroy();this.currents.destroy();this.buffers.forEach(b=>b.destroy());}
}
