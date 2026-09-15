import {GpuAdmmSolver} from './gpu-admm-solver';
import {GpuCircuit} from './gpu-circuit';
import {GpuSparseSolver,type SparseTriplets} from './gpu-solver';
import type {SynapticControlMetadata} from './synaptic-controller';

/** Image-space feedback on observed spike traces. Averages at shared anatomical
 * pixels are controlled jointly, rather than demanding identical rates from
 * every cell there. The only outputs of this controller are upstream inputs. */
const code=`
struct Params {pixels:u32,inputs:u32,muted:u32,rateScale:f32,seconds:f32,inputMode:u32,compensate:u32,robust:u32,traceToHz:f32,rateMode:u32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> pointers:array<u32>;
@group(0) @binding(2) var<storage,read> entries:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read> cells:array<Cell>;
@group(0) @binding(4) var<storage,read> desiredPixels:array<vec2<f32>>;
@group(0) @binding(5) var<storage,read_write> integral:array<f32>;
@group(0) @binding(6) var<storage,read_write> currentTarget:array<vec2<f32>>;
@group(0) @binding(7) var<storage,read> control:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> drive:array<vec2<f32>>;
@group(0) @binding(9) var<storage,read> inputMap:array<u32>;
@group(0) @binding(10) var<storage,read> backgroundPointer:array<u32>;
@group(0) @binding(11) var<storage,read> backgroundEntries:array<vec2<u32>>;
@compute @workgroup_size(128)
fn changeReference(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.pixels){integral[g.x]*=desiredPixels[g.x].y;}}
@compute @workgroup_size(128)
fn feedback(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.pixels){return;}
 var actual=0.0;for(var i=pointers[row];i<pointers[row+1u];i++){let e=entries[i];actual+=bitcast<f32>(e.y)*min(p.rateScale,cells[e.x].trace*p.traceToHz);}
 let desired=desiredPixels[row].x*p.rateScale;let error=desired-actual;
 // Exact discrete mean-current response: the current accumulates for ten
 // blocked ticks after reset, then both state variables integrate on tick 11.
 // Ignoring this refractory accumulation badly overdrives bright pixels.
 let freeTicks=max(1.0,(1000.0/max(desired,.001)-2.0)/.2);
 let a=0.9900498337491681;let b=0.9607894391523232;
 let am=pow(a,freeTicks);let bm=pow(b,freeTicks);
 let response=.009753464865615247*.04/(1.0-b)*((1.0-am)/(1.0-a)+(10.0*(1.0-b)-b)*(am-bm)/(a-b));
 let ff=select(select(0.0,7.0/response,desired>0.0),desired,p.rateMode!=0u);
 if(p.muted==0u){if(p.robust!=0u){integral[row]=clamp(integral[row]+(error*4.0+20.0*currentTarget[row].y)*p.seconds,-50.0,50.0);}else{integral[row]=clamp(integral[row]+error*p.seconds*4.0,-250.0,250.0);}}
 var background=0.0;
 if(p.compensate!=0u){for(var i=backgroundPointer[row];i<backgroundPointer[row+1u];i++){let e=backgroundEntries[i];background+=bitcast<f32>(e.y)*cells[e.x].trace*p.traceToHz;}}
 currentTarget[row].x=clamp(ff+integral[row]+.5*error-background,-1000.0,1000.0);
}
@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.inputs){return;}let q=control[i];
 drive[inputMap[i]]=vec2(select(4000.0*q.x/max(q.z,.000001),0.0,p.muted!=0u),select(f32(p.inputMode),0.0,p.muted!=0u));
}
`;
export class GpuPixelController{
 readonly current:GpuSparseSolver|GpuAdmmSolver;
 readonly outputIds:number[];
 readonly rateScale:number;
 private circuit:GpuCircuit;
 private target:GPUBuffer;
 private integral:GPUBuffer;
 private parameters:GPUBuffer;
 private feedback:GPUComputePipeline;
 private apply:GPUComputePipeline;
 private feedbackGroup:GPUBindGroup;
 private applyGroup:GPUBindGroup;
 private buffers:GPUBuffer[]=[];
 private first=true;
 private fresh=false;
 private previousFrame:Float32Array;
 private changeReference:GPUComputePipeline;
 private changeGroup:GPUBindGroup;
 private interval:number;
 private iterations:number;
 private initialIterations:number;private maintenance:number;
 private pixels:number;
 constructor(circuit:GpuCircuit,meta:SynapticControlMetadata,h:SparseTriplets,motorIds:number[]=[],options:{rateScale?:number;inputMode?:number;feedbackTicks?:number;controlIterations?:number;compensate?:boolean;robust?:boolean;gradientLimit?:number;ridge?:number;inputRatePenalty?:number;kp?:number;ki?:number;antiwindup?:number;integralLimit?:number;initialIterations?:number;maintenance?:number;maxControlDegree?:number;rateMode?:boolean;inverse?:Float32Array;admmRho?:number}={}){
  this.circuit=circuit;this.outputIds=meta.outputIds;this.rateScale=options.rateScale??120;this.interval=options.feedbackTicks??25;this.iterations=options.controlIterations??6;this.pixels=h.rows;this.initialIterations=options.initialIterations??64;this.maintenance=options.maintenance??Math.max(2,this.iterations-6);this.previousFrame=new Float32Array(h.rows);
  if(h.columns!==meta.outputIds.length)throw Error('Observation identities differ.');
  const forbidden=new Set([...meta.outputIds,...motorIds]);if(meta.inputIds.some(id=>forbidden.has(id)))throw Error('Displayed and motor neurons cannot be actuators.');
  const pixelForNeuron=new Int32Array(h.columns).fill(-1),gain=new Float32Array(h.columns),pointer=new Uint32Array(h.rows+1);
  for(let i=0;i<h.row.length;i++){if(pixelForNeuron[h.column[i]]>=0)throw Error('Pixel controller requires one fixed observation site per neuron.');pixelForNeuron[h.column[i]]=h.row[i];gain[h.column[i]]=h.weight[i];pointer[h.row[i]+1]++;}
  for(let i=0;i<h.rows;i++)pointer[i+1]+=pointer[i];
  const packed=new Uint32Array(h.row.length*2),floats=new Float32Array(packed.buffer),cursor=pointer.slice();
  for(let i=0;i<h.row.length;i++){const k=cursor[h.row[i]]++;packed[k*2]=meta.outputIds[h.column[i]];floats[k*2+1]=h.weight[i];}
  const degree=new Uint32Array(meta.inputIds.length);for(const column of meta.column)degree[column]++;const allowed=Array.from(degree,d=>d<=(options.maxControlDegree??Infinity));
  const row:number[]=[],column:number[]=[],weight:number[]=[];
  for(let i=0;i<meta.row.length;i++){const output=meta.row[i],pixel=pixelForNeuron[output];if(pixel<0||!allowed[meta.column[i]])continue;row.push(pixel);column.push(meta.column[i]);weight.push((options.rateMode?Math.max(-1,Math.min(1,meta.weight[i]*meta.strength[meta.column[i]]/160)):meta.weight[i])*gain[output]);}
  const backgroundRow:number[]=[],backgroundPre:number[]=[],backgroundWeight:number[]=[];
  const modelToOutput=new Int32Array(circuit.n).fill(-1);meta.outputIds.forEach((id,i)=>modelToOutput[id]=i);const controlled=new Set(meta.inputIds);
  for(let pre=0;pre<circuit.n;pre++){if(controlled.has(pre))continue;for(let e=circuit.offsets[pre];e<circuit.offsets[pre+1];e++){const output=modelToOutput[circuit.targets[e]];if(output<0)continue;const pixel=pixelForNeuron[output];if(pixel<0||!circuit.weights[e])continue;backgroundRow.push(pixel);backgroundPre.push(pre);backgroundWeight.push((options.rateMode?Math.max(-1,Math.min(1,circuit.weights[e]/160)):circuit.weights[e]*.001375)*gain[output]);}}
  const bp=new Uint32Array(h.rows+1);for(const row of backgroundRow)bp[row+1]++;for(let i=0;i<h.rows;i++)bp[i+1]+=bp[i];const bcursor=bp.slice(),be=new Uint32Array(backgroundRow.length*2),bf=new Float32Array(be.buffer);for(let i=0;i<backgroundRow.length;i++){const k=bcursor[backgroundRow[i]]++;be[k*2]=backgroundPre[i];bf[k*2+1]=backgroundWeight[i];}
  const device=circuit.device;
  const matrix={rows:h.rows,columns:meta.inputIds.length,row,column,weight},upper=meta.strength.map((w,i)=>allowed[i]?(options.rateMode?4000:4000*.001375*w):0);
  this.current=options.inverse?new GpuAdmmSolver(device,matrix,upper,options.inverse,{rho:options.admmRho??.1,ridge:options.ridge??.001,gradientLimit:options.gradientLimit??5}):new GpuSparseSolver(device,matrix,upper,options.robust?{gradientLimit:options.gradientLimit??5,ridge:options.inputRatePenalty?meta.strength.map(w=>(options.ridge??.001)+options.inputRatePenalty!/(.001375*w)**2):options.ridge??.001}:{});
  const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{const b=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});if(typeof data!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();}this.buffers.push(b);return b;};
  this.target=create(h.rows*8);this.integral=create(h.rows*4);const values=new Uint32Array([h.rows,meta.inputIds.length,0,0,0,options.inputMode??2,options.compensate?1:0,options.robust?1:0,0,options.rateMode?1:0,0,0]);new Float32Array(values.buffer)[3]=this.rateScale;new Float32Array(values.buffer)[4]=this.interval*.0002;new Float32Array(values.buffer)[8]=1000/circuit.traceMs;this.parameters=create(values,GPUBufferUsage.UNIFORM);
  const configured=code.replaceAll('error*4.0',`error*${(options.ki??4).toFixed(6)}`).replaceAll('20.0*currentTarget',`${(options.antiwindup??20).toFixed(6)}*currentTarget`).replaceAll('+.5*error',`+${(options.kp??.5).toFixed(6)}*error`).replaceAll('-50.0,50.0',`-${(options.integralLimit??50).toFixed(6)},${(options.integralLimit??50).toFixed(6)}`);
  const module=device.createShaderModule({code:configured});this.feedback=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'feedback'}});this.changeReference=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'changeReference'}});this.apply=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'apply'}});
  const bind=(pipeline:GPUComputePipeline,entries:Array<[number,GPUBuffer]>)=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))});
  this.changeGroup=bind(this.changeReference,[[0,this.parameters],[4,this.target],[5,this.integral]]);
  this.feedbackGroup=bind(this.feedback,[[0,this.parameters],[1,create(pointer)],[2,create(packed)],[3,circuit.cells],[4,this.target],[5,this.integral],[6,this.current.residual],[10,create(bp)],[11,create(be)]]);
  this.applyGroup=bind(this.apply,[[0,this.parameters],[7,this.current.states],[8,circuit.inputs],[9,create(Uint32Array.from(meta.inputIds))]]);
  circuit.setInputs(meta.inputIds,new Float32Array(meta.inputIds.length));
 }
 setFrame(values:ArrayLike<number>){if(values.length!==this.pixels)throw Error('Target dimensions differ.');const packed=new Float32Array(values.length*2);for(let i=0;i<values.length;i++){packed[2*i]=values[i];packed[2*i+1]=Math.exp(-8*Math.abs(values[i]-this.previousFrame[i]));this.previousFrame[i]=values[i];}this.circuit.device.queue.writeBuffer(this.target,0,packed);this.fresh=true;}
 setMuted(muted:boolean){this.circuit.device.queue.writeBuffer(this.parameters,8,new Uint32Array([muted?1:0]));}
 encode(pass:GPUComputePassEncoder,tick:number){if(tick%this.interval!==0)return;if(this.fresh){pass.setPipeline(this.changeReference);pass.setBindGroup(0,this.changeGroup);pass.dispatchWorkgroups(Math.ceil(this.pixels/128));}pass.setPipeline(this.feedback);pass.setBindGroup(0,this.feedbackGroup);pass.dispatchWorkgroups(Math.ceil(this.pixels/128));this.current.encode(pass,this.current instanceof GpuAdmmSolver?(this.first||this.fresh?this.iterations:1):(this.first||this.fresh?this.initialIterations:this.maintenance));this.first=false;this.fresh=false;pass.setPipeline(this.apply);pass.setBindGroup(0,this.applyGroup);pass.dispatchWorkgroups(Math.ceil(this.current.columns/128));}
 reset(){this.previousFrame.fill(0);this.current.reset();this.circuit.device.queue.writeBuffer(this.integral,0,new Float32Array(this.pixels));this.first=true;}
 destroy(){this.current.destroy();this.buffers.forEach(b=>b.destroy());}
}
