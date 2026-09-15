import {GpuCircuit} from './gpu-circuit';
import {GpuSparseSolver,type SparseTriplets} from './gpu-solver';
import type {SynapticControlMetadata} from './synaptic-controller';

const shader=`
struct Params {outputs:u32,inputs:u32,muted:u32,rateScale:f32,feedbackSeconds:f32,inputMode:u32,traceToHz:f32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> outputMap:array<u32>;
@group(0) @binding(2) var<storage,read> inputMap:array<u32>;
@group(0) @binding(3) var<storage,read> desiredActivity:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> currentTarget:array<vec2<f32>>;
@group(0) @binding(5) var<storage,read_write> integral:array<f32>;
@group(0) @binding(6) var<storage,read> cells:array<Cell>;
@group(0) @binding(7) var<storage,read> currentControl:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> drive:array<vec2<f32>>;
@compute @workgroup_size(128)
fn feedback(@builtin(global_invocation_id) g:vec3<u32>){
  let row=g.x;if(row>=p.outputs){return;}
  let desired=p.rateScale*desiredActivity[row].x;let actual=cells[outputMap[row]].trace*p.traceToHz;
  let interval=max(.1,1000.0/max(desired,.001)-2.2);
  let ff=select(0.0,7.0/(1.0-20.0/15.0*exp(-interval/20.0)+5.0/15.0*exp(-interval/5.0)),desired>0.0);
  let error=desired-actual;
  if(p.muted==0u){integral[row]=clamp(integral[row]+error*p.feedbackSeconds*4.0,-250.0,250.0);}
  currentTarget[row].x=clamp(ff+integral[row]+.5*error,-300.0,200.0);
}
@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) g:vec3<u32>){
  let column=g.x;if(column>=p.inputs){return;}let current=currentControl[column];
  drive[inputMap[column]]=vec2(select(4000.0*current.x/current.z,0.0,p.muted!=0u),select(f32(p.inputMode),0.0,p.muted!=0u));
}
`;

export class GpuImageController {
  readonly image:GpuSparseSolver;
  readonly current:GpuSparseSolver;
  readonly outputIds:number[];
  private circuit:GpuCircuit;
  private feedback:GPUComputePipeline;
  private apply:GPUComputePipeline;
  private group:GPUBindGroup;
  private integral:GPUBuffer;
  private parameters:GPUBuffer;
  private buffers:GPUBuffer[]=[];
  private freshFrame=false;
  private firstControl=true;
  readonly rateScale:number;
  private feedbackTicks:number;
  private imageIterations:number;
  private controlIterations:number;
  constructor(circuit:GpuCircuit,meta:SynapticControlMetadata,operator:SparseTriplets,motorIds:number[]=[],options:{rateScale?:number;feedbackTicks?:number;imageIterations?:number;controlIterations?:number;inputMode?:number}={}){
    this.rateScale=options.rateScale??60;this.feedbackTicks=options.feedbackTicks??25;this.imageIterations=options.imageIterations??40;this.controlIterations=options.controlIterations??6;
    this.circuit=circuit;this.outputIds=meta.outputIds;
    const forbidden=new Set([...meta.outputIds,...motorIds]);
    if(meta.inputIds.some(id=>forbidden.has(id)))throw Error('Direct video stimulation of displayed or motor-output neurons is forbidden.');
    if(operator.columns!==meta.outputIds.length)throw Error('Anatomy and output identities differ.');
    const device=circuit.device;
    this.image=new GpuSparseSolver(device,operator,new Float32Array(operator.columns).fill(1));
    this.current=new GpuSparseSolver(device,{rows:meta.outputIds.length,columns:meta.inputIds.length,row:meta.row,column:meta.column,weight:meta.weight},meta.strength.map(w=>4000*.001375*w));
    const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
      const buffer=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});
      if(typeof data!=='number'){new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));buffer.unmap();}
      this.buffers.push(buffer);return buffer;
    };
    this.integral=create(meta.outputIds.length*4);this.parameters=create(32,GPUBufferUsage.UNIFORM);
    const values=new Uint32Array([meta.outputIds.length,meta.inputIds.length,0,0,0,0,0,0]);new Float32Array(values.buffer)[3]=this.rateScale;new Float32Array(values.buffer)[4]=this.feedbackTicks*.0002;values[5]=options.inputMode??1;new Float32Array(values.buffer)[6]=1000/circuit.traceMs;
    device.queue.writeBuffer(this.parameters,0,values);
    const module=device.createShaderModule({code:shader});
    const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',minBindingSize:32}},
      ...[1,2,3,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage' as const}})),
      ...[4,5,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}))]});
    this.feedback=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'feedback'}});
    this.apply=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'apply'}});
    const buffers=[this.parameters,create(Uint32Array.from(meta.outputIds)),create(Uint32Array.from(meta.inputIds)),this.image.states,this.current.residual,this.integral,circuit.cells,this.current.states,circuit.inputs];
    this.group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    circuit.setInputs(meta.inputIds,new Float32Array(meta.inputIds.length));
  }
  setFrame(pixels:ArrayLike<number>){this.image.setTarget(pixels);this.freshFrame=true;}
  setMuted(muted:boolean){this.circuit.device.queue.writeBuffer(this.parameters,8,new Uint32Array([muted?1:0]));}
  encode(pass:GPUComputePassEncoder,tick:number){
    if(this.freshFrame){this.image.encode(pass,this.imageIterations);this.freshFrame=false;}
    if(tick%this.feedbackTicks!==0)return;
    pass.setPipeline(this.feedback);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(Math.ceil(this.outputIds.length/128));
    this.current.encode(pass,this.firstControl?60:this.controlIterations);this.firstControl=false;
    pass.setPipeline(this.apply);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(Math.ceil(this.current.columns/128));
  }
  reset(){this.image.reset();this.current.reset();this.circuit.device.queue.writeBuffer(this.integral,0,new Float32Array(this.outputIds.length));this.firstControl=true;this.freshFrame=true;}
  destroy(){this.image.destroy();this.current.destroy();this.buffers.forEach(b=>b.destroy());}
}
