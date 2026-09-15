import {GpuCircuit} from './gpu-circuit';
import type {SparseTriplets} from './gpu-solver';

const code=`
struct Params {neurons:u32,pixels:u32,inputs:u32,muted:u32,traceScale:f32,step:f32,maxDrive:f32,gain:f32,method:u32,rateFloor:f32,traceToHz:f32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> rowPointer:array<u32>;
@group(0) @binding(2) var<storage,read> rowEntries:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read> columnPointer:array<u32>;
@group(0) @binding(4) var<storage,read> columnEntries:array<vec2<u32>>;
@group(0) @binding(5) var<storage,read> cells:array<Cell>;
@group(0) @binding(6) var<storage,read> desired:array<f32>;
@group(0) @binding(7) var<storage,read_write> pixelError:array<f32>;
@group(0) @binding(8) var<storage,read_write> ownError:array<vec2<f32>>;
@group(0) @binding(9) var<storage,read> errorIn:array<vec2<f32>>;
@group(0) @binding(10) var<storage,read_write> errorOut:array<vec2<f32>>;
@group(0) @binding(11) var<storage,read> offsets:array<u32>;
@group(0) @binding(12) var<storage,read> synapses:array<vec2<u32>>;
@group(0) @binding(13) var<storage,read> inputIds:array<u32>;
@group(0) @binding(14) var<storage,read_write> drive:array<vec2<f32>>;
@group(0) @binding(15) var<storage,read> calibration:array<f32>;

@compute @workgroup_size(128)
fn project(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.pixels){return;}var actual=0.0;
 for(var e=rowPointer[row];e<rowPointer[row+1u];e++){
  let item=rowEntries[e];actual+=bitcast<f32>(item.y)*clamp(cells[item.x].trace*p.traceScale/calibration[item.x]-p.rateFloor,0.0,1.0);
 }
 pixelError[row]=desired[row]-actual;
}
@compute @workgroup_size(128)
fn backproject(@builtin(global_invocation_id) g:vec3<u32>){
 let neuron=g.x;if(neuron>=p.neurons){return;}var error=0.0;var scale=0.0;
 for(var e=columnPointer[neuron];e<columnPointer[neuron+1u];e++){
  let item=columnEntries[e];let w=bitcast<f32>(item.y);error+=w*pixelError[item.x];scale+=w;
 }
 let value=select(vec2(error/max(scale,.00000001),1.0),vec2(error,scale),p.method!=0u);ownError[neuron]=value;errorOut[neuron]=value;
}
@compute @workgroup_size(128)
fn reverseConnections(@builtin(global_invocation_id) g:vec3<u32>){
 let pre=g.x;if(pre>=p.neurons){return;}var signal=0.0;var scale=0.0;
 for(var e=offsets[pre];e<offsets[pre+1u];e++){
  let edge=synapses[e];var w=bitcast<f32>(edge.y);
  if(p.method!=0u){
   let rate=clamp(cells[edge.x].trace*p.traceToHz,5.0,250.0);
   let a=exp(-(1000.0/rate-2.2)/20.0);
   let slope=clamp(20.0*rate*rate*(1.0-a)*(1.0-a)/(7000.0*a),2.0,30.0);
   w*=.001375*slope;
  }
  signal+=w*errorIn[edge.x].x;scale+=abs(w)*errorIn[edge.x].y;
 }
 errorOut[pre]=select(vec2(clamp(ownError[pre].x+p.gain*signal/max(scale,1.0),-2.0,2.0),1.0),ownError[pre]+p.gain*vec2(signal,scale),p.method!=0u);
}
@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.inputs){return;}let id=inputIds[i];let old=drive[id].x;
 let next=clamp(old+p.step*errorIn[id].x/max(errorIn[id].y,.00000001)-0.0005*old,-p.maxDrive,p.maxDrive);
 drive[id]=vec2(select(next,0.0,p.muted!=0u),3.0);
}
`;

/** Experimental image feedback through the complete graph. The loss observes
 * every neuron's full cable footprint, including externally stimulated cells.
 * It only changes bounded tonic input currents. It never writes voltages,
 * refractory periods, synaptic state or displayed activity. Reverse graph
 * propagation is a heuristic control sensitivity, not an exact LIF gradient.
 */
export class GpuWholeBrainController {
  readonly inputIds:Uint32Array;
  readonly rateScale:number;
  readonly rateFloor:number;
  rateCalibration:Float32Array<ArrayBuffer>;
  private calibration:GPUBuffer;
  private buffers:GPUBuffer[]=[];
  private parameters:GPUBuffer;
  private target:GPUBuffer;
  private own:GPUBuffer;
  private errors:GPUBuffer[];
  private project:GPUComputePipeline;
  private backproject:GPUComputePipeline;
  private reverse:GPUComputePipeline;
  private apply:GPUComputePipeline;
  private projectGroup:GPUBindGroup;
  private backGroup:GPUBindGroup;
  private reverseGroups:GPUBindGroup[]=[];
  private applyGroup:GPUBindGroup;
  private feedbackTicks:number;
  private depth:number;
  private muted=false;
  private subgroup:boolean;

  constructor(private circuit:GpuCircuit,h:SparseTriplets,motorIds:number[],options:{rateScale?:number;feedbackTicks?:number;depth?:number;step?:number;gain?:number;maxDrive?:number;sensitivity?:number;rateFloor?:number}={}){
    if(h.columns!==circuit.n)throw Error('Whole-brain controller must observe every model neuron.');
    this.rateScale=options.rateScale??120;this.rateFloor=options.rateFloor??0;this.feedbackTicks=options.feedbackTicks??50;this.depth=options.depth??3;
    const motor=new Set(motorIds),ids:number[]=[];
    for(let i=0;i<circuit.n;i++)if(!motor.has(i)&&((Math.imul(i+1,2654435761)>>>0)<858993459))ids.push(i);
    this.inputIds=Uint32Array.from(ids);
    const device=circuit.device;this.subgroup=device.features.has('subgroups')&&device.adapterInfo.subgroupMinSize===32&&device.adapterInfo.subgroupMaxSize===32;
    const create=(value:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
      const b=device.createBuffer({size:Math.max(4,typeof value==='number'?value:value.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof value!=='number'});
      if(typeof value!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));b.unmap();}
      this.buffers.push(b);return b;
    };
    const compressed=(major:ArrayLike<number>,minor:ArrayLike<number>,n:number)=>{
      const pointer=new Uint32Array(n+1);for(let e=0;e<major.length;e++)pointer[major[e]+1]++;
      for(let i=0;i<n;i++)pointer[i+1]+=pointer[i];
      const cursor=pointer.slice(),entry=new Uint32Array(major.length*2),weight=new Float32Array(entry.buffer);
      for(let e=0;e<major.length;e++){const k=cursor[major[e]]++;entry[k*2]=minor[e];weight[k*2+1]=h.weight[e];}
      return [create(pointer),create(entry)];
    };
    const [rp,re]=compressed(h.row,h.column,h.rows),[cp,ce]=compressed(h.column,h.row,h.columns);
    const synapses=new Uint32Array(circuit.edgeCount*2),weights=new Float32Array(synapses.buffer);
    for(let e=0;e<circuit.edgeCount;e++){synapses[e*2]=circuit.targets[e];weights[e*2+1]=circuit.weights[e];}
    const edges=create(synapses),offsets=create(circuit.offsets),inputIds=create(this.inputIds);
    this.target=create(h.rows*4);const residual=create(h.rows*4);this.own=create(circuit.n*8);
    this.errors=[create(circuit.n*8),create(circuit.n*8)];
    const params=new Uint32Array([circuit.n,h.rows,ids.length,0,0,0,0,0,options.sensitivity??0,0,0,0]),floats=new Float32Array(params.buffer);
    floats[4]=1000/circuit.traceMs/this.rateScale;floats[5]=options.step??5;floats[6]=options.maxDrive??50;floats[7]=options.gain??1.3;floats[9]=this.rateFloor/this.rateScale;floats[10]=1000/circuit.traceMs;
    this.parameters=create(params,GPUBufferUsage.UNIFORM);this.rateCalibration=new Float32Array(circuit.n).fill(1);this.calibration=create(this.rateCalibration);
    let shader=code;
    if(this.subgroup){
      shader='enable subgroups;\n'+shader;
      for(const name of ['project','backproject','reverseConnections'])shader=shader.replace(`@compute @workgroup_size(128)\nfn ${name}(@builtin(global_invocation_id) g:vec3<u32>)`, `@compute @workgroup_size(32)\nfn ${name}(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32)`);
      shader=shader.replace('let row=g.x;', 'let row=g.x+g.y*65535u;').replace('let neuron=g.x;', 'let neuron=g.x+g.y*65535u;').replace('let pre=g.x;', 'let pre=g.x+g.y*65535u;');
      shader=shader.replace('e=rowPointer[row];e<rowPointer[row+1u];e++', 'e=rowPointer[row]+lane;e<rowPointer[row+1u];e+=32u');
      shader=shader.replace('e=columnPointer[neuron];e<columnPointer[neuron+1u];e++', 'e=columnPointer[neuron]+lane;e<columnPointer[neuron+1u];e+=32u');
      shader=shader.replace('e=offsets[pre];e<offsets[pre+1u];e++', 'e=offsets[pre]+lane;e<offsets[pre+1u];e+=32u');
      shader=shader.replace('pixelError[row]=', 'actual=subgroupAdd(actual);if(lane!=0u){return;}pixelError[row]=');
      shader=shader.replace('let value=select', 'error=subgroupAdd(error);scale=subgroupAdd(scale);if(lane!=0u){return;}let value=select');
      shader=shader.replace('errorOut[pre]=', 'signal=subgroupAdd(signal);scale=subgroupAdd(scale);if(lane!=0u){return;}errorOut[pre]=');
    }
    const module=device.createShaderModule({code:shader});
    const pipeline=(entryPoint:string)=>device.createComputePipeline({layout:'auto',compute:{module,entryPoint}});
    this.project=pipeline('project');this.backproject=pipeline('backproject');this.reverse=pipeline('reverseConnections');this.apply=pipeline('apply');
    const bind=(p:GPUComputePipeline,entries:Array<[number,GPUBuffer]>)=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))});
    this.projectGroup=bind(this.project,[[0,this.parameters],[1,rp],[2,re],[5,circuit.cells],[6,this.target],[7,residual],[15,this.calibration]]);
    this.backGroup=bind(this.backproject,[[0,this.parameters],[3,cp],[4,ce],[7,residual],[8,this.own],[10,this.errors[0]]]);
    for(let i=0;i<this.depth;i++)this.reverseGroups.push(bind(this.reverse,[[0,this.parameters],[8,this.own],[9,this.errors[i%2]],[10,this.errors[(i+1)%2]],[11,offsets],[12,edges],[5,circuit.cells] ]));
    this.applyGroup=bind(this.apply,[[0,this.parameters],[9,this.errors[this.depth%2]],[13,inputIds],[14,circuit.inputs]]);
    this.reset();
  }
  setRateCalibration(rates:Float32Array<ArrayBuffer>){if(rates.length!==this.circuit.n||rates.some(x=>!Number.isFinite(x)||x<10))throw Error('Invalid measured rate calibration.');this.rateCalibration=Float32Array.from(rates,x=>x/this.rateScale);this.circuit.device.queue.writeBuffer(this.calibration,0,this.rateCalibration);}
  setFrame(values:Float32Array<ArrayBuffer>){if(values.length*4!==this.target.size)throw Error('Whole-brain target dimensions differ.');this.circuit.device.queue.writeBuffer(this.target,0,values);}
  setMuted(value:boolean){this.muted=value;this.circuit.device.queue.writeBuffer(this.parameters,12,new Uint32Array([value?1:0]));}
  encode(pass:GPUComputePassEncoder,tick:number){
    if(tick%this.feedbackTicks)return;
    const dispatch=(rows:number)=>pass.dispatchWorkgroups(this.subgroup?Math.min(65535,rows):Math.ceil(rows/128),this.subgroup?Math.ceil(rows/65535):1);
    pass.setPipeline(this.project);pass.setBindGroup(0,this.projectGroup);dispatch(this.target.size/4);
    pass.setPipeline(this.backproject);pass.setBindGroup(0,this.backGroup);dispatch(this.circuit.n);
    for(const group of this.reverseGroups){pass.setPipeline(this.reverse);pass.setBindGroup(0,group);dispatch(this.circuit.n);}
    pass.setPipeline(this.apply);pass.setBindGroup(0,this.applyGroup);pass.dispatchWorkgroups(Math.ceil(this.inputIds.length/128));
  }
  reset(){this.circuit.setTonicInputs(this.inputIds,new Float32Array(this.inputIds.length).fill(this.muted?0:8));}
  destroy(){this.buffers.forEach(b=>b.destroy());}
}
