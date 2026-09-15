/// <reference types="@webgpu/types" />

export interface SparseTriplets {rows:number;columns:number;row:ArrayLike<number>;column:ArrayLike<number>;weight:ArrayLike<number>;imageWidth?:number}
const code=`
struct Params {rows:u32,columns:u32,momentum:f32,gradientLimit:f32,observationScale:f32,observationStep:f32,observationAnchor:f32,spatialWeight:f32,imageWidth:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> pointer:array<u32>;
@group(0) @binding(2) var<storage,read> entries:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> variables:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> residual:array<vec2<f32>>;
@group(0) @binding(5) var<storage,read> regularization:array<f32>;
@group(0) @binding(6) var<storage,read> observations:array<f32>;
@group(0) @binding(7) var<storage,read_write> filtered:array<f32>;
@group(0) @binding(8) var<storage,read> support:array<f32>;

fn errorAt(row:u32)->f32{if(p.spatialWeight>0.0){return filtered[row];}return residual[row].y;}
fn difference(row:u32,other:u32)->f32{return select(0.0,residual[row].y-residual[other].y,support[other]>0.0);}
@compute @workgroup_size(128)
fn spatialGradient(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.rows){return;}if(support[row]<=0.0){filtered[row]=0.0;return;}
 var laplacian=0.0;
 if(row%p.imageWidth>0u){laplacian+=difference(row,row-1u);}
 if(row%p.imageWidth+1u<p.imageWidth){laplacian+=difference(row,row+1u);}
 if(row>=p.imageWidth){laplacian+=difference(row,row-p.imageWidth);}
 if(row+p.imageWidth<p.rows){laplacian+=difference(row,row+p.imageWidth);}
 filtered[row]=residual[row].y+p.spatialWeight*laplacian;
}

@compute @workgroup_size(128)
fn restart(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.columns){variables[g.x].y=variables[g.x].x;}}
@compute @workgroup_size(128)
fn project(@builtin(global_invocation_id) g:vec3<u32>){
  let row=g.x;if(row>=p.rows){return;}var sum=0.0;
  for(var e=pointer[row];e<pointer[row+1u];e++){let item=entries[e];sum+=bitcast<f32>(item.y)*variables[item.x].y;}
  residual[row].y=clamp(sum-residual[row].x,-p.gradientLimit,p.gradientLimit);
}
@compute @workgroup_size(128)
fn descend(@builtin(global_invocation_id) g:vec3<u32>){
  let col=g.x;if(col>=p.columns){return;}var grad=0.0;
  for(var e=pointer[col];e<pointer[col+1u];e++){let item=entries[e];grad+=bitcast<f32>(item.y)*errorAt(item.x);}
  let old=variables[col];let value=clamp(old.y-(grad+regularization[col]*old.y)/old.w,0.0,old.z);
  variables[col]=vec4(value,value+p.momentum*(value-old.x),old.z,old.w);
}
@compute @workgroup_size(128)
fn projectObserved(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.rows){return;}var sum=0.0;
 for(var e=pointer[row];e<pointer[row+1u];e++){let item=entries[e];sum+=bitcast<f32>(item.y)*clamp(observations[item.x*5u+2u]*p.observationScale,0.0,1.0);}
 residual[row].y=sum-residual[row].x;
}
@compute @workgroup_size(128)
fn correctObserved(@builtin(global_invocation_id) g:vec3<u32>){
 let col=g.x;if(col>=p.columns){return;}var grad=0.0;
 for(var e=pointer[col];e<pointer[col+1u];e++){let item=entries[e];grad+=bitcast<f32>(item.y)*errorAt(item.x);}
 let old=variables[col];let baseline=mix(old.x,clamp(observations[col*5u+2u]*p.observationScale,0.0,1.0),p.observationAnchor);
 let value=clamp(baseline-p.observationStep*grad/old.w,0.0,old.z);variables[col]=vec4(value,grad/old.w,old.z,old.w);
}
`;
const subgroupCode=`
@compute @workgroup_size(32)
fn projectRows(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let row=g.x+g.y*65535u;if(row>=p.rows){return;}var sum=0.0;
  for(var e=pointer[row]+lane;e<pointer[row+1u];e+=32u){let item=entries[e];sum+=bitcast<f32>(item.y)*variables[item.x].y;}
  let total=subgroupAdd(sum);if(lane==0u){residual[row].y=clamp(total-residual[row].x,-p.gradientLimit,p.gradientLimit);}
}
@compute @workgroup_size(32)
fn descendColumns(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let col=g.x+g.y*65535u;if(col>=p.columns){return;}var grad=0.0;
  for(var e=pointer[col]+lane;e<pointer[col+1u];e+=32u){let item=entries[e];grad+=bitcast<f32>(item.y)*errorAt(item.x);}
  let total=subgroupAdd(grad);
  if(lane==0u){let old=variables[col];let value=clamp(old.y-(total+regularization[col]*old.y)/old.w,0.0,old.z);variables[col]=vec4(value,value+p.momentum*(value-old.x),old.z,old.w);}
}
@compute @workgroup_size(32)
fn projectObservedRows(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let row=g.x+g.y*65535u;if(row>=p.rows){return;}var sum=0.0;
 for(var e=pointer[row]+lane;e<pointer[row+1u];e+=32u){let item=entries[e];sum+=bitcast<f32>(item.y)*clamp(observations[item.x*5u+2u]*p.observationScale,0.0,1.0);}
 let total=subgroupAdd(sum);if(lane==0u){residual[row].y=total-residual[row].x;}
}
@compute @workgroup_size(32)
fn correctObservedColumns(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let col=g.x+g.y*65535u;if(col>=p.columns){return;}var grad=0.0;
 for(var e=pointer[col]+lane;e<pointer[col+1u];e+=32u){let item=entries[e];grad+=bitcast<f32>(item.y)*errorAt(item.x);}
 let total=subgroupAdd(grad);if(lane==0u){let old=variables[col];let baseline=mix(old.x,clamp(observations[col*5u+2u]*p.observationScale,0.0,1.0),p.observationAnchor);let value=clamp(baseline-p.observationStep*total/old.w,0.0,old.z);variables[col]=vec4(value,total/old.w,old.z,old.w);}
}
`;

/** GPU FISTA with a diagonal majorizer. Output variables are optimization
 * targets/control currents; they must never be used as observed neural activity.
 */
export class GpuSparseSolver {
  readonly device:GPUDevice;
  readonly states:GPUBuffer;
  readonly residual:GPUBuffer;
  readonly rows:number;
  readonly columns:number;
  private project:GPUComputePipeline;
  private descend:GPUComputePipeline;
  private restart:GPUComputePipeline;
  private projectObserved:GPUComputePipeline;
  private correctObserved:GPUComputePipeline;
  private spatialGradient:GPUComputePipeline;
  private spatialWeight:number;
  private rowGroup:GPUBindGroup;
  private columnGroup:GPUBindGroup;
  private uniforms:GPUBuffer;
  private initial:Float32Array<ArrayBuffer>;
  private buffers:GPUBuffer[]=[];
  private subgroup:boolean;
  constructor(device:GPUDevice,m:SparseTriplets,upper:ArrayLike<number>,options:{gradientLimit?:number;ridge?:number|ArrayLike<number>;observations?:GPUBuffer;observationScale?:number;observationStep?:number;observationAnchor?:number;spatialWeight?:number}={}){
    this.device=device;this.rows=m.rows;this.columns=m.columns;
    this.spatialWeight=options.spatialWeight??0;
    if(!Number.isFinite(this.spatialWeight)||this.spatialWeight<0||(this.spatialWeight>0&&(!Number.isInteger(m.imageWidth)||!m.imageWidth||m.imageWidth<1||m.rows%m.imageWidth)))throw Error('Spatial loss requires a finite weight and the full rectangular image dimensions.');
    this.subgroup=device.features.has('subgroups') && device.adapterInfo.subgroupMinSize===32 && device.adapterInfo.subgroupMaxSize===32;
    const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
      const buffer=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});
      if(typeof data!=='number'){new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));buffer.unmap();}
      this.buffers.push(buffer);return buffer;
    };
    const compressed=(major:ArrayLike<number>,minor:ArrayLike<number>,n:number)=>{
      const pointers=new Uint32Array(n+1);for(let e=0;e<major.length;e++)pointers[major[e]+1]++;
      for(let i=0;i<n;i++)pointers[i+1]+=pointers[i];
      const cursor=pointers.slice(),indices=new Uint32Array(major.length*2),floats=new Float32Array(indices.buffer);
      for(let e=0;e<major.length;e++){const slot=cursor[major[e]]++;indices[slot*2]=minor[e];floats[slot*2+1]=m.weight[e];}
      return [create(pointers),create(indices)];
    };
    const [rowPointer,rowEntries]=compressed(m.row,m.column,m.rows),[columnPointer,columnEntries]=compressed(m.column,m.row,m.columns);
    const rowSum=new Float64Array(m.rows),diagonal=new Float64Array(m.columns);
    for(let e=0;e<m.row.length;e++)rowSum[m.row[e]]+=Math.abs(m.weight[e]);
    // Gershgorin majorizer for H.T (I + lambda D.T D) H. D contains only
    // neighbor differences wholly inside the anatomical support. The absolute
    // row sums also make this valid for sparse operators with signed entries.
    const majorRow=rowSum.slice();
    if(this.spatialWeight){const width=m.imageWidth!;
      for(let row=0;row<m.rows;row++)if(rowSum[row]>0){
        const adjacent=[...(row%width>0?[row-1]:[]),...(row%width+1<width?[row+1]:[]),...(row>=width?[row-width]:[]),...(row+width<m.rows?[row+width]:[])];
        for(const other of adjacent)if(rowSum[other]>0)majorRow[row]+=this.spatialWeight*(rowSum[row]+rowSum[other]);
      }
    }
    for(let e=0;e<m.row.length;e++)diagonal[m.column[e]]+=Math.abs(m.weight[e])*majorRow[m.row[e]];
    const ridge=Float32Array.from({length:m.columns},(_,i)=>typeof options.ridge==='number'?options.ridge:options.ridge?.[i]??0);const ridgeBuffer=create(ridge);
    this.initial=new Float32Array(m.columns*4);
    for(let i=0;i<m.columns;i++){this.initial[i*4+2]=upper[i];this.initial[i*4+3]=Math.max(1e-8,diagonal[i]+ridge[i]);}
    this.states=create(this.initial,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    this.residual=create(m.rows*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const uniform=new ArrayBuffer(256*256),u=new Uint32Array(uniform),f=new Float32Array(uniform);let t=1;
    for(let i=0;i<256;i++){const next=(1+Math.sqrt(1+4*t*t))/2;u[i*64]=m.rows;u[i*64+1]=m.columns;f[i*64+2]=(t-1)/next;f[i*64+3]=options.gradientLimit??1e30;f[i*64+4]=options.observationScale??1;f[i*64+5]=options.observationStep??1;f[i*64+6]=options.observationAnchor??0;f[i*64+7]=this.spatialWeight;u[i*64+8]=m.imageWidth??0;t=next;}
    this.uniforms=create(new Uint8Array(uniform),GPUBufferUsage.UNIFORM);
    const module=device.createShaderModule({code:this.subgroup?'enable subgroups;\n'+code+subgroupCode:code});
    const layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:48}},
      ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage' as const}})),
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
      ...[3,4,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}))]});
    const pipeline=(entryPoint:string)=>device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
    this.project=pipeline(this.subgroup?'projectRows':'project');this.descend=pipeline(this.subgroup?'descendColumns':'descend');this.restart=pipeline('restart');
    this.projectObserved=pipeline(this.subgroup?'projectObservedRows':'projectObserved');this.correctObserved=pipeline(this.subgroup?'correctObservedColumns':'correctObserved');
    this.spatialGradient=pipeline('spatialGradient');
    const observed=options.observations??create(4);
    const filtered=create(m.rows*4),support=create(Float32Array.from(rowSum));
    const group=(pointer:GPUBuffer,entries:GPUBuffer)=>device.createBindGroup({layout,entries:[
      {binding:0,resource:{buffer:this.uniforms,size:48}},{binding:1,resource:{buffer:pointer}},{binding:2,resource:{buffer:entries}},
      {binding:3,resource:{buffer:this.states}},{binding:4,resource:{buffer:this.residual}},{binding:5,resource:{buffer:ridgeBuffer}},{binding:6,resource:{buffer:observed}},{binding:7,resource:{buffer:filtered}},{binding:8,resource:{buffer:support}}]});
    this.rowGroup=group(rowPointer,rowEntries);this.columnGroup=group(columnPointer,columnEntries);
  }
  setTarget(values:ArrayLike<number>){
    if(values.length!==this.rows)throw Error('Image target dimensions differ.');
    const packed=new Float32Array(this.rows*2);for(let i=0;i<values.length;i++)packed[i*2]=values[i];this.device.queue.writeBuffer(this.residual,0,packed);
  }
  encode(pass:GPUComputePassEncoder,iterations:number){
    pass.setPipeline(this.restart);pass.setBindGroup(0,this.columnGroup,[0]);pass.dispatchWorkgroups(Math.ceil(this.columns/128));
    for(let i=0;i<iterations;i++){
      pass.setPipeline(this.project);pass.setBindGroup(0,this.rowGroup,[i*256]);pass.dispatchWorkgroups(this.subgroup?Math.min(65535,this.rows):Math.ceil(this.rows/128),this.subgroup?Math.ceil(this.rows/65535):1);
      if(this.spatialWeight){pass.setPipeline(this.spatialGradient);pass.setBindGroup(0,this.rowGroup,[i*256]);pass.dispatchWorkgroups(Math.ceil(this.rows/128));}
      pass.setPipeline(this.descend);pass.setBindGroup(0,this.columnGroup,[i*256]);pass.dispatchWorkgroups(this.subgroup?Math.min(65535,this.columns):Math.ceil(this.columns/128),this.subgroup?Math.ceil(this.columns/65535):1);
    }
  }
  /** Correct rate references from actual neuron traces. This never writes into
   * the observed cells or into the anatomical renderer. After this operation,
   * state.y contains the normalized measured-image gradient for upstream control.
   * Ordinary encode() starts by rebuilding its extrapolation state from state.x. */
  encodeObservedCorrection(pass:GPUComputePassEncoder){
    pass.setPipeline(this.projectObserved);pass.setBindGroup(0,this.rowGroup,[0]);pass.dispatchWorkgroups(this.subgroup?Math.min(65535,this.rows):Math.ceil(this.rows/128),this.subgroup?Math.ceil(this.rows/65535):1);
    if(this.spatialWeight){pass.setPipeline(this.spatialGradient);pass.setBindGroup(0,this.rowGroup,[0]);pass.dispatchWorkgroups(Math.ceil(this.rows/128));}
    pass.setPipeline(this.correctObserved);pass.setBindGroup(0,this.columnGroup,[0]);pass.dispatchWorkgroups(this.subgroup?Math.min(65535,this.columns):Math.ceil(this.columns/128),this.subgroup?Math.ceil(this.columns/65535):1);
  }
  reset(){this.device.queue.writeBuffer(this.states,0,this.initial);}
  destroy(){this.buffers.forEach(b=>b.destroy());}
}
