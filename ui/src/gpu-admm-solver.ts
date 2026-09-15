import type {SparseTriplets} from './gpu-solver';

/** Box-constrained current solve. The precomputed matrix contains only H B;
 * it never contains a video or any measured/synthetic neural response. */
const code=`
struct Params {rows:u32,columns:u32,rho:f32,c:f32,clip:f32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> pointer:array<u32>;
@group(0) @binding(2) var<storage,read> entries:array<vec2<u32>>;
// z, scaled dual u, upper bound, unused. z alone drives upstream stimulation.
@group(0) @binding(3) var<storage,read_write> variables:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> residual:array<vec2<f32>>;
@group(0) @binding(5) var<storage,read_write> rhs:array<f32>;
@group(0) @binding(6) var<storage,read_write> solved:array<f32>;
@group(0) @binding(7) var<storage,read> inverse:array<f32>;
@compute @workgroup_size(128)
fn rightHandSide(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.rows){return;}var sum=0.0;
 for(var e=pointer[row];e<pointer[row+1u];e++){let item=entries[e];let v=variables[item.x];sum+=bitcast<f32>(item.y)*(v.x-v.y)*(p.rho/p.c);}
 rhs[row]=residual[row].x-sum;
}
var<workgroup> partial:array<f32,128>;
@compute @workgroup_size(128)
fn denseSolve(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let row=g.x;var sum=0.0;
 for(var i=lane;i<p.rows;i+=128u){sum+=inverse[row*p.rows+i]*rhs[i];}
 partial[lane]=sum;workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){partial[lane]+=partial[lane+stride];}workgroupBarrier();}
 if(lane==0u){solved[row]=partial[0];}
}
@compute @workgroup_size(128)
fn projectBounds(@builtin(global_invocation_id) g:vec3<u32>){
 let column=g.x;if(column>=p.columns){return;}var sum=0.0;
 for(var e=pointer[column];e<pointer[column+1u];e++){let item=entries[e];sum+=bitcast<f32>(item.y)*solved[item.x];}
 let old=variables[column];let q=(old.x-old.y)*(p.rho/p.c)+sum;
 let z=clamp(q+old.y,0.0,old.z);variables[column]=vec4(z,old.y+q-z,old.z,0.0);
}
@compute @workgroup_size(128)
fn measureResidual(@builtin(global_invocation_id) g:vec3<u32>){
 let row=g.x;if(row>=p.rows){return;}var sum=0.0;
 for(var e=pointer[row];e<pointer[row+1u];e++){let item=entries[e];sum+=bitcast<f32>(item.y)*variables[item.x].x;}
 residual[row].y=clamp(sum-residual[row].x,-p.clip,p.clip);
}
`;

export class GpuAdmmSolver {
 readonly states:GPUBuffer;
 readonly residual:GPUBuffer;
 readonly columns:number;
 readonly rows:number;
 private buffers:GPUBuffer[]=[];
 private initial:Float32Array<ArrayBuffer>;
 private pipelines:GPUComputePipeline[];
 private groups:GPUBindGroup[];
 constructor(private device:GPUDevice,m:SparseTriplets,upper:ArrayLike<number>,inverse:Float32Array,options:{rho:number;ridge:number;gradientLimit?:number}){
  this.columns=m.columns;this.rows=m.rows;
  if(inverse.length!==m.rows*m.rows)throw Error('Control inverse dimensions differ.');
  const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
   const b=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});
   if(typeof data!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();}this.buffers.push(b);return b;
  };
  const compressed=(major:ArrayLike<number>,minor:ArrayLike<number>,n:number)=>{
   const pointer=new Uint32Array(n+1);for(const index of Array.from(major))pointer[index+1]++;for(let i=0;i<n;i++)pointer[i+1]+=pointer[i];
   const cursor=pointer.slice(),packed=new Uint32Array(major.length*2),f=new Float32Array(packed.buffer);
   for(let i=0;i<major.length;i++){const j=cursor[major[i]]++;packed[2*j]=minor[i];f[2*j+1]=m.weight[i];}
   return [create(pointer),create(packed)];
  };
  const [rp,re]=compressed(m.row,m.column,m.rows),[cp,ce]=compressed(m.column,m.row,m.columns);
  this.initial=new Float32Array(m.columns*4);for(let i=0;i<m.columns;i++)this.initial[4*i+2]=upper[i];
  this.states=create(this.initial,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);this.residual=create(m.rows*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const parameters=new Uint32Array(8);parameters[0]=m.rows;parameters[1]=m.columns;new Float32Array(parameters.buffer).set([options.rho,options.rho+options.ridge,options.gradientLimit??5],2);
  const params=create(parameters,GPUBufferUsage.UNIFORM),rhs=create(m.rows*4),solution=create(m.rows*4),matrix=create(inverse);
  const module=device.createShaderModule({code});
  const specs:Array<[string,Array<[number,GPUBuffer]>]>=[
   ['rightHandSide',[[0,params],[1,rp],[2,re],[3,this.states],[4,this.residual],[5,rhs]]],
   ['denseSolve',[[0,params],[5,rhs],[6,solution],[7,matrix]]],
   ['projectBounds',[[0,params],[1,cp],[2,ce],[3,this.states],[6,solution]]],
   ['measureResidual',[[0,params],[1,rp],[2,re],[3,this.states],[4,this.residual]]],
  ];
  this.pipelines=specs.map(([entryPoint])=>device.createComputePipeline({layout:'auto',compute:{module,entryPoint}}));
  this.groups=specs.map(([,entries],i)=>device.createBindGroup({layout:this.pipelines[i].getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))}));
 }
 encode(pass:GPUComputePassEncoder,iterations:number){
  const run=(i:number,count:number)=>{pass.setPipeline(this.pipelines[i]);pass.setBindGroup(0,this.groups[i]);pass.dispatchWorkgroups(count);};
  for(let i=0;i<iterations;i++){run(0,Math.ceil(this.rows/128));run(1,this.rows);run(2,Math.ceil(this.columns/128));}
  run(3,Math.ceil(this.rows/128));
 }
 reset(){this.device.queue.writeBuffer(this.states,0,this.initial);}
 destroy(){this.buffers.forEach(b=>b.destroy());}
}
