/// <reference types="@webgpu/types" />
// This worker never receives a video frame, target image or control references.
// It projects measured neural activity through immutable, complete anatomy.
const code=`
struct Params {first:u32,rows:u32,neurons:u32,pad:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> pointer:array<u32>;
@group(0) @binding(2) var<storage,read> cable:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read> activity:array<f32>;
@group(0) @binding(4) var<storage,read_write> light:array<f32>;
@compute @workgroup_size(32)
fn project(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let row=group.x+group.y*65535u;if(row>=p.rows){return;}var sum=0.0;
 for(var e=pointer[row]+lane;e<pointer[row+1u];e+=32u){let item=cable[e];sum+=bitcast<f32>(item.y)*activity[item.x];}
 let total=subgroupAdd(sum);
 if(lane==0u){light[p.first+row]=select(total,-1.0,pointer[row]==pointer[row+1u]);}
}
`;
let device:GPUDevice,pipeline:GPUComputePipeline,activity:GPUBuffer,light:GPUBuffer,readback:GPUBuffer;
let groups:{bind:GPUBindGroup;rows:number}[]=[],buffers:GPUBuffer[]=[],busy=false,neurons=0;
let gpuError:Error|null=null;
async function checked(url:string,expected:string){
 const r=await fetch(url);if(!r.ok)throw Error('Anatomical light cache is unavailable: '+url);
 const raw=await r.arrayBuffer(),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==expected)throw Error('Anatomical light cache checksum differs: '+url);return raw;
}
self.onmessage=async event=>{
 const m=event.data;
 try{
  if(m.type==='init'){
   const r=await fetch(m.base+'render-1280.json');if(!r.ok)throw Error('Complete render-cache manifest is unavailable.');const meta=await r.json();
   const manifest=await checked(m.base+'brain.json',meta.sourceAnatomySha256),anatomy=JSON.parse(new TextDecoder().decode(manifest));
   if(!anatomy.complete||anatomy.missingRootIds.length||anatomy.neuronCount!==139255||meta.neurons!==anatomy.neuronCount||meta.vertices!==anatomy.vertexCount||meta.edges!==anatomy.edgeCount)
    throw Error('The render cache does not include the complete anatomy.');
   const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('The complete anatomical cache requires WebGPU.');
   const subgroup=adapter.features.has('subgroups')&&adapter.info.subgroupMinSize===32&&adapter.info.subgroupMaxSize===32;
   device=await adapter.requestDevice({requiredFeatures:subgroup?['subgroups']:[]});
   device.addEventListener('uncapturederror',e=>{gpuError??=Error(e.error.message);});
   void device.lost.then(info=>{if(info.reason!=='destroyed')gpuError=Error('Anatomical renderer lost its GPU: '+info.message);});
   const create=(value:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
    const b=device.createBuffer({size:Math.max(4,typeof value==='number'?value:value.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof value!=='number'});
    if(typeof value!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));b.unmap();}buffers.push(b);return b;
   };
   neurons=meta.neurons;activity=create(neurons*4);light=create(meta.resolution[0]*meta.resolution[1]*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
   readback=create(light.size,GPUBufferUsage.MAP_READ);
   const fallback=code.replace('@workgroup_size(32)','@workgroup_size(1)').replace('e+=32u','e++').replace('subgroupAdd(sum)','sum');
   const module=device.createShaderModule({code:subgroup?'enable subgroups;\n'+code:fallback});
   pipeline=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'project'}});
   let rows=0,entries=0;
   for(const chunk of meta.chunks){
    const raw=await checked(m.base+chunk.file,chunk.sha256),[version,first,count,nnz]=new Uint32Array(raw,0,4);
    if(version!==786||first!==rows||first!==chunk.start||count!==chunk.rows||nnz!==chunk.entries||raw.byteLength!==16+(count+1)*4+nnz*8)throw Error('Incomplete anatomical light block.');
    const pointer=new Uint32Array(raw,16,count+1);if(pointer[0]!==0||pointer[count]!==nnz)throw Error('Anatomical light row coverage differs.');
    const bound=[create(new Uint32Array([first,count,neurons,0]),GPUBufferUsage.UNIFORM),create(pointer),create(new Uint32Array(raw,16+(count+1)*4,nnz*2)),activity,light];
    const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bound.map((buffer,binding)=>({binding,resource:{buffer}}))});groups.push({bind,rows:count});rows+=count;entries+=nnz;
    self.postMessage({type:'progress',loaded:groups.length,total:meta.chunks.length});
   }
   if(rows!==meta.resolution[0]*meta.resolution[1]||entries!==meta.entries)throw Error('Anatomical cache coverage differs.');
   if(gpuError)throw gpuError;self.postMessage({type:'ready',resolution:meta.resolution,footprintMicrometers:meta.footprintMicrometers,center:meta.center,entries,neurons,branches:meta.edges,sourceAnatomySha256:meta.sourceAnatomySha256});
  }else if(m.type==='activity'){
   if(busy)throw Error('An anatomical frame is already in flight.');if(m.values.length!==neurons)throw Error('Every neuron must supply activity.');busy=true;
   const start=performance.now();device.queue.writeBuffer(activity,0,m.values);
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);
   for(const group of groups){pass.setBindGroup(0,group.bind);pass.dispatchWorkgroups(Math.min(65535,group.rows),Math.ceil(group.rows/65535));}pass.end();encoder.copyBufferToBuffer(light,0,readback,0,light.size);device.queue.submit([encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const values=new Float32Array(readback.getMappedRange().slice(0));readback.unmap();if(gpuError)throw gpuError;
   busy=false;self.postMessage({type:'light',values,computeMs:performance.now()-start},{transfer:[values.buffer]});
  }
 }catch(error){busy=false;self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
self.onclose=()=>{buffers.forEach(b=>b.destroy());device?.destroy();};
