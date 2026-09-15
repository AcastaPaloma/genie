/// <reference types="@webgpu/types" />

/** Full-connectome LIF on WebGPU. Each spike dispatches its real outgoing CSR
 * row; no edges are sampled and no rate surrogate replaces threshold events.
 * Synaptic inputs accumulate integer contact counts before conversion to mV.
 */
const shader=`
struct Params {tick:u32,n:u32,enabled:u32,steps:u32}
struct Cell {voltage:f32,current:f32,trace:f32,refractory:u32,rng:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> offsets:array<u32>;
@group(0) @binding(2) var<storage,read> targets:array<u32>;
@group(0) @binding(3) var<storage,read> weights:array<i32>;
@group(0) @binding(4) var<storage,read_write> queue:array<u32>;
@group(0) @binding(5) var<storage,read_write> incoming:array<atomic<i32>>;
@group(0) @binding(6) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(7) var<storage,read> inputs:array<vec2<f32>>;
@group(0) @binding(9) var<storage,read_write> stats:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read> degrees:array<u32>;
@group(0) @binding(11) var<storage,read_write> queueSizes:array<atomic<u32>>;
@group(0) @binding(12) var<storage,read_write> prefix:array<u32>;
@group(0) @binding(13) var<storage,read_write> dispatch:array<u32>;
@group(0) @binding(15) var<storage,read_write> spikeCounts:array<u32>;
fn addStat(index:u32,value:u32){let previous=atomicAdd(&stats[index*2u],value);if(previous>0xffffffffu-value){atomicAdd(&stats[index*2u+1u],1u);}}
fn membraneStep(voltage:f32,current:f32,drive:vec2<f32>)->f32{
 if(drive.y>3.5){
  // Experimental external shunting conductance, reversal at the resting
  // potential. Integrate the passive membrane and exponentially decaying
  // synaptic current analytically; never overwrite a voltage or spike target.
  // The fixed depolarizing bias remains below the isolated-cell threshold.
  let leak=1.0+max(0.0,drive.x);let membraneDecay=exp(-.01*leak);
  var coupling=(.9607894391523232-membraneDecay)/(leak-4.0);
  if(abs(leak-4.0)<.0001){coupling=.01*.9607894391523232;}
  // Mode 5 uses the externally configured depolarizing reversal,
  // still strictly below threshold. This prepares a closed gate without
  // the long membrane-recharging delay of the rest-reversal mode 4.
  let equilibrium=select(6.8/leak,6.8,drive.y>4.5);
  return -52.0+(voltage+52.0)*membraneDecay+current*coupling+equilibrium*(1.0-membraneDecay);
 }
 return -52.0+(voltage+52.0)*0.9900498337491681+current*0.009753464865615247+select(0.0,drive.x*0.009950166250831893,drive.y>2.5);
}

// No spike produced in this window arrives until the window has ended:
// window <= 9 ticks = the measured model's 1.8 ms synaptic delay.
@compute @workgroup_size(1)
fn prepare(){
 prefix[0]=0u;for(var j=0u;j<p.steps;j++){prefix[j+1u]=prefix[j]+atomicExchange(&queueSizes[(p.tick+j)%10u],0u);}
 let groups=(prefix[p.steps]+3u)/4u;dispatch[0]=min(65535u,groups);dispatch[1]=(groups+65534u)/65535u;dispatch[2]=1u;addStat(0u,prefix[p.steps]);
}
@compute @workgroup_size(128)
fn propagate(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 if(p.enabled==0u){return;}let index=(group.x+group.y*65535u)*4u+lane/32u;if(index>=prefix[p.steps]){return;}
 var step=0u;for(var j=1u;j<p.steps;j++){if(index>=prefix[j]){step=j;}}
 let pre=queue[((p.tick+step)%10u)*p.n+index-prefix[step]];
 for(var e=offsets[pre]+lane%32u;e<offsets[pre+1u];e+=32u){let w=weights[e];let post=targets[e];
  // An ideal voltage clamp cancels every incoming current on this cell.
  // Skipping that canceled addition is algebraically identical; the separate
  // delivery counter still counts the measured synaptic event.
  if(w!=0 && (inputs[post].y<1.5 || inputs[post].y>2.5)){atomicAdd(&incoming[step*p.n+post],w);}}

}
var<workgroup> deliveredPartial:array<u32,128>;
@compute @workgroup_size(128)
fn countDeliveries(@builtin(global_invocation_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 var sum=0u;
 if(p.enabled!=0u){for(var index=g.x;index<prefix[p.steps];index+=32768u){
  var step=0u;for(var j=1u;j<p.steps;j++){if(index>=prefix[j]){step=j;}}
  let pre=queue[((p.tick+step)%10u)*p.n+index-prefix[step]];sum+=degrees[pre];
 }}
 deliveredPartial[lane]=sum;workgroupBarrier();
 if(lane==0u){var total=0u;for(var i=0u;i<128u;i++){total+=deliveredPartial[i];}if(total>0u){addStat(1u,total);}}
}
@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) g:vec3<u32>){
 let id=g.x;if(id>=p.n){return;}var c=cells[id];let drive=inputs[id];
 for(var j=0u;j<p.steps;j++){
  let tick=p.tick+j;
  if(drive.y>1.5 && drive.y<2.5){let phase=c.rng;let increment=u32(min(4294967040.0,drive.x*858993.4592));c.rng+=increment;c.voltage=-52.0+select(0.0,68.75,c.rng<phase);c.current=0.0;c.refractory=0u;}
  else if(drive.y>0.5 && drive.y<1.5){var r=c.rng;r=r^(r<<13u);r=r^(r>>17u);r=r^(r<<5u);c.rng=r;if(f32(r)*(1.0/4294967296.0)<drive.x*.0002){c.voltage+=68.75;c.refractory=0u;}}
  c.current+=f32(atomicExchange(&incoming[j*p.n+id],0))*0.275;if(drive.y>1.5 && drive.y<2.5){c.current=0.0;}
  if(tick>=c.refractory){
   c.voltage=membraneStep(c.voltage,c.current,drive);c.current*=0.9607894391523232;
   if(c.voltage > -45.0){c.voltage=-52.0;c.current=0.0;c.refractory=tick+select(11u,0u,drive.y>0.5 && drive.y<2.5);c.trace+=1.0;spikeCounts[id]+=1u;
    let slot=(tick+9u)%10u;let index=atomicAdd(&queueSizes[slot],1u);queue[slot*p.n+index]=id;
   }
  }
  c.trace*=0.9980019986673331;
 }
 cells[id]=c;
}
`;

const subgroupIntegration=`
@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) g:vec3<u32>,@builtin(subgroup_invocation_id) lane:u32){
 let id=g.x;var c:Cell;if(id<p.n){c=cells[id];}let drive=inputs[id];
 for(var j=0u;j<p.steps;j++){
  let tick=p.tick+j;var fired=false;
  if(id<p.n){
   if(drive.y>1.5 && drive.y<2.5){let phase=c.rng;let increment=u32(min(4294967040.0,drive.x*858993.4592));c.rng+=increment;c.voltage=-52.0+select(0.0,68.75,c.rng<phase);c.current=0.0;c.refractory=0u;}
   else if(drive.y>0.5 && drive.y<1.5){var r=c.rng;r=r^(r<<13u);r=r^(r>>17u);r=r^(r<<5u);c.rng=r;if(f32(r)*(1.0/4294967296.0)<drive.x*.0002){c.voltage+=68.75;c.refractory=0u;}}
   c.current+=f32(atomicExchange(&incoming[j*p.n+id],0))*.275;if(drive.y>1.5 && drive.y<2.5){c.current=0.0;}
   if(tick>=c.refractory){c.voltage=membraneStep(c.voltage,c.current,drive);c.current*=0.9607894391523232;
    if(c.voltage > -45.0){c.voltage=-52.0;c.current=0.0;c.refractory=tick+select(11u,0u,drive.y>0.5 && drive.y<2.5);c.trace+=1.0;spikeCounts[id]+=1u;fired=true;}
   }
   c.trace*=0.9980019986673331;
  }
  let ballot=subgroupBallot(fired);var count=0u;var before=0u;
  for(var word=0u;word<4u;word++){count+=countOneBits(ballot[word]);if(word<lane/32u){before+=countOneBits(ballot[word]);}}
  before+=countOneBits(ballot[lane/32u]&((1u<<(lane%32u))-1u));
  var first=0u;let slot=(tick+9u)%10u;
  if(subgroupElect()&&count>0u){first=atomicAdd(&queueSizes[slot],count);}
  let start=subgroupBroadcastFirst(first);if(fired){queue[slot*p.n+start+before]=id;}
 }
 if(id<p.n){cells[id]=c;}
}
`;

export class GpuCircuit {
  readonly n:number;
  readonly traceMs:number;
  readonly gateBias:number;
  readonly observation:string;
  readonly edgeCount:number;
  readonly offsets:Uint32Array;
  readonly targets:Uint32Array;
  readonly weights:Int16Array;
  readonly device:GPUDevice;
  readonly cells:GPUBuffer;
  readonly inputs:GPUBuffer;
  private integrate:GPUComputePipeline;
  private propagate:GPUComputePipeline;
  private integrateGroup:GPUBindGroup;
  private propagateGroup:GPUBindGroup;
  private uniforms:GPUBuffer;
  private dispatchArgs:GPUBuffer;
  private prefix:GPUBuffer;
  private prepare:GPUComputePipeline;
  private countDeliveries:GPUComputePipeline;
  private countGroup:GPUBindGroup;
  private prepareGroup:GPUBindGroup;
  private stats:GPUBuffer;
  private incoming:GPUBuffer;
  private queue:GPUBuffer;
  private queueSizes:GPUBuffer;
  private history:GPUBuffer;
  private spikeCounts:GPUBuffer;
  private readback:GPUBuffer;
  private buffers:GPUBuffer[]=[];
  private tick=0;

  private gpuError:Error|null=null;
  synapsesEnabled=true;
  private constructor(device:GPUDevice,raw:ArrayBuffer,traceMs=100,observation='exponential',skipCanceledInputs=true,gateBias=6.8){
    this.gateBias=gateBias;
    if(!(traceMs>=1&&traceMs<=1000))throw Error('Invalid observation trace duration.');this.traceMs=traceMs;this.observation=observation;
    this.device=device;
    device.addEventListener('uncapturederror',event=>{this.gpuError??=Error(event.error.message);});
    void device.lost.then(info=>{if(info.reason!=='destroyed')this.gpuError=Error('GPU device lost: '+info.message);});
    const [version,n,e]=new Uint32Array(raw,0,3);if(version!==783)throw Error('Unknown graph format.');
    this.n=n;this.edgeCount=e;
    const pointers=new Uint32Array(raw,12,n+1),to=new Uint32Array(raw,12+(n+1)*4,e),weights=new Int16Array(raw,12+(n+1)*4+e*4,e);
    this.offsets=pointers;this.targets=to;this.weights=weights;
    const create=(data:ArrayBufferView|number,usage=GPUBufferUsage.STORAGE)=>{
      const b=device.createBuffer({size:Math.max(4,typeof data==='number'?data:data.byteLength),usage:usage|GPUBufferUsage.COPY_DST,mappedAtCreation:typeof data!=='number'});
      if(typeof data!=='number'){new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();}
      this.buffers.push(b);return b;
    };
    const offsetBuffer=create(pointers),targetBuffer=create(to),weightBuffer=create(Int32Array.from(weights));
    const degrees=new Uint32Array(n);for(let i=0;i<n;i++)for(let j=pointers[i];j<pointers[i+1];j++)if(weights[j])degrees[i]++;const degreeBuffer=create(degrees);

    this.history=create(n*8);this.spikeCounts=create(n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    this.cells=create(n*20,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    this.inputs=create(n*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);this.queue=create(n*10*4);this.incoming=create(n*9*4);
    this.queueSizes=create(40,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    this.dispatchArgs=create(12,GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT);this.prefix=create(40);
    this.stats=create(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    this.uniforms=create(256*5000,GPUBufferUsage.UNIFORM);
    this.readback=create(n*24+16+40,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    let code=device.features.has('subgroups')?'enable subgroups;\n'+shader.slice(0,shader.indexOf('@compute @workgroup_size(128)\nfn integrate'))+subgroupIntegration:shader;
    if(!skipCanceledInputs)code=code.replace('w!=0 && (inputs[post].y<1.5 || inputs[post].y>2.5)','w!=0');
    if(observation==='interval'){
      code=code.replace('struct Cell', '@group(0) @binding(14) var<storage,read_write> spikeHistory:array<vec2<u32>>;\nstruct Cell');
      code=code.replaceAll('c.trace+=1.0;spikeCounts[id]+=1u;',`c.trace+=1.0;spikeCounts[id]+=1u;
        let prior=spikeHistory[id];let stamp=tick+1u;
        let period=select(prior.y,stamp-prior.x,prior.x>0u);
        spikeHistory[id]=vec2(stamp,period);`);
      code=code.replaceAll('c.trace*=0.9980019986673331;',`let history=spikeHistory[id];
        let age=f32(tick+1u-history.x)*.2;let interval=f32(history.y)*.2;
        let rate=select(0.0,1000.0/max(.2,interval)*exp(-max(0.0,age-interval)/15.0),history.y>0u);
        c.trace=rate*${traceMs/1000};`);
    }
    // Only the external conductance bias/reversal changes. Threshold, reset,
    // membrane integration and all measured incoming currents remain intact.
    code=code.replace('select(6.8/leak,6.8,drive.y>4.5)',`select(${gateBias}/leak,${gateBias},drive.y>4.5)`);
    const module=device.createShaderModule({code:code.replaceAll('0.9980019986673331',String(Math.exp(-.2/traceMs)))});
    const entry=(binding:number,buffer:GPUBuffer,size?:number)=>({binding,resource:{buffer,...(size?{size}:{})}});
    const uniformLayout={binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform' as const,hasDynamicOffset:true,minBindingSize:16}};
    const layout=(types:Array<[number,GPUBufferBindingType]>)=>device.createBindGroupLayout({entries:[uniformLayout,...types.map(([binding,type])=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}}))]});
    const propagationLayout=layout([[1,'read-only-storage'],[2,'read-only-storage'],[3,'read-only-storage'],[4,'storage'],[5,'storage'],[7,'read-only-storage'],[12,'storage']]);
    const preparationLayout=layout([[9,'storage'],[11,'storage'],[12,'storage'],[13,'storage']]);
    const countLayout=layout([[4,'storage'],[9,'storage'],[10,'read-only-storage'],[12,'storage']]);
    const integrationTypes:Array<[number,GPUBufferBindingType]>=[[4,'storage'],[5,'storage'],[6,'storage'],[7,'read-only-storage'],[9,'storage'],[11,'storage'],[15,'storage']];if(observation==='interval')integrationTypes.push([14,'storage']);const integrationLayout=layout(integrationTypes);
    this.prepare=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[preparationLayout]}),compute:{module,entryPoint:'prepare'}});
    this.prepareGroup=device.createBindGroup({layout:preparationLayout,entries:[entry(0,this.uniforms,16),entry(9,this.stats),entry(11,this.queueSizes),entry(12,this.prefix),entry(13,this.dispatchArgs)]});
    this.countDeliveries=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[countLayout]}),compute:{module,entryPoint:'countDeliveries'}});
    this.countGroup=device.createBindGroup({layout:countLayout,entries:[entry(0,this.uniforms,16),entry(4,this.queue),entry(9,this.stats),entry(10,degreeBuffer),entry(12,this.prefix)]});
    this.propagate=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[propagationLayout]}),compute:{module,entryPoint:'propagate'}});
    this.integrate=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[integrationLayout]}),compute:{module,entryPoint:'integrate'}});
    this.propagateGroup=device.createBindGroup({layout:propagationLayout,entries:[entry(0,this.uniforms,16),entry(1,offsetBuffer),entry(2,targetBuffer),entry(3,weightBuffer),entry(4,this.queue),entry(5,this.incoming),entry(7,this.inputs),entry(12,this.prefix)]});
    this.integrateGroup=device.createBindGroup({layout:integrationLayout,entries:[entry(0,this.uniforms,16),entry(4,this.queue),entry(5,this.incoming),entry(6,this.cells),entry(7,this.inputs),entry(9,this.stats),entry(11,this.queueSizes),entry(15,this.spikeCounts),...(observation==='interval'?[entry(14,this.history)]:[])]});
    this.reset();
  }
  static async create(raw:ArrayBuffer,options:{traceMs?:number;largeBufferBytes?:number;observation?:string;skipCanceledInputs?:boolean;gateBias?:number}={}){
    const gateBias=options.gateBias??6.8;
    if(!Number.isFinite(gateBias)||gateBias<0||gateBias>6.99)throw Error('External gate bias must remain at least 0.01 mV below the isolated threshold.');
    if(!navigator.gpu)throw Error('This browser does not expose WebGPU.');
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter.');
    const required=options.largeBufferBytes??0;if(required>adapter.limits.maxStorageBufferBindingSize||required>adapter.limits.maxBufferSize)throw Error('This GPU cannot hold the requested control inverse.');
    const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],...(required?{requiredLimits:{maxStorageBufferBindingSize:Math.max(134217728,required),maxBufferSize:Math.max(268435456,required)}}:{})});
    device.addEventListener('uncapturederror',event=>console.error(event.error.message));
    return new GpuCircuit(device,raw,options.traceMs,options.observation,options.skipCanceledInputs,options.gateBias);
  }
  get timeMs(){return this.tick*.2;}
  reset(){
    this.tick=0;this.device.queue.writeBuffer(this.spikeCounts,0,new Uint32Array(this.n));this.device.queue.writeBuffer(this.history,0,new Uint32Array(this.n*2));
    const data=new ArrayBuffer(this.n*20),f=new Float32Array(data),u=new Uint32Array(data);
    for(let i=0;i<this.n;i++){f[i*5]=-52;u[i*5+4]=((i+1)*2654435761)>>>0;}
    this.device.queue.writeBuffer(this.cells,0,data);this.device.queue.writeBuffer(this.incoming,0,new Uint32Array(this.n*9));
    this.device.queue.writeBuffer(this.queueSizes,0,new Uint32Array(10));this.device.queue.writeBuffer(this.stats,0,new Uint32Array(4));
  }
  setInputs(ids:ArrayLike<number>,rates:ArrayLike<number>){
    const inputs=new Float32Array(this.n*2);for(let j=0;j<ids.length;j++){inputs[ids[j]*2]=rates[j];inputs[ids[j]*2+1]=1;}
    this.device.queue.writeBuffer(this.inputs,0,inputs);
  }
  /** External tonic drive in equivalent membrane-potential units. This mode
   * preserves incoming synaptic currents, thresholding and refractoriness. */
  setTonicInputs(ids:ArrayLike<number>,amplitudes:ArrayLike<number>){
    if(ids.length!==amplitudes.length)throw Error('Tonic input lengths differ.');
    const inputs=new Float32Array(this.n*2);
    for(let j=0;j<ids.length;j++){if(ids[j]<0||ids[j]>=this.n||!Number.isFinite(amplitudes[j]))throw Error('Invalid tonic input.');inputs[ids[j]*2]=amplitudes[j];inputs[ids[j]*2+1]=3;}
    this.device.queue.writeBuffer(this.inputs,0,inputs);
  }
  async advance(ms:number,read=true,beforeStep?:(pass:GPUComputePassEncoder,tick:number)=>void,maxBatchTicks=9){
    if(this.gpuError)throw this.gpuError;
    if(!Number.isInteger(maxBatchTicks)||maxBatchTicks<1||maxBatchTicks>9)throw Error('Integration batches must preserve the 9-tick synaptic delay.');
    const steps=Math.round(ms/.2);if(steps<1||steps>5000)throw Error('GPU advance must be 0.2–1000 ms.');
    const batches:{tick:number;steps:number}[]=[];
    for(let j=0;j<steps;){const tick=this.tick+j;const length=Math.min(maxBatchTicks,steps-j,25-tick%25);batches.push({tick,steps:length});j+=length;}
    const params=new Uint32Array(64*batches.length);batches.forEach((batch,j)=>{params[j*64]=batch.tick;params[j*64+1]=this.n;params[j*64+2]=this.synapsesEnabled?1:0;params[j*64+3]=batch.steps;});
    this.device.queue.writeBuffer(this.uniforms,0,params);
    const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();
    batches.forEach((batch,j)=>{
      beforeStep?.(pass,batch.tick);
      pass.setPipeline(this.prepare);pass.setBindGroup(0,this.prepareGroup,[j*256]);pass.dispatchWorkgroups(1);
      pass.setPipeline(this.countDeliveries);pass.setBindGroup(0,this.countGroup,[j*256]);pass.dispatchWorkgroups(256);
      pass.setPipeline(this.propagate);pass.setBindGroup(0,this.propagateGroup,[j*256]);pass.dispatchWorkgroupsIndirect(this.dispatchArgs,0);
      pass.setPipeline(this.integrate);pass.setBindGroup(0,this.integrateGroup,[j*256]);pass.dispatchWorkgroups(Math.ceil(this.n/128));
    });
    pass.end();this.tick+=steps;
    if(read){encoder.copyBufferToBuffer(this.cells,0,this.readback,0,this.n*20);encoder.copyBufferToBuffer(this.stats,0,this.readback,this.n*20,16);encoder.copyBufferToBuffer(this.queueSizes,0,this.readback,this.n*20+16,40);encoder.copyBufferToBuffer(this.spikeCounts,0,this.readback,this.n*20+56,this.n*4);}
    this.device.queue.submit([encoder.finish()]);
    if(!read){await this.device.queue.onSubmittedWorkDone();return null;}
    await this.readback.mapAsync(GPUMapMode.READ);
    const copy=this.readback.getMappedRange().slice(0);this.readback.unmap();
    if(this.gpuError)throw this.gpuError;
    const floats=new Float32Array(copy,0,this.n*5),statistics=new Uint32Array(copy,this.n*20,4);
    return {cells:floats,counts:new Uint32Array(copy,this.n*20+56,this.n),spikes:statistics[0]+4294967296*statistics[1]+new Uint32Array(copy,this.n*20+16,10).reduce((a,b)=>a+b,0),deliveries:statistics[2]+4294967296*statistics[3]};
  }
  destroy(){this.buffers.forEach(b=>b.destroy());this.device.destroy();}
}
