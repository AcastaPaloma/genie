import {GpuCircuit} from './gpu-circuit';
import {SpikingCircuit} from './neural';
import {GpuWholeBrainController} from './gpu-whole-brain-controller';
import {GpuSparseSolver} from './gpu-solver';
import {GpuRateSupport} from './gpu-rate-support';
import {GpuSynapticGateController} from './gpu-synaptic-gate-controller';

function assert(condition:boolean,message:string){if(!condition)throw Error(message);}
function graph(n:number,weights:number[]){const e=weights.length,raw=new ArrayBuffer(12+4*(n+1)+6*e);new Uint32Array(raw,0,3).set([783,n,e]);const offsets=new Uint32Array(raw,12,n+1);for(let i=0;i<=n;i++)offsets[i]=Math.min(i,e);new Uint32Array(raw,12+4*(n+1),e).set(weights.map((_,i)=>i+1));new Int16Array(raw,12+4*(n+1)+4*e,e).set(weights);return raw;}
try{
 const checks=[];
 for(const maxBatchTicks of [1,3,9])for(const traceMs of [40,100])for(const weight of [800,-800]){
  const raw=graph(3,[weight,weight]),cpu=new SpikingCircuit(raw,traceMs),gpu=await GpuCircuit.create(raw,{traceMs}),ids=new Int32Array([0]),rates=new Float32Array([5000]);gpu.setInputs(ids,rates);
  for(let t=0;t<500;t++)cpu.step(ids,rates);const state=(await gpu.advance(100,true,undefined,maxBatchTicks))!;
  assert(state.counts.reduce((a,b)=>a+b,0)===state.spikes,'Per-neuron counts do not conserve emitted events');
  assert(state.spikes===cpu.spikes,`CPU/GPU spike count differs for weight ${weight}`);assert(state.deliveries===cpu.deliveries,'Delivery count differs');
  for(let id=0;id<3;id++)assert(Math.abs(state.cells[id*5+2]*1000/traceMs-cpu.rateHz(id))<.03,'Spike-trace filtering differs');
  gpu.reset();gpu.setInputs(ids,rates);const replay=(await gpu.advance(100,true,undefined,maxBatchTicks))!;assert(replay.cells.every((x,i)=>x===state.cells[i]),'GPU reset is not deterministic');assert(replay.counts.every((x,i)=>x===state.counts[i]),'Count reset differs');
  gpu.synapsesEnabled=false;gpu.reset();const cut=(await gpu.advance(100,true,undefined,maxBatchTicks))!;assert(cut.cells[7]===0&&cut.cells[12]===0,'Cut synapses still drive downstream traces');
  checks.push({maxBatchTicks,traceMs,weight,spikes:state.spikes,deliveries:state.deliveries,traceToleranceHz:.03});gpu.destroy();
 }
 // Tonic current preserves the ordinary LIF threshold and refractory interval.
 const tonic=await GpuCircuit.create(graph(3,[800,800]),{traceMs:40});
 tonic.setTonicInputs([0],[8]);let propagated=(await tonic.advance(1000))!;
 assert(propagated.counts[0]===22,'8 mV tonic drive changed the analytic isolated-cell spike count');
 assert(propagated.counts[1]>0&&propagated.counts[2]>0,'Tonic input did not propagate through both synapses');
 tonic.synapsesEnabled=false;tonic.reset();const tonicCut=(await tonic.advance(1000))!;
 assert(tonicCut.counts[0]===22&&tonicCut.counts[1]===0&&tonicCut.counts[2]===0,'Cut transmission did not isolate external drive');
 tonic.destroy();
 // The optional shunt changes passive conductance, not threshold events.
 // In isolation its subthreshold bias must stay silent, including at the
 // removable singularity where the membrane and synaptic time constants match.
 for(const gateBias of [6.8,6.95,6.99])for(const mode of [4,5])for(const conductance of [0,3,256,16384,65536,327680]){
  const shunt=await GpuCircuit.create(graph(1,[]),{traceMs:40,gateBias});
  shunt.device.queue.writeBuffer(shunt.inputs,0,new Float32Array([conductance,mode]));
  const state=(await shunt.advance(100))!,leak=1+conductance;
  const expected=-52+gateBias/(mode===5?1:leak)*(1-Math.exp(-leak*100/20));
  assert(state.spikes===0,'Subthreshold shunt input fired without any synapses');
  assert(Math.abs(state.cells[0]-expected)<.002,'Shunt passive membrane differs from the independent analytic solution');shunt.destroy();
 }
 const gated=await GpuCircuit.create(graph(3,[800,800]),{traceMs:40});
 for(const closed of [true,false]){
  gated.reset();gated.device.queue.writeBuffer(gated.inputs,0,new Float32Array([50,3,closed?256:0,4,0,0]));
  const state=(await gated.advance(1000))!;
  assert(state.counts[0]>0,'Shunt experiment lost the seed spikes');
  assert(closed?state.counts[1]===0:state.counts[1]>0&&state.counts[2]>0,'Shunt did not reversibly gate measured incoming events');
 }
 gated.reset();gated.synapsesEnabled=false;
 const gatedCut=(await gated.advance(1000))!;
 assert(gatedCut.counts[1]===0&&gatedCut.counts[2]===0,'Open subthreshold gate spiked after transmission was cut');gated.destroy();
 // Independent identity-operator check for the observed-image gradient. Its
 // output is a reference update; the neural state must remain bit-for-bit intact.
 const measured=await GpuCircuit.create(graph(3,[800,800]),{traceMs:40});measured.setTonicInputs([0],[8]);
 const beforeCorrection=(await measured.advance(1000))!,target=new Float32Array([1,.5,0]);
 for(const observationAnchor of [0,.5,1])for(const spatialWeight of [0,1])for(const gap of spatialWeight?[false,true]:[false]){
 const inverse=new GpuSparseSolver(measured.device,{rows:3,columns:3,imageWidth:3,row:gap?[0,2]:[0,1,2],column:gap?[0,2]:[0,1,2],weight:gap?[1,1]:[1,1,1]},[1,1,1],{observations:measured.cells,observationScale:.25,observationStep:.5,observationAnchor,spatialWeight});inverse.setTarget(target);
 const encoding=measured.device.createCommandEncoder(),correctionPass=encoding.beginComputePass();inverse.encodeObservedCorrection(correctionPass);correctionPass.end();
 const correctionReadback=measured.device.createBuffer({size:108,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 encoding.copyBufferToBuffer(inverse.states,0,correctionReadback,0,48);encoding.copyBufferToBuffer(measured.cells,0,correctionReadback,48,60);measured.device.queue.submit([encoding.finish()]);await correctionReadback.mapAsync(GPUMapMode.READ);
 const corrected=correctionReadback.getMappedRange().slice(0),correctedValues=new Float32Array(corrected),correctedBits=new Uint32Array(corrected),priorBits=new Uint32Array(beforeCorrection.cells.buffer,beforeCorrection.cells.byteOffset,15);
 const observations=[0,1,2].map(id=>Math.min(1,Math.max(0,beforeCorrection.cells[id*5+2]*.25))),error=observations.map((value,id)=>value-target[id]);
 // Independent 3-pixel Laplacian [[1,-1,0],[-1,2,-1],[0,-1,1]].
 const edgeError=[error[0]-error[1],2*error[1]-error[0]-error[2],error[2]-error[1]];
 for(let id=0;id<3;id++){
  const gradient=gap&&id===1?0:error[id]+(gap?0:spatialWeight*edgeError[id]),majorizer=1+(gap?0:spatialWeight*(id===1?4:2));
  const expected=Math.max(0,Math.min(1,observationAnchor*observations[id]-.5*gradient/majorizer));
  assert(Math.abs(correctedValues[id*4]-expected)<1e-5,'Observed image/spatial gradient differs from the independent matrix result');
  assert(Math.abs(correctedValues[id*4+1]-gradient/majorizer)<1e-5,'Upstream image gradient differs from the independent matrix result');
 }
 assert(priorBits.every((value,index)=>value===correctedBits[index+12]),'Reference correction mutated measured neural state');correctionReadback.unmap();correctionReadback.destroy();inverse.destroy();
 }measured.destroy();
 // Two presynaptic cells converge on one missing target: +10 and -10 contacts.
 // A half-size support step must raise excitation by .25 and lower inhibition
 // by .25. Protected seeds/motors must remain unchanged, as must neural state.
 const supportRaw=graph(3,[10,-10]);new Uint32Array(supportRaw,28,2).set([2,2]);
 const supportCircuit=await GpuCircuit.create(supportRaw,{traceMs:40}),supportBefore=(await supportCircuit.advance(.2))!;
 const requestBuffer=supportCircuit.device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 for(const protectedInputs of [false,true])for(const needsLight of [false,true])for(const targetImportance of [1,4])for(const inputRole of [1,3]){
  supportCircuit.device.queue.writeBuffer(requestBuffer,0,new Float32Array([.2,0,1,1,.8,0,1,1,1,needsLight?-1:1,1,1]));
  const support=new GpuRateSupport(supportCircuit,requestBuffer,new Uint32Array(protectedInputs?[2,0,1]:[inputRole,inputRole,1]),[1,1,targetImportance],60,.5);
  const encoder=supportCircuit.device.createCommandEncoder(),pass=encoder.beginComputePass();support.encode(pass,3);pass.end();
  const read=supportCircuit.device.createBuffer({size:108,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  encoder.copyBufferToBuffer(requestBuffer,0,read,0,48);encoder.copyBufferToBuffer(supportCircuit.cells,0,read,48,60);supportCircuit.device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
  const mapped=read.getMappedRange(),values=new Float32Array(mapped),bits=new Uint32Array(mapped),prior=new Uint32Array(supportBefore.cells.buffer,supportBefore.cells.byteOffset,15);
  const inactive=protectedInputs||!needsLight;
  const change=inactive?0:targetImportance===1?.25:.5;
  assert(Math.abs(values[0]-(.2+change))<1e-6&&Math.abs(values[4]-(.8-change))<1e-6&&values[8]===1,'Signed rate-support differs from the independent three-cell calculation');
  assert(prior.every((value,i)=>value===bits[i+12]),'Upstream reference support changed actual neural state');read.unmap();read.destroy();support.destroy();
 }requestBuffer.destroy();supportCircuit.destroy();
 // A bounded tonic stimulus enters only the explicitly selected afferent.
 // The gate and motor cells can fire only after their incoming chain fires.
 const visualCircuit=await GpuCircuit.create(graph(3,[800,800]),{traceMs:40});
 const visualController=new GpuSynapticGateController(visualCircuit,{rows:3,columns:3,imageWidth:3,row:[0,1,2],column:[0,1,2],weight:[1,1,1]},[2],{seedCount:0,visualInputIds:[0],gateMode:4,shunt:16384,rateScale:60,traceTrigger:1,feedbackTicks:1,simulationBatchTicks:3,imageFeedback:4,supportGain:.5});
 assert(visualController.visualInputIds.length===1&&visualController.visualInputIds[0]===0&&visualController.seedIds.length===0,'Direct visual inputs were classified incorrectly');
 for(const mode of ['connected','disconnected','muted']){
  visualCircuit.reset();visualController.reset();visualCircuit.synapsesEnabled=mode!=='disconnected';visualController.setMuted(mode==='muted');visualController.setFrame(new Float32Array(3).fill(.7));
  const state=(await visualCircuit.advance(100,true,(pass,tick)=>visualController.encode(pass,tick),3))!;
  if(mode==='connected')assert(state.counts.every(n=>n>0),'Afferent stimulation did not recruit the downstream chain');
  if(mode==='disconnected')assert(state.counts[0]>0&&state.counts[1]===0&&state.counts[2]===0&&state.deliveries===0,'Direct afferent activity was confused with synapse-dependent activity');
  if(mode==='muted')assert(state.spikes===0,'Muted afferent input generated activity from rest');
  const activity=Float32Array.from([0,1,2],id=>Math.min(1,state.cells[id*5+2]*1000/40/60)),share=visualController.lightAttribution(activity);
  if(mode==='disconnected')assert(share.visualInputLightFraction===1&&share.synapseDependentLightFraction===0,'Direct-input optical attribution is incorrect');
 }
 visualController.destroy();visualCircuit.destroy();
 // A single-contact input cannot bridge the original preparation margin.
 // A closer but strictly subthreshold external reversal makes those measured
 // events usable without changing threshold, contacts or the spike readout.
 const weakGateChecks=[];
 for(const gateBias of [6.8,6.99])for(const batch of [1,3,9]){
  const weak=await GpuCircuit.create(graph(3,[1,800]),{traceMs:40,gateBias});
  const control=new GpuSynapticGateController(weak,{rows:3,columns:3,imageWidth:3,row:[0,1,2],column:[0,1,2],weight:[1,1,1]},[2],{seedCount:0,visualInputIds:[0],gateMode:4,shunt:327680,rateScale:60,traceTrigger:1,feedbackTicks:1,imageFeedback:0});
  control.setFrame(new Float32Array([.7,.7,0]));
  const state=(await weak.advance(1000,true,(pass,tick)=>control.encode(pass,tick),batch))!;
  assert(state.counts[0]>0,'Weak-synapse test has no afferent events');
  assert(gateBias===6.8?state.counts[1]===0:state.counts[1]>0&&state.counts[2]>0,'Weak synaptic response differs from the preparation-margin prediction');
  weak.synapsesEnabled=false;weak.reset();control.reset();control.setFrame(new Float32Array([.7,.7,0]));
  const cut=(await weak.advance(1000,true,(pass,tick)=>control.encode(pass,tick),batch))!;
  assert(cut.counts[0]>0&&cut.counts[1]===0&&cut.counts[2]===0&&cut.deliveries===0,'Prepared weak-input cells fired without incoming events');
  weakGateChecks.push({gateBias,batch,counts:Array.from(state.counts),cutCounts:Array.from(cut.counts)});
  control.destroy();weak.destroy();
 }
 for(const gateBias of [-1,7,Infinity,NaN]){
  let rejected=false;try{const invalid=await GpuCircuit.create(graph(1,[]),{gateBias});invalid.destroy();}catch{rejected=true;}
  assert(rejected,'An invalid/suprathreshold external preparation was accepted');
 }
 // Compile and execute the new feedback paths on a tiny complete graph. This
 // checks buffer aliasing and verifies that the controller only writes inputs.
 const controlled=await GpuCircuit.create(graph(8,[800,800,800,800,800,800,800]),{traceMs:40});
 const feedback=new GpuWholeBrainController(controlled,{rows:8,columns:8,row:Uint32Array.from([0,1,2,3,4,5,6,7]),column:Uint32Array.from([0,1,2,3,4,5,6,7]),weight:new Float32Array(8).fill(1)},[]);
 feedback.setFrame(new Float32Array(8).fill(.7));controlled.synapsesEnabled=false;
 const controlledCut=(await controlled.advance(100,true,(pass,tick)=>feedback.encode(pass,tick)))!;
 const externallyDriven=new Set(feedback.inputIds);
 assert(controlledCut.counts.every((count,id)=>externallyDriven.has(id)||count===0),'Feedback wrote downstream activity directly');
 feedback.destroy();controlled.destroy();
 const clampStates:Array<{cells:Float32Array;deliveries:number}>=[];
 for(const skipCanceledInputs of [false,true]){const gpu=await GpuCircuit.create(graph(3,[800,800]),{skipCanceledInputs});gpu.device.queue.writeBuffer(gpu.inputs,0,new Float32Array([2000,2,700,2,0,0]));clampStates.push((await gpu.advance(100))!);gpu.destroy();}
 assert(clampStates[0].cells.every((v,i)=>v===clampStates[1].cells[i]),'Skipping canceled clamp currents changed the neural state');assert(clampStates[0].deliveries===clampStates[1].deliveries,'Clamp optimization changed event accounting');
 const big=await GpuCircuit.create(graph(138639,new Array(138638).fill(1)));const all=Int32Array.from({length:big.n},(_,i)=>i);big.setInputs(all,new Float32Array(big.n).fill(5000));const state=(await big.advance(4))!;assert(state.spikes===20*big.n,'Large indirect spike dispatch dropped events');assert(state.deliveries===11*(big.n-1),'Batched indirect propagation dropped connections');big.destroy();
 document.getElementById('result')!.textContent=JSON.stringify({status:'passed',checks,weakGateChecks,largeDispatchSpikes:state.spikes});document.body.dataset.done='true';
}catch(error){document.getElementById('result')!.textContent=String(error);document.body.dataset.error='true';}
