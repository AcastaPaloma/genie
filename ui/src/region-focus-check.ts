// Diagnostic: how the brain panel looks for each region and focus choice, driven
// by a Freedoom clip through the play page's controller. It never contacts the
// world model, whose connect() would reset a shared server.
import {FullBrainView,type FullBrainMetadata} from './full-brain';
import {LiveSource} from './live-source';
import {fetchRegionManifest,type RegionName} from './brain-regions';
import {applyBrainFocus,describeFocus,type FocusName} from './brain-focus';
import {fetchJson} from './types';
const out=document.getElementById('result')!;
const directory='/data/full-brain-783/',controlWidth=640;
// Same controller as play.ts.
const parameters={controller:'gate',gateMode:4,shunt:327680,rateScale:180,traceMs:40,traceTrigger:1,feedbackTicks:1,simulationBatchTicks:3,imageFeedback:4,supportGain:.5,useVisualInputs:1,seedCount:0,gateBias:6.99};
try{
  const [meta,operator,regions]=await Promise.all([fetchJson<FullBrainMetadata>(directory+'brain.json'),fetchJson<{resolution:number[];footprintMicrometers:number[];center:number[]}>(directory+`whole-arbor-${controlWidth}.json`),fetchRegionManifest(directory,139255)]);
  const brain=new FullBrainView(document.getElementById('brain')!,meta);
  brain.setPalette('source');brain.setExposure(1.5);brain.setReferenceProjection(operator.footprintMicrometers,operator.center);brain.setNeuralActivity(new Float32Array(meta.neuronCount));
  const worker=new Worker(new URL('./full-neural.worker.ts',import.meta.url),{type:'module'});
  let next:{resolve:(m:any)=>void;reject:(e:Error)=>void}|null=null;
  worker.onmessage=e=>{if(e.data.type==='error'){next?.reject(Error(e.data.message));return;}if(e.data.type==='ready'||e.data.type==='state'){const n=next;next=null;n?.resolve(e.data);}};
  const send=(message:object)=>new Promise<any>((resolve,reject)=>{next={resolve,reject};worker.postMessage(message);});
  out.textContent='Loading cache and neural model…';
  await Promise.all([brain.loadLightCache(directory),send({type:'init',base:directory,controlWidth,parameters})]);
  worker.postMessage({type:'config',synapses:true,muted:false});
  const source=new LiveSource(document.createElement('canvas'),operator.resolution[0],operator.resolution[1]);
  if(!await source.load('/data/live-clips/map01.mp4'))throw Error('Clip did not load.');
  let values:Float32Array|null=null;
  const steps=Number(new URL(location.href).searchParams.get('steps')??30);
  for(let i=0;i<steps;i++){
    const t=2+i*.1;
    if(t!==source.video.currentTime)await new Promise<void>(resolve=>{source.video.onseeked=()=>resolve();source.video.currentTime=t;});
    const state=await send({type:'step',epoch:0,values:source.capture(false),capturedAt:performance.now(),decodedAt:performance.now(),sourceTime:t,frameId:i});
    values=Float32Array.from(state.values);out.textContent=`Warming the simulation: ${i+1}/${steps}`;
  }
  const show=async()=>{if(source.lastCapture)brain.setSourceFrame(source.lastCapture);await brain.prepareNeuralFrame(new Float32Array(values!));brain.render(performance.now(),0,false);await brain.waitForDraw();};
  (window as any).showDefault=async()=>{await show();return 'play page default (no region call)';};
  (window as any).showVariant=async(region:RegionName,focus:FocusName)=>{applyBrainFocus(brain,regions,region,focus);await show();return describeFocus(regions,region,focus);};
  await show();
  out.textContent='ready';document.body.dataset.done='true';
}catch(error){out.textContent=String(error);document.body.dataset.error='true';}
