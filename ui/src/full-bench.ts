import {LiveSource} from './live-source';
import {anatomicalAperture} from './anatomical-aperture';
const result=document.getElementById('result')!,canvas=document.getElementById('comparison') as HTMLCanvasElement;
const ctx=canvas.getContext('2d')!,query=new URL(location.href).searchParams;
const worker=new Worker(new URL('./full-neural.worker.ts',import.meta.url),{type:'module'});
let next:{resolve:(m:any)=>void;reject:(e:Error)=>void}|null=null;
worker.onmessage=event=>{if(event.data.type==='error'){next?.reject(Error(event.data.message));return;}next?.resolve(event.data);next=null;};
worker.onerror=event=>next?.reject(Error(event.message));
const send=(message:object)=>new Promise<any>((resolve,reject)=>{next={resolve,reject};worker.postMessage(message);});
function floatPairsBase64(first:Float32Array,second:Float32Array){
  const packed=new Float32Array(first.length+second.length);packed.set(first);packed.set(second,first.length);
  const bytes=new Uint8Array(packed.buffer);let text='';for(let offset=0;offset<bytes.length;offset+=32768)text+=String.fromCharCode(...bytes.subarray(offset,offset+32768));return btoa(text);
}
try{
  const directory='/data/full-brain-783/',collector=query.get('collector'),label=query.get('label')??'default';
  const controlWidth=Number(query.get('controlWidth')??320);if(controlWidth!==320&&controlWidth!==640)throw Error('Control width must be 320 or 640.');
  const meta=await (await fetch(directory+`whole-arbor-${controlWidth}.json`)).json(),raw=await (await fetch(directory+`whole-arbor-${controlWidth}.bin`)).arrayBuffer();
  canvas.width=meta.resolution[0]*2;canvas.height=meta.resolution[1];
  const [version,pixels,neurons,entries]=new Uint32Array(raw,0,4);if(version!==784||neurons!==139255)throw Error('Complete anatomy required.');
  const row=new Uint32Array(raw,16,entries),column=new Uint32Array(raw,16+4*entries,entries),weight=new Float32Array(raw,16+8*entries,entries);
  const mask=new Uint8Array(pixels);for(const id of row)mask[id]=1;const support=mask.reduce((a,b)=>a+b,0);
  const parameters={...Object.fromEntries(['step','gain','depth','maxDrive','rateScale','feedbackTicks','imageIterations','currentIterations','sensitivity','rateFloor','calibrate','seedCount','kp','ki','seedDrive','minDrive','gateMode','shunt','traceMs','traceTrigger','imageFeedback','imageAnchor','seedCoverage','simulationBatchTicks','supportGain','edgeWeight','useVisualInputs','gateBias'].filter(key=>query.has(key)).map(key=>[key,Number(query.get(key))])),controller:query.get('controller')??'heuristic'};
  const ready=await send({type:'init',base:directory,parameters,controlWidth});
  const aperture=query.has('fit')?anatomicalAperture(mask,meta.resolution[0],meta.resolution[1],4/3,Number(query.get('fit'))):undefined;
  const source=new LiveSource(document.createElement('canvas'),meta.resolution[0],meta.resolution[1],aperture);
  const frames:any[]=[];let epoch=0,frameId=0;
  const post=async(record:object)=>{if(collector){const r=await fetch(collector,{method:'POST',headers:{'Content-Type':'text/plain'},body:JSON.stringify(record)});if(!r.ok)throw Error('Local validation collector failed.');}};
  const sourceUrl=query.get('source'),steps=Number(query.get('steps')??20);
  if(!Number.isInteger(steps)||steps<1||steps>10000)throw Error('Evaluation length must be 1–10,000 frames.');
  const snapshots=new Set((query.get('snapshots')??'').split(',').filter(Boolean).map(Number));
  if([...snapshots].some(i=>!Number.isInteger(i)||i<0||i>=steps))throw Error('Snapshot indices must be inside the evaluated sequence.');
  for(const clip of sourceUrl?['map01']:['map01','map02','map03']){
    const url=sourceUrl??'/data/live-clips/'+clip+'.mp4';
    if(query.has('fileInput')){
      const response=await fetch(url);if(!response.ok)throw Error('Video File fixture unavailable.');
      const blob=await response.blob();await source.load(new File([blob],url.split('/').at(-1)??'input-video',{type:blob.type}));
    }else await source.load(url);
    for(let i=0;i<steps;i++){
      const t=query.has('hold')?Number(query.get('hold')):Number(query.get('start')??0)+i*.1;
      if(!Number.isFinite(t)||t<0||t>=source.video.duration)throw Error('Diagnostic source time is outside the decoded video.');
      if(t!==source.video.currentTime)await new Promise<void>(resolve=>{source.video.onseeked=()=>resolve();source.video.currentTime=t;});
      const target=source.capture(false),state=await send({type:'step',epoch,audit:snapshots.has(i)||(query.has('audit')&&i===steps-1),values:target,capturedAt:performance.now(),decodedAt:performance.now(),sourceTime:t,frameId:frameId++});
      const prediction=new Float32Array(pixels);for(let e=0;e<entries;e++)prediction[row[e]]+=weight[e]*state.values[column[e]];
      let squared=0,maskedSquared=0,mean=0;for(let p=0;p<pixels;p++){const error=(prediction[p]-target[p])**2;squared+=error;if(mask[p]){maskedSquared+=error;mean+=target[p];}}mean/=support;
      let variance=0;for(let p=0;p<pixels;p++)if(mask[p])variance+=(target[p]-mean)**2;
      let apertureMse:number|undefined;
      if(aperture){let error=0;for(let y=aperture.y;y<aperture.y+aperture.height;y++)for(let x=aperture.x;x<aperture.x+aperture.width;x++){const id=y*meta.resolution[0]+x;error+=(prediction[id]-target[id])**2;}apertureMse=error/(aperture.width*aperture.height);}
      let referenceMse:number|undefined;
      if(state.referenceRates){const ideal=new Float32Array(pixels);for(let e=0;e<entries;e++)ideal[row[e]]+=weight[e]*state.referenceRates[column[e]]/ready.rateScale;let error=0;for(let p=0;p<pixels;p++)if(mask[p])error+=(ideal[p]-target[p])**2;referenceMse=error/support;}
      const metrics={referenceMse,apertureMse,clip,index:i,time:t,computeMs:state.computeMs,auditMs:state.auditMs,mse:squared/pixels,maskedMse:maskedSquared/support,constantMse:variance/support,inputSpikes:state.inputSpikes,downstreamSpikes:state.downstreamSpikes,seedSpikes:state.seedSpikes,visualInputSpikes:state.visualInputSpikes,synapseDependentSpikes:state.synapseDependentSpikes,seedLightFraction:state.seedLightFraction,visualInputLightFraction:state.visualInputLightFraction,synapseDependentLightFraction:state.synapseDependentLightFraction};frames.push(metrics);
      const display=new ImageData(meta.resolution[0]*2,meta.resolution[1]);
      for(let y=0;y<meta.resolution[1];y++)for(let x=0;x<meta.resolution[0];x++)for(let side=0;side<2;side++){
        const p=y*meta.resolution[0]+x,k=4*(y*meta.resolution[0]*2+x+side*meta.resolution[0]),value=(side?prediction[p]:target[p])*255;
        display.data[k]=value;display.data[k+1]=value;display.data[k+2]=value;display.data[k+3]=255;
      }ctx.putImageData(display,0,0);
      const pixelRecord=()=>controlWidth>320?{pixelEncoding:'float32-le: target[P], prediction[P]',resolution:meta.resolution,pixelsBase64:floatPairsBase64(target,prediction)}:{resolution:meta.resolution,target:Array.from(target),prediction:Array.from(prediction)};
      const neuronRatesBase64=state.referenceRates&&state.measuredRates?floatPairsBase64(state.referenceRates,state.measuredRates):undefined;
      const auditRecord={worstCells:state.worstCells,neuronRatesEncoding:neuronRatesBase64?'float32-le: requested Hz[N], measured Hz[N], complete model index order':undefined,neuronRatesBase64};
      if(query.has('sequence')||snapshots.has(i))await post({kind:'frame',label:label+'-f'+String(i).padStart(4,'0'),clip,sourceUrl,metrics,aperture,...auditRecord,...pixelRecord()});
      if(i===steps-1)await post({kind:'frame',label,clip,sourceUrl,metrics,aperture,...auditRecord,...pixelRecord()});
      result.textContent=JSON.stringify({progress:`${clip}: ${i+1} frames`,...metrics});
    }
  }
  const darkFrames:any[]=[];
  if(query.has('dark')){
    const importance=new Float64Array(neurons);for(let e=0;e<entries;e++)importance[column[e]]+=weight[e];
    let priorGated=frames.at(-1).inputSpikes-(frames.at(-1).seedSpikes??0)-(frames.at(-1).visualInputSpikes??0);
    for(let i=0;i<Number(query.get('dark'));i++){
      const state=await send({type:'step',epoch,values:new Float32Array(pixels),capturedAt:performance.now(),decodedAt:performance.now(),sourceTime:0,frameId:frameId++});
      let light=0;for(let id=0;id<neurons;id++)light+=importance[id]*state.values[id];
      const gated=state.inputSpikes-(state.seedSpikes??0)-(state.visualInputSpikes??0);
      darkFrames.push({millisecondsAfterBlack:(i+1)*100,meanAnatomicalLight:light/support,newGatedSpikes:gated-priorGated});priorGated=gated;
    }
  }
  worker.postMessage({type:'config',synapses:false,muted:false});await send({type:'reset',epoch:++epoch});
  const cut=await send({type:'step',epoch,values:new Float32Array(pixels).fill(.7),capturedAt:performance.now(),decodedAt:performance.now(),sourceTime:0,frameId:frameId++});
  if(parameters.controller==='gate'&&(cut.synapseDependentSpikes!==0||((ready.seeds??0)>0&&cut.seedSpikes===0)))throw Error('Subthreshold-gate causal intervention failed.');
  if(cut.downstreamSpikes!==0||cut.deliveries!==0||cut.inputSpikes===0)throw Error('Full-graph causal intervention failed.');
  const summary={status:'completed',causality:'passed',recognizability:'not established by this numerical harness',scope:'Full graph and complete cable operator; excludes 3D render and body timing',label,parameters,controlWidth,resolution:meta.resolution,operatorSha256:meta.operatorSha256,aperture,sourceUrl,snapshotIndices:[...snapshots],sourceNativeResolution:[source.video.videoWidth,source.video.videoHeight],inputMechanism:query.has('fileInput')?'Browser File object and local object URL':'Video URL',sourceStart:Number(query.get('start')??0),heldSourceTime:query.has('hold')?Number(query.get('hold')):undefined,neurons,entries,inputCount:ready.inputs,seedCount:ready.seeds,seedIds:Array.from(ready.seedIds??[]),visualInputIds:Array.from(ready.visualInputIds??[]),anatomySha256:meta.sourceAnatomySha256,
    meanComputeMs:frames.reduce((s,x)=>s+x.computeMs,0)/frames.length,frames,darkFrames,cut:{inputSpikes:cut.inputSpikes,downstreamSpikes:cut.downstreamSpikes,deliveries:cut.deliveries,seedSpikes:cut.seedSpikes,visualInputSpikes:cut.visualInputSpikes,synapseDependentSpikes:cut.synapseDependentSpikes,synapseDependentLightFraction:cut.synapseDependentLightFraction}};
  await post({kind:'summary',label,summary});result.textContent=JSON.stringify({...summary,visualInputIds:undefined,seedIds:undefined,frames:summary.frames.length});document.body.dataset.done='true';source.dispose();worker.terminate();
}catch(error){result.textContent=String(error);document.body.dataset.error='true';worker.terminate();}
