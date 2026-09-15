import '@fontsource/barlow/latin-400.css';
import '@fontsource/barlow/latin-500.css';
import '@fontsource/barlow/latin-600.css';
import '@fontsource/barlow-condensed/latin-600.css';
import './style.css';
import './embodied.css';
import {BrainView} from './brain';
import {FlyBodyView} from './body';
import {fetchJson,loadImage,type BrainMetadata,type ClipManifest,type Clip} from './types';
import type {CircuitMetadata} from './neural';

const base=import.meta.env.BASE_URL;
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const app=$('experiment');
app.innerHTML=`
<a class="skip-link" href="#experiment-main">Skip to the experiment</a>
<header class="app-bar"><a class="brand" href="${base}"><img src="${base}favicon.svg" alt="" width="32" height="32"/><span>NEUROFRAME</span></a><nav class="experiment-nav" aria-label="Experiments"><a href="${base}">Projected image</a><a href="${base}embodied.html" aria-current="page">Whole-branch baseline</a><a href="${base}live.html">Live synaptic cinema</a></nav></header>
<main id="experiment-main">
  <div class="workspace-heading"><div><h1>From pixels to spikes to movement.</h1><p>Measured fly connectivity. Simulated spikes. A body governed by physics.</p></div><a class="method-link" href="#science-method">Model & limitations</a></div>
  <div class="experiment-grid">
    <section class="experiment-viewers" aria-label="Neural activity and body response">
      <div class="paired-stage">
        <section class="neural-stage"><div class="stage-title"><h2>The brain</h2><select id="neural-view" aria-label="Brain viewpoint" disabled><option value="image">Image close-up</option><option value="full">Full brain</option><option value="oblique">Angled view</option><option value="side">Side view</option></select></div><div id="neural-canvas" class="brain-canvas"></div><div class="stage-caption"><strong id="brain-display-method">Image from synaptic spikes</strong><span id="brain-display-detail">Whole-neuron activity · density-normalized anatomy</span></div></section>
        <section class="body-stage"><div class="stage-title"><h2>The body</h2><select id="body-view" aria-label="Body viewpoint" disabled><option value="angled">Angled view</option><option value="front">Front view</option><option value="side">Side view</option><option value="top">Top view</option></select></div><div id="fly-canvas" class="brain-canvas"></div><div class="stage-caption">NeuroMechFly / MuJoCo · physical joints and contact<span>Neural outputs drive a CPG controller</span></div></section>
        <div id="experiment-loading" class="loading-state" role="status"><strong id="experiment-loading-title">Loading the neural experiment</strong><span id="experiment-loading-detail">Preparing the connectome and biomechanical fly…</span><button id="experiment-retry" class="button primary" hidden>Reload experiment</button></div>
      </div>
      <div class="experiment-stats"><span><strong id="network-count">—</strong> neurons simulated</span><span><strong id="network-edges">—</strong> weighted connections</span><span><strong id="sim-rate">—</strong>× real time</span><span><strong id="experiment-fps">—</strong> render FPS</span></div>
      <div class="experiment-transport"><button id="experiment-play" class="button primary" disabled>Pause experiment</button><button id="experiment-reset" class="button" disabled>Reset</button><div class="experiment-timeline"><label for="experiment-seek">Dataset frame <strong id="experiment-frame">001</strong> / 128</label><input id="experiment-seek" type="range" min="0" max="127" step="1" value="0" disabled/></div><span id="neural-time">0.000 s simulated</span></div>
      <p class="experiment-clock-note">Brain and body share a simulation clock, targeting 0.1× speed. Dataset playback is 10 FPS in simulation time. Drag either view to orbit, or use its viewpoint menu.</p>
      <section class="readout" aria-label="Measured model output"><div class="readout-heading"><h2>Signals leaving the brain</h2><span id="activity-summary">Waiting for spikes</span></div><div class="readout-grid"><div><span>DNa02 · left steering</span><strong id="dn-left">0.0 Hz</strong><meter id="dn-left-meter" min="0" max="200" value="0" aria-label="Left DNa02 firing rate"></meter></div><div><span>DNa02 · right steering</span><strong id="dn-right">0.0 Hz</strong><meter id="dn-right-meter" min="0" max="200" value="0" aria-label="Right DNa02 firing rate"></meter></div><div><span>DNp09 · forward drive</span><strong id="dn-forward">0.0 Hz</strong><meter id="dn-forward-meter" min="0" max="200" value="0" aria-label="DNp09 firing rate"></meter></div><div><span>MDN · backward drive</span><strong id="dn-back">0.0 Hz</strong><meter id="dn-back-meter" min="0" max="200" value="0" aria-label="MDN firing rate"></meter></div></div></section>
    </section>
    <aside class="inspector experiment-inspector" aria-label="Stimulation controls">
      <section class="source-section"><div class="section-heading"><h2>The stimulus</h2><span class="small-label">Dataset</span></div><div class="source-monitor"><canvas id="stimulus" width="160" height="90" role="img" aria-label="DOOM frame supplied to the external stimulation encoder"></canvas><span>DOOM / TEST</span></div><label for="experiment-clip" class="field-label">Dataset clip</label><select id="experiment-clip" class="clip-select" disabled></select><div class="source-facts"><span>80 × 60 original pixels</span><span>4 clips</span></div></section>
      <section class="display-section"><h2>How the brain is stimulated</h2><label for="stimulus-encoder" class="field-label">External controller</label><select id="stimulus-encoder" class="clip-select" disabled><option value="synaptic">Synaptic image controller</option><option value="luminance">Frame luminance → stimulation</option><option value="inverse">Earlier 37-group controller</option></select><p id="encoder-description" class="control-help">Feedback controls 8,534 upstream neurons. The 1,500 displayed neurons receive no direct stimulation.</p><div class="control-row"><label for="stimulus-rate">Maximum stimulation</label><output id="stimulus-rate-value">180 Hz</output></div><input id="stimulus-rate" type="range" min="0" max="300" step="10" value="180" disabled/><p class="control-help">This is artificial optical control, not a model of natural vision. The image controller uses strong stimulation; its input targets follow the reference model’s zero-refractory optical protocol.</p></section>
      <section class="causal-controls"><h2>Test the connection</h2><label class="check-row"><input id="synapses-enabled" type="checkbox" checked disabled/><span>Synaptic transmission</span></label><label class="check-row"><input id="stimulus-enabled" type="checkbox" checked disabled/><span>Video stimulation</span></label><button id="motor-pulse" class="button" disabled>Pulse forward command neurons</button><p class="control-help">The pulse directly stimulates DNp09 for 150 simulated ms. It is a separate positive control, independent of the video.</p></section>
    </aside>
  </div>
  <section id="science-method" class="science-method"><div><h2>What is actually connected?</h2><p>The video sets external stimulation rates. A leaky integrate-and-fire model propagates spikes through all 15,091,983 published weighted connections. Each displayed neuron receives one brightness from its own spike trace. In image-control mode, branch contributions are summed and normalized by anatomical density. No video texture colors the branches in this view.</p><p>The same simulation’s DNa02, DNp09 and MDN outputs modulate NeuroMechFly’s low-level gait controller. MuJoCo computes the joints, forces and contacts. This adapter is engineered; the ventral nerve cord, muscles and natural visual processing are not reconstructed. Body feedback is not sent into the brain in this prototype.</p><p><strong>Synaptic image control.</strong> An image solver proposes one target activity per whole neuron. A feedback controller stimulates a separate set of 8,534 real upstream neurons to approach those targets through the published connections. The renderer uses the resulting spikes, never the target activities. The result is a coarse 40 × 30 grayscale image, clearest from Image close-up. Its 15° reference view stays fixed as you orbit. The earlier 37-group controller remains available for comparison.</p><div class="science-links"><a href="${base}data/synaptic-control-comparison.png" target="_blank">Source vs. actual spikes</a><a href="${base}data/synaptic-playback.gif" target="_blank">Recorded spike playback</a><a href="${base}data/synaptic-control-validation.json" target="_blank">Sequence results and disconnection test</a><a href="${base}data/stimulation-validation.png" target="_blank">Earlier controller results</a><a href="https://github.com/philshiu/Drosophila_brain_model" target="_blank" rel="noreferrer">Shiu model and connection data</a><a href="https://neuromechfly.org/" target="_blank" rel="noreferrer">NeuroMechFly documentation</a></div></div><div><h2>Reading the result</h2><p>Firing rates are exponentially filtered over 100 simulated ms. Brightness is a display convention, not a calcium recording. The recorded comparison plays at 10 FPS in simulation time; live simulation is slower. High optical rates and the simplified input protocol are computational control choices, not a validated procedure for a living fly. The neuron model omits detailed membranes, graded signaling, plasticity and biological internal state.</p><p>Disable synaptic transmission to interrupt downstream propagation. Existing activity takes time to decay. Reset with transmission disabled for a clean comparison; only the externally stimulated neurons can then fire. Reset restores the seeded random sequence.</p><p>Scrubbing or changing clips changes the stimulus while preserving neural history. Reset clears the brain, body and movie together. A stationary or unstable fly is a model result; no fallback animation is substituted.</p><p class="license-note">FlyWire data: CC BY 4.0. Shiu reference code: MIT. NeuroMechFly and MuJoCo: Apache 2.0. DOOM dataset: CC0 per metadata. The body is a scan-derived model of another fly, not the original FlyWire specimen.</p></div></section>
  <footer class="app-footer"><span>Neuroframe · connectome and embodiment experiment</span><a href="${base}">Return to projection display</a></footer>
</main>`;

let brain:BrainView|null=null, body:FlyBodyView|null=null, worker:Worker|null=null;
let ready=false, playing=!matchMedia('(prefers-reduced-motion: reduce)').matches, pending=false, resetting=false;
let manifest:ClipManifest,clip:Clip,sheet:HTMLImageElement,frame=0,lastMovieTime=0,simTime=0;
let sceneTimer=0,raf=0,clipRequest=0,lastTick=0;
const source=$<HTMLCanvasElement>('stimulus'),ctx=source.getContext('2d',{alpha:false})!;
const small=document.createElement('canvas');small.width=80;small.height=60;
const smallContext=small.getContext('2d',{willReadFrequently:true})!;
ctx.imageSmoothingEnabled=false;smallContext.imageSmoothingEnabled=false;

function drawFrame(){
  if(!sheet)return;
  ctx.drawImage(sheet,(frame%clip.columns)*160,Math.floor(frame/clip.columns)*90,160,90,0,0,160,90);
  smallContext.drawImage(source,20,0,120,90,0,0,80,60);
  const pixels=smallContext.getImageData(0,0,80,60).data,values=new Float32Array(4800);
  for(let p=0;p<values.length;p++)values[p]=(.2126*pixels[p*4]+.7152*pixels[p*4+1]+.0722*pixels[p*4+2])/255;
  worker?.postMessage({type:'frame',values},[values.buffer]);
  $('experiment-frame').textContent=String(frame+1).padStart(3,'0');
  $<HTMLInputElement>('experiment-seek').value=String(frame);
  app.dataset.frame=String(frame);
}
async function selectClip(id:string){
  const next=manifest.clips.find(c=>c.id===id)!;const request=++clipRequest;
  const loaded=await loadImage(`${base}data/${next.file}`);
  if(request!==clipRequest)return;
  clip=next;sheet=loaded;frame=0;lastMovieTime=simTime;drawFrame();
}
function config(){
  const encoder=$<HTMLSelectElement>('stimulus-encoder').value;
  const inverse=encoder==='inverse',synaptic=encoder==='synaptic';
  $<HTMLInputElement>('stimulus-rate').disabled=!ready||inverse||synaptic;
  $('stimulus-rate-value').textContent=synaptic?'0–4,000 Hz':inverse?'0–160 Hz':`${$<HTMLInputElement>('stimulus-rate').value} Hz`;
  $('encoder-description').textContent=synaptic?'Feedback controls 8,534 upstream neurons. The 1,500 displayed neurons receive no direct stimulation.':inverse?'A calibrated local inverse controller chooses rates for 37 groups. The first spiking test produced poor image fidelity.':'An artificial encoder drives 1,024 identified visual projection cells. Downstream neurons receive only synaptic input.';
  worker?.postMessage({type:'config',encoder,maxRate:Number($<HTMLInputElement>('stimulus-rate').value),muted:!$<HTMLInputElement>('stimulus-enabled').checked,synapses:$<HTMLInputElement>('synapses-enabled').checked});
  app.dataset.encoder=encoder;app.dataset.synapses=String($<HTMLInputElement>('synapses-enabled').checked);
}
function updatePlaying(){
  $('experiment-play').textContent=playing?'Pause experiment':'Start experiment';app.dataset.playing=String(playing);
}
function fail(error:unknown){
  ready=false;playing=false;clearTimeout(sceneTimer);updatePlaying();
  $('experiment-loading').hidden=false;$('experiment-loading-title').textContent='The experiment could not continue';
  $('experiment-loading-detail').textContent=error instanceof Error?error.message:String(error);$('experiment-retry').hidden=false;
  app.dataset.ready='error';
  document.querySelectorAll<HTMLInputElement|HTMLButtonElement|HTMLSelectElement>('.experiment-transport button,.experiment-transport input,.experiment-inspector input,.experiment-inspector button,.experiment-inspector select').forEach(n=>{n.disabled=true;});
}
$('experiment-retry').onclick=()=>location.reload();
$('experiment-play').onclick=()=>{playing=!playing;updatePlaying();};
$('experiment-reset').onclick=()=>{
  playing=false;updatePlaying();frame=0;lastMovieTime=0;resetting=true;worker?.postMessage({type:'reset'});drawFrame();
};
$('neural-view').onchange=()=>brain?.goTo($<HTMLSelectElement>('neural-view').value as 'full'|'oblique'|'side'|'image');
$('body-view').onchange=()=>body?.setView($<HTMLSelectElement>('body-view').value);
$('motor-pulse').onclick=()=>{worker?.postMessage({type:'pulse'});playing=true;updatePlaying();};
$('experiment-seek').oninput=()=>{frame=Number($<HTMLInputElement>('experiment-seek').value);lastMovieTime=simTime;drawFrame();};
$('experiment-clip').onchange=()=>{void selectClip($<HTMLSelectElement>('experiment-clip').value).catch(fail);};
for(const id of ['stimulus-encoder','stimulus-rate','synapses-enabled','stimulus-enabled'])$(id).addEventListener('input',config);

async function start(){
  try{
    const [anatomy,circuit,clips,controller,imageView]=await Promise.all([
      fetchJson<BrainMetadata>(`${base}data/brain.json`),fetchJson<CircuitMetadata>(`${base}data/connectome.json`),
      fetchJson<ClipManifest>(`${base}data/clips.json`),fetchJson(`${base}data/stimulation-controller.json`),fetchJson<{normal:number[]}>(`${base}data/synaptic-image-operator.json`)]);
    manifest=clips;
    for(const c of clips.clips){const option=document.createElement('option');option.value=c.id;option.textContent=c.name;$('experiment-clip').append(option);}
    brain=new BrainView($('neural-canvas'),source,anatomy);brain.setNeuralActivity(new Float32Array(anatomy.neuronCount));brain.setExposure(1.7);brain.setImageView(imageView.normal);
    $('neural-canvas').querySelector('canvas')!.setAttribute('aria-label','Simulated neural spike traces on real fly anatomy. Use the Brain viewpoint menu for keyboard camera control.');
    brain.onContextLost=()=>fail(Error('The brain graphics context was lost. Reload to recover.'));
    body=new FlyBodyView($('fly-canvas'));
    worker=new Worker(new URL('./neural.worker.ts',import.meta.url),{type:'module'});
    let markWorkerReady:()=>void;
    let rejectWorkerReady:(error:Error)=>void;
    const workerReady=new Promise<void>((resolve,reject)=>{markWorkerReady=resolve;rejectWorkerReady=reject;});
    let previousSim=0,sampleStart=performance.now(),renders=0;
    worker.onmessage=event=>{
      const m=event.data;
      if(m.type==='error'){const error=Error(m.message);rejectWorkerReady(error);fail(error);return;}
      if(m.type==='ready'){markWorkerReady();return;}
      if(m.type!=='state')return;
      try{
        if(resetting){if(m.timeMs>0)return;body?.reset();simTime=0;resetting=false;}
        const delta=m.timeMs-simTime;simTime=m.timeMs;
        if(delta>0)body?.step(delta,m.motor.left,m.motor.right);
        brain?.setNeuralActivity(m.values);brain?.setActivityNormalization(m.encoder==='synaptic');pending=false;
        $('brain-display-method').textContent=m.encoder==='synaptic'?'Image from synaptic spikes':'White branches = spike traces';
        $('brain-display-detail').textContent=m.encoder==='synaptic'?'Whole-neuron activity · density-normalized anatomy':'1,500 displayed · full model simulated';
        if(playing&&simTime-lastMovieTime>=100){const advance=Math.floor((simTime-lastMovieTime)/100);frame=(frame+advance)%clip.frames;lastMovieTime+=advance*100;drawFrame();}
        $('neural-time').textContent=`${(simTime/1000).toFixed(3)} s simulated`;
        $('activity-summary').textContent=`${m.spikes.toLocaleString()} spikes · ${m.deliveries.toLocaleString()} connection deliveries`;
        const mean=(type:string,side?:string)=>{const rows=m.motor.rates.filter((n:{cellType:string;side:string})=>n.cellType===type&&(!side||n.side===side));return rows.reduce((s:number,n:{hz:number})=>s+n.hz,0)/Math.max(1,rows.length);};
        for(const [id,value] of [['left',mean('DNa02','left')],['right',mean('DNa02','right')],['forward',mean('DNp09')],['back',mean('MDN')]] as const){$(`dn-${id}`).textContent=`${value.toFixed(1)} Hz`;$<HTMLMeterElement>(`dn-${id}-meter`).value=value;}
        app.dataset.simTime=String(simTime);app.dataset.spikes=String(m.spikes);app.dataset.deliveries=String(m.deliveries);
        app.dataset.motor=String(Math.abs(m.motor.left)+Math.abs(m.motor.right));
        const bs=body?.getState();if(bs)app.dataset.bodyState=JSON.stringify(bs);
      }catch(error){fail(error);}
    };
    worker.onerror=event=>{const error=Error(event.message);rejectWorkerReady(error);fail(error);};
    worker.postMessage({type:'init',metadata:circuit,controller,url:`${base}data/connectome.bin.gz`,controlUrl:`${base}data/presynaptic-controller.json`,operatorUrl:`${base}data/synaptic-image-operator.bin`});
    await Promise.all([brain.load(()=>{}),body.load(text=>{if(app.dataset.ready!=='error')$('experiment-loading-detail').textContent=text;}),selectClip(clips.clips[0].id),workerReady]);
    ready=true;app.dataset.ready='true';$('experiment-loading').hidden=true;
    document.querySelectorAll<HTMLInputElement|HTMLButtonElement|HTMLSelectElement>('button:disabled,input:disabled,select:disabled').forEach(n=>{n.disabled=false;});
    $('network-count').textContent=circuit.neuronCount.toLocaleString();$('network-edges').textContent=circuit.edgeCount.toLocaleString();
    config();drawFrame();updatePlaying();
    const tick=()=>{
      sceneTimer=window.setTimeout(tick,20);
      if(ready&&playing&&!pending&&!document.hidden){pending=true;lastTick=performance.now();worker!.postMessage({type:'step',ms:2});}
      if(pending&&performance.now()-lastTick>15000)fail(Error('The neural worker stopped responding. Reload to recover.'));
    };tick();
    let previous=performance.now();
    const animate=(now:number)=>{
      raf=requestAnimationFrame(animate);
      if(document.hidden)return;
      brain!.render(now,Math.min(.1,(now-previous)/1000));body!.render();previous=now;renders++;
      if(now-sampleStart>=1000){$('experiment-fps').textContent=String(Math.round(renders*1000/(now-sampleStart)));$('sim-rate').textContent=Math.max(0,(simTime-previousSim)/(now-sampleStart)).toFixed(2);previousSim=simTime;sampleStart=now;renders=0;}
    };raf=requestAnimationFrame(animate);
  }catch(error){fail(error);}
}
window.addEventListener('pagehide',event=>{if(event.persisted)return;clearTimeout(sceneTimer);cancelAnimationFrame(raf);worker?.terminate();body?.dispose();brain?.dispose();});
void start();
