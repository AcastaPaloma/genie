/** Play DOOM on the complete fly brain, side by side: the generated frame, the brain it drives, and the
 * physical NeuroMechFly body below. The page connects to the Genie world model (experiments/play_server.py
 * through the /world proxy) by itself and starts as soon as the anatomy is uploaded. Keyboard only.
 *
 * The body uses an ARCADE motor adapter, unlike live.html: the descending-neuron readout is amplified
 * (bodyGain) and a twitch proportional to the brain's mean spike rate is added, so the fly visibly reacts;
 * physics runs at wall-clock speed instead of 100 ms per neural frame. It is labelled as such on the page. */
import {FullBrainView,PALETTES,type FullBrainMetadata,type Palette} from './full-brain';
import {FlyBodyView} from './body';
import {LiveSource} from './live-source';
import {WorldModelClient,parseKeys} from './world-model';
import {fetchJson} from './types';
import {fetchRegionManifest,isRegionName,type RegionManifest,type RegionName} from './brain-regions';
import {applyBrainFocus,describeFocus,isFocusName,type FocusName} from './brain-focus';

const base=import.meta.env.BASE_URL,query=new URL(location.href).searchParams,app=document.getElementById('play')!;
const controlWidth=Number(query.get('controlWidth')??640);
const bodyGain=Number(query.get('bodyGain')??4),twitchGain=Number(query.get('twitch')??1);
// The visual-input controller of live.html, unchanged.
const parameters={controller:'gate',gateMode:4,shunt:327680,rateScale:180,traceMs:40,traceTrigger:1,feedbackTicks:1,simulationBatchTicks:3,imageFeedback:4,supportGain:.5,useVisualInputs:1,seedCount:0,gateBias:6.99};
const VIEWS=['full','oblique','side'] as const,BODY_VIEWS=['angled','front','side','top'];
// Display processing of the simulated light only (level stretch, gamma, local contrast, saturation).
const LOOKS=[{name:'refined',autoLevels:true,gamma:.9,sharpen:.35,saturation:1.5,floor:.12},
             {name:'punchy',autoLevels:true,gamma:.75,sharpen:.8,saturation:1.9,floor:.12},
             {name:'raw',autoLevels:false,gamma:1,sharpen:0,saturation:1,floor:0}] as const;
let lookIndex=Math.max(0,LOOKS.findIndex(l=>l.name===(query.get('look')??'refined')));
// Where the generated frame sits in the controller's raster. Letterboxed across the whole raster (the
// live.html behaviour, ?frame=full here) its lower rows fall below the brain's ventral edge: measured on
// whole-arbor-640, only 8.1% of the status-bar band and 65.5% of the weapon band have any branch, so those
// parts of the picture cannot light up. Placed on the brain's dense band instead, both reach about 98%,
// still over 69% of the brain's cable. 480 x 270 keeps 3 raster pixels per game pixel, and the offsets are
// multiples of 3 so the pixel grid stays aligned with the game's pixels.
const fullFrame=query.get('frame')==='full',rasterScale=controlWidth/640;
const APERTURE={x:75*rasterScale,y:39*rasterScale,width:480*rasterScale,height:270*rasterScale};
const rasterPerGamePixel=(fullFrame?controlWidth:APERTURE.width)/160;
const GAME_CELLS:[number,number]=[controlWidth/rasterPerGamePixel,controlWidth*.75/rasterPerGamePixel];
const PIXEL_MODES:{name:string;cells:[number,number]|null;note:string}[]=[
  {name:'game pixels',cells:GAME_CELLS,note:'one cell per pixel of the 160 × 90 generated frame'},
  {name:'half pixels',cells:[GAME_CELLS[0]*2,GAME_CELLS[1]*2],note:'two cells per game pixel'},
  {name:'full detail',cells:null,note:''}];
let pixelIndex=PIXEL_MODES.findIndex(m=>m.name===(query.get('pixels')??'game pixels'));
if(pixelIndex<0)pixelIndex=0;
app.innerHTML=`<div id="doom"><span class="label">World model · generated frame${fullFrame?' · mapped across the whole brain':' · mapped onto the brain’s dense band'}</span></div><div id="brain"><span class="label" id="brain-label">139,255 neurons · light from simulated spikes</span></div><div id="fly"><span class="label" id="fly-label">NeuroMechFly · loading</span></div>
<div id="hud"><div id="status">Loading…</div><div id="body-status"></div><div id="keys">A / D turn · W / S straight · 0–7 raw codes · R new prompt · Q 3/25 passes · [ ] passes · Space pause · P colours · V brain view · B fly view · X look · Z pixels · nothing held = hold</div></div>
<div id="overlay"><div id="overlay-title">Loading the complete fly brain</div><div id="overlay-detail">139,255 neurons, every released branch</div></div>`;
const $=(id:string)=>document.getElementById(id)!;
const clamp=(value:number)=>Math.max(-1.2,Math.min(1.2,value));
let brain:FullBrainView,body:FlyBodyView,source:LiveSource,neural:Worker,physics:Worker,world:WorldModelClient;
let neuronCount=0,ready=false,playing=false,neuralBusy=false,bodyBusy=false,epoch=0,lastSerial=-1,nextCapture=0,frameId=0,completed=0,lastCompleted=0,sampleTime=performance.now(),brainFps='—',worldText='';
let lastSpikes=0,twitchPhase=0,command={left:0,right:0},meanHz=0,lastBodyStep=0,bodyBackend='',bodySteps=0;
let palette:Palette=(PALETTES as readonly string[]).includes(query.get('palette')??'')?query.get('palette') as Palette:'source';
let viewIndex=0,bodyViewIndex=0;
// Display-only choices (brain-focus.ts): every neuron is still simulated whatever is drawn or framed.
let region:RegionName=isRegionName(query.get('region'))?query.get('region') as RegionName:'all';
// The brain panel opens zoomed on the bottom of the brain; ?focus=whole shows the full view.
let focus:FocusName=isFocusName(query.get('focus'))?query.get('focus') as FocusName:'bottom';
let regions:RegionManifest;
function applyLook(){const look=LOOKS[lookIndex],pixels=PIXEL_MODES[pixelIndex];
  brain.setDisplay({...look,pixelCells:pixels.cells});
  app.dataset.look=look.name;app.dataset.pixels=pixels.name;if(regions)showRegion();}
function showRegion(){const look=LOOKS[lookIndex];
  const pixels=PIXEL_MODES[pixelIndex];
  $('brain-label').textContent=describeFocus(regions,region,focus)
    +(look.autoLevels?` · display: auto levels + local contrast (${look.name})`:' · display: raw light')
    +(pixels.cells?` · ${pixels.name}: ${pixels.note}`:' · full detail');
  app.dataset.region=region;app.dataset.focus=focus;}

function status(){
  $('status').textContent=`${brainFps} brain frames/s · ${playing?'':'paused · '}${worldText}`;
  $('body-status').textContent=`fly: ${bodyBackend||'loading'} · drive L ${command.left>=0?'+':''}${command.left.toFixed(2)} R ${command.right>=0?'+':''}${command.right.toFixed(2)} · ${meanHz.toFixed(0)} Hz mean spike rate · arcade adapter (descending-neuron readout ×${bodyGain} + spike-rate twitch ×${twitchGain})`;
}
function fail(error:unknown){
  console.error(error);ready=false;playing=false;world?.setRunning(false);
  const message=error instanceof Error?error.message:String(error);app.dataset.error=message;
  let overlay=document.getElementById('overlay');if(!overlay){overlay=document.createElement('div');overlay.id='overlay';overlay.innerHTML='<div id="overlay-title"></div><div id="overlay-detail"></div>';app.append(overlay);}
  $('overlay-title').textContent='Stopped';$('overlay-detail').textContent=message;
}
function setPlaying(value:boolean){playing=value;app.dataset.playing=String(value);world.setRunning(value);if(value){nextCapture=performance.now();lastBodyStep=performance.now();}status();}
/** Arcade motor command: amplified readout plus a spike-rate twitch. Not the measured adapter of live.html. */
function motorCommand(m:{spikes:number;motor:{forward:number;backward:number;turn:number}}){
  const delta=Math.max(0,m.spikes-lastSpikes);lastSpikes=m.spikes;
  meanHz=delta/Math.max(1,neuronCount)/.1;                       // each neural frame spans 100 ms of simulated time
  const excitement=Math.min(1,meanHz/40);
  twitchPhase++;
  const twitch=twitchGain*excitement*Math.sin(twitchPhase*2.4);
  const drive=bodyGain*(m.motor.forward-m.motor.backward)+.6*excitement;
  const turn=Math.tanh(bodyGain*m.motor.turn)+.8*twitch;
  command={left:clamp(drive-.55*turn),right:clamp(drive+.55*turn)};
}
function stepBody(){
  if(!ready||!playing||bodyBusy||document.hidden||!bodyBackend)return;
  const now=performance.now();
  // Wall-clock pacing so the fly moves in real time whatever the brain's frame rate; the native service needs exactly 100 ms.
  const ms=bodyBackend.startsWith('Native')?100:Math.max(10,Math.min(200,Math.round(now-lastBodyStep)));
  lastBodyStep=now;bodyBusy=true;
  physics.postMessage({type:'step',epoch,ms,left:command.left,right:command.right});
}
async function resetAll(){
  const resume=playing;playing=false;world.setRunning(false);
  while(neuralBusy||bodyBusy)await new Promise(resolve=>setTimeout(resolve,20));
  epoch++;lastSpikes=0;lastSerial=-1;command={left:0,right:0};
  neural.postMessage({type:'reset',epoch});physics.postMessage({type:'reset',epoch});brain.setNeuralActivity(new Float32Array(neuronCount));
  if(resume)setPlaying(true);
}

async function start(){
  try{
    const directory=`${base}data/full-brain-783/`;
    const [meta,operator,regionManifest]=await Promise.all([fetchJson<FullBrainMetadata>(directory+'brain.json'),fetchJson<{resolution:number[];footprintMicrometers:number[];center:number[]}>(directory+`whole-arbor-${controlWidth}.json`),fetchRegionManifest(directory,139255)]);
    neuronCount=meta.neuronCount;
    brain=new FullBrainView($('brain'),meta);brain.setPalette(palette);app.dataset.palette=palette;brain.setExposure(1.5);applyLook();brain.setReferenceProjection(operator.footprintMicrometers,operator.center);brain.setNeuralActivity(new Float32Array(meta.neuronCount));
    // The operator's footprint above stays the source-colour reference; a region or focus only reframes the camera.
    regions=regionManifest;if(region!=='all'||focus!=='whole')applyBrainFocus(brain,regions,region,focus);showRegion();
    brain.onContextLost=()=>fail(Error('Graphics context lost. Reload the page.'));
    body=new FlyBodyView($('fly'));
    source=new LiveSource(document.createElement('canvas'),operator.resolution[0],operator.resolution[1],fullFrame?undefined:APERTURE);
    world=new WorldModelClient({url:query.get('world')??`${base}world`,keys:parseKeys(query.get('keys'))??undefined,steps:query.has('steps')?Number(query.get('steps')):undefined,idleKey:query.get('idle')===null||query.get('idle')==='none'?null:query.get('idle')});
    world.canvas.setAttribute('aria-label','Frame generated by the world model');$('doom').append(world.canvas);
    world.onStatus=text=>{worldText=text;status();};world.onError=fail;
    neural=new Worker(new URL('./full-neural.worker.ts',import.meta.url),{type:'module'});
    physics=new Worker(new URL('./body.worker.ts',import.meta.url),{type:'module'});
    const neuralReady=new Promise<void>((resolve,reject)=>{
      neural.onmessage=async event=>{
        const m=event.data;
        if(m.type==='error'){neuralBusy=false;const error=Error(m.message);reject(error);fail(error);return;}
        if(m.type==='ready'){resolve();return;}
        if(m.type!=='state'||m.epoch!==epoch){if(m.type==='state')neuralBusy=false;return;}
        motorCommand(m);
        try{await brain.prepareNeuralFrame(m.values);brain.render(performance.now(),0);await brain.waitForDraw();}catch(error){fail(error);return;}
        neuralBusy=false;completed++;app.dataset.frames=String(completed);app.dataset.spikes=String(m.spikes);
      };
      neural.onerror=e=>{const error=Error(e.message);reject(error);fail(error);};
    });
    const bodyReady=new Promise<void>((resolve,reject)=>{
      physics.onmessage=event=>{
        const m=event.data;
        if(m.type==='error'){bodyBusy=false;const error=Error(m.message);reject(error);fail(error);return;}
        if(m.type==='ready'){bodyBackend=m.backend;$('fly-label').textContent=`NeuroMechFly · ${m.backend} · arcade motor adapter`;resolve();return;}
        bodyBusy=false;if(m.epoch!==epoch)return;
        body.applyPhysicalState(m);bodySteps++;app.dataset.bodySteps=String(bodySteps);
        const state=body.getState();if(state)app.dataset.fly=JSON.stringify({time:state.time,position:state.position.map(v=>Number(v.toFixed(3)))});
      };
      physics.onerror=e=>{const error=Error(e.message);reject(error);fail(error);};
    });
    neural.postMessage({type:'init',base:directory,controlWidth,parameters});
    physics.postMessage({type:'init',runtime:new URL(`${base}body/runtime.js`,location.href).href,assets:`${base}body/assets`,timestep:.00025,native:'http://127.0.0.1:8769'});
    await Promise.all([brain.loadLightCache(directory),body.load(()=>{}),neuralReady,bodyReady]);
    // The brain panel is drawn from the cable cache, which already integrates
    // every released branch of all 139,255 neurons, so the 7 GB of indexed
    // skeletons is uploaded only once the camera leaves that fixed front view.
    brain.onGeometryError=fail;
    brain.loadOnDemand(directory,(loaded,total)=>{
      app.dataset.anatomyLoaded=String(loaded);
      const detail=document.getElementById('overlay-detail');
      if(detail)detail.textContent=`${loaded.toLocaleString()} / ${total.toLocaleString()} neurons uploaded`;
    });
    body.setView(BODY_VIEWS[bodyViewIndex]);
    $('overlay-title').textContent='Connecting to the world model';$('overlay-detail').textContent=world.url;
    await world.connect();
    if(!await source.load(world))throw Error('The world model could not become the stimulus.');
    world.attachKeyboard();
    neural.postMessage({type:'config',synapses:true,muted:false});
    document.getElementById('overlay')?.remove();
    ready=true;app.dataset.ready='true';
    setPlaying(true);
    let lastPost=0;
    const loop=()=>{
      window.setTimeout(loop,5);const now=performance.now();
      if(!ready||!playing||document.hidden||neuralBusy||now<nextCapture||!source.ready)return;
      // A held frame is re-presented four times a second so the simulation keeps integrating it.
      if(source.frameSerial===lastSerial&&now-lastPost<250)return;
      lastSerial=source.frameSerial;nextCapture=now+90;lastPost=now;
      const capturedAt=performance.now(),values=source.capture(false);if(source.lastCapture)brain.setSourceFrame(source.lastCapture);neuralBusy=true;
      neural.postMessage({type:'step',epoch,values,capturedAt,decodedAt:source.decodedAt,sourceTime:source.mediaTime,frameId:++frameId},[values.buffer]);
    };
    loop();
    window.setInterval(stepBody,40);
    let previous=performance.now();
    const animate=(now:number)=>{
      requestAnimationFrame(animate);if(document.hidden)return;
      if(now-previous>=1000/30){brain.render(now,Math.min(.1,(now-previous)/1000),true);body.render(true);previous=now;}
      if(now-sampleTime>=1000){brainFps=((completed-lastCompleted)*1000/(now-sampleTime)).toFixed(1);sampleTime=now;lastCompleted=completed;status();}
    };
    requestAnimationFrame(animate);
  }catch(error){fail(error);}
}

document.addEventListener('keydown',event=>{
  if(!ready||event.repeat||event.metaKey||event.ctrlKey||event.altKey)return;
  const target=event.target;if(target instanceof HTMLElement&&/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))return;
  const key=event.key.length===1?event.key.toUpperCase():event.key;
  if(key===' '){setPlaying(!playing);event.preventDefault();}
  else if(key==='P'){palette=PALETTES[(PALETTES.indexOf(palette)+1)%PALETTES.length];brain.setPalette(palette);app.dataset.palette=palette;brain.render(performance.now(),0,false);}
  else if(key==='V'){viewIndex=(viewIndex+1)%VIEWS.length;brain.goTo(VIEWS[viewIndex]);brain.render(performance.now(),0,false);}
  else if(key==='B'){bodyViewIndex=(bodyViewIndex+1)%BODY_VIEWS.length;body.setView(BODY_VIEWS[bodyViewIndex]);body.render();}
  else if(key==='X'){lookIndex=(lookIndex+1)%LOOKS.length;applyLook();brain.render(performance.now(),0,false);}
  else if(key==='Z'){pixelIndex=(pixelIndex+1)%PIXEL_MODES.length;applyLook();brain.render(performance.now(),0,false);}
  else if(key==='R'){void resetAll();}   // the world client takes its own new prompt on the same key
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&ready&&playing)setPlaying(false);});
window.addEventListener('pagehide',event=>{if(event.persisted)return;neural?.terminate();physics?.terminate();world?.dispose();source?.dispose();brain?.dispose();body?.dispose();});
void start();
