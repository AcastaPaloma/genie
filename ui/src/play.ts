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
import {fetchRegionManifest,isRegionName,regionMask,type RegionManifest,type RegionName} from './brain-regions';
import {applyBrainFocus,describeFocus,isFocusName,type FocusName} from './brain-focus';

const base=import.meta.env.BASE_URL,query=new URL(location.href).searchParams,app=document.getElementById('play')!;
const controlWidth=Number(query.get('controlWidth')??640);
let bodyGain=Number(query.get('bodyGain')??9),twitchGain=Number(query.get('twitch')??2);
// The visual-input controller of live.html, unchanged.
const parameters={controller:'gate',gateMode:4,shunt:327680,rateScale:Number(query.get('rate')??840),traceMs:40,traceTrigger:1,feedbackTicks:1,simulationBatchTicks:3,imageFeedback:4,supportGain:.5,useVisualInputs:1,seedCount:0,gateBias:6.99};
const VIEWS=['full','oblique','side'] as const,BODY_VIEWS=['angled','front','side','top'];
// Which panels are on screen. Everything keeps running when a panel is hidden: the world model still
// generates, the simulation still integrates every frame, and the body still steps.
const SCREENS=[{name:'all',label:'all three'},{name:'brain',label:'the brain'},
  {name:'doom',label:'the generated game'},{name:'fly',label:'the fly'}] as const;
let screenIndex=Math.max(0,SCREENS.findIndex(s=>s.name===(query.get('screen')??'all')));
// Display processing of the simulated light only (level stretch, gamma, local contrast, saturation).
const LOOKS=[{name:'refined',autoLevels:true,gamma:1.15,sharpen:.35,saturation:1.35,floor:.06},
             {name:'punchy',autoLevels:true,gamma:.9,sharpen:.8,saturation:1.7,floor:.06},
             {name:'raw',autoLevels:false,gamma:1,sharpen:0,saturation:1,floor:0}] as const;
// Overall brightness, adjustable live with - and = because the right level depends on the room and screen.
let brightness=Math.min(2,Math.max(.1,Number(query.get('brightness')??.65)));
let lookIndex=Math.max(0,LOOKS.findIndex(l=>l.name===(query.get('look')??'refined')));
// Where the generated frame is drawn into the controller's raster. Share of each band's pixels that have
// any branch behind them, measured on whole-arbor-640 (content rows 0-59 scene, 60-77 weapon, 78-89 status
// bar), with the share of the brain's cable inside the rectangle:
//   full  640x360 at (0,60):    scene 59.7%  weapon 58.0%  status  5.1%  cable 100%   <- lower rows fall
//                                                                                        below the brain
//   band  480x270 at (78,60):   scene 58.4%  weapon 98.8%  status 85.2%  cable 75.6%
//   tight 320x180 at (172,138): scene 98.5%  weapon 99.1%  status 86.8%  cable 46.6%  <- default
// Offsets are multiples of the cell size, so the pixel grid stays aligned with the game's pixels.
//   strict 224x126 at (351,181): every row of the picture fully supported, cable 23.5%
// featherY fades the bands that cannot light up: at tight, support is >=99% for game rows 9-80 and falls
// to 84-94% outside them, so about 9 game rows top and bottom fade while the usable middle stays crisp.
const FRAMES={tight:{x:172,y:132,width:320,height:180,featherY:18,feather:6},
  band:{x:78,y:60,width:480,height:270,featherY:18,feather:9},
  strict:{x:351,y:181,width:224,height:126,featherY:4,feather:4},full:null} as const;
const frameName=(['tight','band','strict','full'] as const).includes(query.get('frame') as never)?query.get('frame') as keyof typeof FRAMES:'tight';
const fullFrame=frameName==='full',rasterScale=controlWidth/640,placement=FRAMES[frameName];
const APERTURE=placement?{x:placement.x*rasterScale,y:placement.y*rasterScale,width:placement.width*rasterScale,
  height:placement.height*rasterScale,feather:placement.feather*rasterScale,featherY:placement.featherY*rasterScale}:undefined;
const rasterPerGamePixel=(APERTURE?APERTURE.width:controlWidth)/160;
const GAME_CELLS:[number,number]=[controlWidth/rasterPerGamePixel,controlWidth*.75/rasterPerGamePixel];
// Start the grid at the picture's own corner so each cell is one game pixel whatever the placement.
const GRID_ORIGIN:[number,number]=APERTURE?[APERTURE.x/controlWidth,APERTURE.y/(controlWidth*.75)]:[0,0];
const PIXEL_MODES:{name:string;cells:[number,number]|null;note:string}[]=[
  {name:'game pixels',cells:GAME_CELLS,note:'one cell per pixel of the 160 × 90 generated frame'},
  {name:'half pixels',cells:[GAME_CELLS[0]*2,GAME_CELLS[1]*2],note:'two cells per game pixel'},
  {name:'full detail',cells:null,note:''}];
let pixelIndex=PIXEL_MODES.findIndex(m=>m.name===(query.get('pixels')??'game pixels'));
if(pixelIndex<0)pixelIndex=0;
// Loading stages, each reporting the count it genuinely has. The cable cache and
// the connectome do not load neuron by neuron, so neither pretends to: only the
// deferred skeleton upload has a real per-neuron count, reported in the HUD.
const STAGES=[['cache','Anatomical cache'],['sim','Simulator'],['body','Fly body'],['world','World model']] as const;
function stage(id:string,state:'wait'|'load'|'done',value:string){
  const row=document.getElementById(`stage-${id}`);if(!row)return;
  row.dataset.state=state;
  const cell=row.querySelector('.stage-value');if(cell)cell.textContent=value;
}
app.innerHTML=`<div id="doom"><span class="label">World model · generated frame${fullFrame?' · mapped across the whole brain':frameName==='band'?' · mapped onto the brain’s dense band':frameName==='strict'?' · mapped onto fully supported cable only':' · mapped onto supported cable'}</span></div><div id="brain"><span class="label" id="brain-label">139,255 neurons · light from simulated spikes</span></div><div id="fly"><span class="label" id="fly-label">NeuroMechFly · loading</span></div>
<div id="hud"><div id="status">Loading…</div><div id="body-status"></div><div id="anatomy-status"></div><div id="keys">A / D turn · W / S straight · 0–7 raw codes · R new prompt · Q 3/25 passes · [ ] passes · Space pause · P colours · V brain view · B fly view · X look · Z pixels · T screen · − = brightness · , . fly energy · K L game speed · G air mode · no code fires the weapon · nothing held = hold</div></div>
<div id="overlay"><div id="overlay-title">Loading the complete fly brain</div><div id="overlay-detail">139,255 neurons · 268,139,177 released branches</div>
<ul id="overlay-stages">${STAGES.map(([id,name])=>`<li id="stage-${id}" data-state="wait"><span class="stage-name">${name}</span><span class="stage-value">waiting</span></li>`).join('')}</ul></div>`;
const $=(id:string)=>document.getElementById(id)!;
// How far the gait adapter may be driven. 1.2 was the live.html range; the arcade page allows more.
// Past about 1.3 the gait adapter stops walking properly and the fly shuffles on the spot, so speed comes
// from running the body clock faster instead: ?bodySpeed= simulated seconds per wall second.
const DRIVE_LIMIT=Number(query.get('drive')??1.3);
const BODY_SPEED=Math.min(6,Math.max(.25,Number(query.get('bodySpeed')??3)));
const clamp=(value:number)=>Math.max(-DRIVE_LIMIT,Math.min(DRIVE_LIMIT,value));
let brain:FullBrainView,body:FlyBodyView,source:LiveSource,neural:Worker,physics:Worker,world:WorldModelClient;
let neuronCount=0,ready=false,playing=false,neuralBusy=false,bodyBusy=false,epoch=0,lastSerial=-1,nextCapture=0,frameId=0,completed=0,lastCompleted=0,sampleTime=performance.now(),brainFps='—',worldText='';
let lastSpikes=0,twitchPhase=0,command={left:0,right:0},meanHz=0,lastBodyStep=0,bodyBackend='',bodySteps=0;
let upsideDownSince=0,rescues=0;
// Air mode: the body worker scales gravity down and kicks the fly up, so the gait keeps running with
// nothing underneath it. Physics being toyed with, stated in the fly label. WASM backend only.
let airMode=false,airAvailable=true,airMessage='';
let palette:Palette=(PALETTES as readonly string[]).includes(query.get('palette')??'')?query.get('palette') as Palette:'source';
let viewIndex=0,bodyViewIndex=0;
// Display-only choices (brain-focus.ts): every neuron is still simulated whatever is drawn or framed.
let region:RegionName=isRegionName(query.get('region'))?query.get('region') as RegionName:'all';
// The brain panel frames the anatomy the picture is mapped onto, so the whole frame is on screen: sky at
// the top, weapon and status bar at the bottom. ?focus=bottom or whole uses the region framing instead.
let focus:FocusName|'image'=isFocusName(query.get('focus'))?query.get('focus') as FocusName:'image';
let imageBounds:[number[],number[]]|null=null;
let regions:RegionManifest;
function flyLabel(){
  $('fly-label').textContent=`NeuroMechFly · ${bodyBackend||'loading'} · arcade motor adapter`
    +(airMode?' · air mode: gravity reduced, legs flailing':'')
    +(airAvailable?'':` · ${airMessage||'air mode needs the browser physics backend'}`);
}
function showScreen(){
  app.dataset.screen=SCREENS[screenIndex].name;
  // Redraw immediately at the new size rather than waiting for the next neural frame.
  if(brain)brain.render(performance.now(),0,false);
  if(body)body.render();
  status();
}
function applyLook(){const look=LOOKS[lookIndex],pixels=PIXEL_MODES[pixelIndex];
  brain.setDisplay({...look,brightness,pixelCells:pixels.cells,pixelOrigin:GRID_ORIGIN});
  app.dataset.look=look.name;app.dataset.pixels=pixels.name;if(regions)showRegion();}
function showRegion(){const look=LOOKS[lookIndex];
  const pixels=PIXEL_MODES[pixelIndex];
  $('brain-label').textContent=describeFocus(regions,region,focus==='image'?'whole':focus)+(focus==='image'?' · framed on the picture area':'')
    +(look.autoLevels?` · display: auto levels + local contrast (${look.name})`:' · display: raw light')
    +(pixels.cells?` · ${pixels.name}: ${pixels.note}`:' · full detail');
  app.dataset.region=region;app.dataset.focus=focus;}

function status(){
  const screen=SCREENS[screenIndex];
  $('status').textContent=`${brainFps} brain frames/s · ${playing?'':'paused · '}${worldText}`
    +(screen.name==='all'?'':` · showing ${screen.label} (T)`)+` · brightness ${brightness.toFixed(1)} (− =)`;
  $('body-status').textContent=`fly: ${bodyBackend||'loading'} · ${BODY_SPEED}× body speed${rescues?` · righted ${rescues}×`:''} · drive L ${command.left>=0?'+':''}${command.left.toFixed(2)} R ${command.right>=0?'+':''}${command.right.toFixed(2)} · ${meanHz.toFixed(0)} Hz mean spike rate · arcade adapter (readout ×${bodyGain.toFixed(1)} + twitch ×${twitchGain.toFixed(1)}, , . to change)`;
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
  // Two beating frequencies plus a little noise, quickening with the brain's own firing rate, so the fly
  // jitters rather than oscillating on one clean sine.
  twitchPhase+=.6+1.8*excitement;
  const twitch=twitchGain*excitement*(.7*Math.sin(twitchPhase*2.4)+.4*Math.sin(twitchPhase*5.7+1.3)+.3*(Math.random()*2-1));
  // Forward drive grows with the gain too, so a louder brain walks faster instead of only shaking.
  const drive=bodyGain*(m.motor.forward-m.motor.backward)+.3*bodyGain*excitement
    +.35*twitchGain*excitement*Math.sin(twitchPhase*3.1);
  const turn=Math.tanh(bodyGain*m.motor.turn)+.5*twitch;
  // Scale both legs together when they exceed the limit, rather than clipping each one: clipping cancels the
  // forward component and leaves the fly spinning on the spot.
  let left=drive-.55*turn,right=drive+.55*turn;
  const peak=Math.max(Math.abs(left),Math.abs(right));
  if(peak>DRIVE_LIMIT){left*=DRIVE_LIMIT/peak;right*=DRIVE_LIMIT/peak;}
  command={left:clamp(left),right:clamp(right)};
}
function stepBody(){
  if(!ready||!playing||bodyBusy||document.hidden||!bodyBackend)return;
  const now=performance.now();
  // Wall-clock pacing so the fly moves in real time whatever the brain's frame rate; the native service needs exactly 100 ms.
  const ms=bodyBackend.startsWith('Native')?100:Math.max(10,Math.min(150,Math.round((now-lastBodyStep)*BODY_SPEED)));
  lastBodyStep=now;bodyBusy=true;
  physics.postMessage({type:'step',epoch,ms,left:command.left,right:command.right});
}
async function resetAll(){
  const resume=playing;playing=false;world.setRunning(false);
  while(neuralBusy||bodyBusy)await new Promise(resolve=>setTimeout(resolve,20));
  epoch++;lastSpikes=0;lastSerial=-1;command={left:0,right:0};
  if(airMode){airMode=false;physics.postMessage({type:'air',on:false,epoch});flyLabel();}
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
    regions=regionManifest;
    // World rectangle of the raster area the frame is drawn into, with a small margin.
    const [cx,cy]=operator.center,[fw,fh]=operator.footprintMicrometers,[rw,rh]=operator.resolution;
    const area=APERTURE??{x:0,y:(rh-rw*.5625)/2,width:rw,height:rw*.5625};
    const toWorldX=(x:number)=>cx+(x/rw-.5)*fw,toWorldY=(y:number)=>cy+(.5-y/rh)*fh;
    const margin=.04*Math.max(area.width/rw*fw,area.height/rh*fh);
    const [zLow,zHigh]=[meta.bounds[0][2],meta.bounds[1][2]];
    imageBounds=[[toWorldX(area.x)-margin,toWorldY(area.y+area.height)-margin,zLow],
                 [toWorldX(area.x+area.width)+margin,toWorldY(area.y)+margin,zHigh]];
    if(focus==='image')brain.setRegion(region,region==='all'?null:regionMask(regions,region),imageBounds,regions.regions[region]);
    else if(region!=='all'||focus!=='whole')applyBrainFocus(brain,regions,region,focus as FocusName);
    showRegion();
    brain.onContextLost=()=>fail(Error('Graphics context lost. Reload the page.'));
    body=new FlyBodyView($('fly'));
    source=new LiveSource(document.createElement('canvas'),operator.resolution[0],operator.resolution[1],APERTURE);
    world=new WorldModelClient({url:query.get('world')??`${base}world`,keys:parseKeys(query.get('keys'))??undefined,steps:query.has('steps')?Number(query.get('steps')):undefined,idleKey:query.get('idle')===null||query.get('idle')==='none'?null:query.get('idle'),
      targetFps:query.get('fps')==='max'?null:Math.max(1,Number(query.get('fps')??5))});
    world.canvas.setAttribute('aria-label','Frame generated by the world model');$('doom').append(world.canvas);
    world.onStatus=text=>{worldText=text;status();};world.onError=fail;
    neural=new Worker(new URL('./full-neural.worker.ts',import.meta.url),{type:'module'});
    physics=new Worker(new URL('./body.worker.ts',import.meta.url),{type:'module'});
    const neuralReady=new Promise<void>((resolve,reject)=>{
      neural.onmessage=async event=>{
        const m=event.data;
        if(m.type==='error'){neuralBusy=false;const error=Error(m.message);reject(error);fail(error);return;}
        if(m.type==='ready'){stage('sim','done',`${m.neurons.toLocaleString()} neurons · ${m.edges.toLocaleString()} connections`);resolve();return;}
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
        if(m.type==='ready'){stage('body','done',m.backend);bodyBackend=m.backend;flyLabel();resolve();return;}
        if(m.type==='air'){airAvailable=m.available!==false;airMode=airAvailable&&!!m.on;airMessage=m.message??'';flyLabel();status();return;}
        bodyBusy=false;if(m.epoch!==epoch)return;
        body.applyPhysicalState(m);bodySteps++;app.dataset.bodySteps=String(bodySteps);
        // The twitch can tip the fly onto its back, where it slides instead of walking. Right it after a
        // second of being inverted: the body's own z axis pointing below the floor.
        const [qw,qx,qy,qz]=[m.qpos[3],m.qpos[4],m.qpos[5],m.qpos[6]];
        const upright=1-2*(qx*qx+qy*qy)>0||Math.abs(qw)+Math.abs(qz)===0;
        if(upright||airMode)upsideDownSince=0;
        else if(!upsideDownSince)upsideDownSince=performance.now();
        else if(performance.now()-upsideDownSince>1000){upsideDownSince=0;rescues++;app.dataset.rescues=String(rescues);physics.postMessage({type:'reset',epoch});}
        const state=body.getState();if(state)app.dataset.fly=JSON.stringify({time:state.time,position:state.position.map(v=>Number(v.toFixed(3)))});
      };
      physics.onerror=e=>{const error=Error(e.message);reject(error);fail(error);};
    });
    neural.postMessage({type:'init',base:directory,controlWidth,parameters});
    physics.postMessage({type:'init',runtime:new URL(`${base}body/runtime.js`,location.href).href,assets:`${base}body/assets`,timestep:.00025,native:'http://127.0.0.1:8769'});
    stage('cache','load','starting');stage('sim','load','connectome and observation operator');
    stage('body','load','compiling MuJoCo');stage('world','wait','waiting for the brain');
    await Promise.all([
      brain.loadLightCache(directory,(loaded,total)=>stage('cache','load',`block ${loaded} of ${total} · every branch of all ${meta.neuronCount.toLocaleString()} neurons`))
        .then(()=>stage('cache','done',`${meta.edgeCount.toLocaleString()} branches integrated`)),
      body.load(()=>{}),neuralReady,bodyReady]);
    // The brain panel is drawn from the cable cache, which already integrates
    // every released branch of all 139,255 neurons, so the 7 GB of indexed
    // skeletons is uploaded only once the camera leaves that fixed front view.
    brain.onGeometryError=fail;
    brain.loadOnDemand(directory,(loaded,total)=>{
      app.dataset.anatomyLoaded=String(loaded);
      $('anatomy-status').textContent=loaded<total?`Uploading released skeletons: ${loaded.toLocaleString()} / ${total.toLocaleString()} neurons`:'';
    });
    body.setView(BODY_VIEWS[bodyViewIndex]);
    stage('world','load',world.url);
    await world.connect();
    if(!await source.load(world))throw Error('The world model could not become the stimulus.');
    world.attachKeyboard();
    neural.postMessage({type:'config',synapses:true,muted:false});
    document.getElementById('overlay')?.remove();
    showScreen();
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
  else if(key==='T'){screenIndex=(screenIndex+1)%SCREENS.length;showScreen();}
  else if(key==='G'){airMode=!airMode;physics.postMessage({type:'air',on:airMode,epoch});flyLabel();status();}
  else if(key==='K'||key==='L'){const step=key==='L'?1:-1;world.targetFps=world.targetFps?Math.min(30,Math.max(1,world.targetFps+step)):5;status();}
  else if(key==='-'||key==='='){brightness=Math.min(2,Math.max(.1,brightness+(key==='='?.1:-.1)));applyLook();brain.render(performance.now(),0,false);}
  else if(key===','||key==='.'){const step=key==='.'?1.25:.8;bodyGain=Math.min(60,Math.max(1,bodyGain*step));twitchGain=Math.min(24,Math.max(0,twitchGain*step));status();}
  else if(key==='R'){void resetAll();}   // the world client takes its own new prompt on the same key
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&ready&&playing)setPlaying(false);});
window.addEventListener('pagehide',event=>{if(event.persisted)return;neural?.terminate();physics?.terminate();world?.dispose();source?.dispose();brain?.dispose();body?.dispose();});
void start();
