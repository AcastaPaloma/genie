/** Browser client for experiments/play_server.py: the trained Genie world model (frozen tokenizer, latent
 * action model and dynamics model, run through the cached fast paths in experiments/fast_infer.py)
 * generating one DOOM frame per latent action. Each frame arrives as a JPEG and is drawn on a canvas; the client
 * is a LiveSource FrameProducer, so the existing capture -> controller path receives it like a decoded video.
 * This module has no access to neural geometry or activity; it only supplies stimulus pixels and reports the
 * server's own timing. */
export interface WorldStep{frame:number;ms:number;msDynamics:number|null;msDecoder:number|null;context:number;rebuilt:boolean;pan:number;code:number;label:string;roundTripMs:number;steps:number}
export interface WorldModelOptions{
  /** Base URL of the play server, without a trailing slash. The dev server proxies `${base}world` to it. */
  url:string;
  /** Latent code per key combo (W, S, A, D, WA, WD, SA, SD). Overrides the map the server reports from GET /keys. */
  keys?:Record<string,number>;
  /** Combo stepped while nothing is held (for example 'W'); the default null holds the last frame instead. */
  idleKey?:string|null;
  /** MaskGIT passes per generated frame. */
  steps?:number;
  /** Step requests kept in flight. Above 1 the round trip stops gating the frame rate, at the cost of
   * committing that many frames of game time before a key change can take effect. */
  inflight?:number;
  /** Generated frames per second to aim for; null runs as fast as the server allows. */
  targetFps?:number|null;
  width?:number;
  height?:number;
}
/** Latent code per key combo, measured on the deployed pair by holding each code for 5 steps from 4 prompts
 * (2026-09-17). Mean camera turn per step, positive = the camera turns right, with its spread across scenes:
 *   0 +11.3 (sd 1.3)   6 +10.6 (2.6)   7 +3.7 (9.9)   4 +1.2 (8.4)
 *   1  -1.3 (8.9)      2  -4.7 (8.2)   3 -4.9 (9.0)   5 -7.5 (6.2)
 * The plain keys carry the most consistent codes: D = 0, A = 5. Every code also walks the player forward
 * (1.02-1.05x zoom per step), so there is no standing-still code, and none fires the weapon: the weapon band
 * moves by at most 0.02 for any code, because ATTACK is one rare action id that the 8-code alphabet never
 * spent a slot on. Pass ?keys= to override; the server's own map is W=1,S=4,A=6,D=3,WA=7,WD=5,SA=0,SD=2. */
export const DEFAULT_KEYS:Record<string,number>={W:4,S:1,A:5,D:0,WA:3,WD:7,SA:2,SD:6};
const ARROWS:Record<string,string>={ArrowUp:'W',ArrowDown:'S',ArrowLeft:'A',ArrowRight:'D'};
const COMBOS=['W','S','A','D','WA','WD','SA','SD'];

/** Parse the server's --keys syntax ("W=1,S=4,..."): null when absent, an error when incomplete. */
export function parseKeys(text:string|null|undefined):Record<string,number>|null{
  if(!text)return null;
  const keys:Record<string,number>={};
  for(const part of text.split(',')){
    const [name,code]=part.split('='),value=Number(code);
    if(!name||!Number.isInteger(value)||value<0)throw Error(`Bad key mapping "${part}".`);
    keys[name.trim().toUpperCase()]=value;
  }
  const missing=COMBOS.filter(combo=>!(combo in keys));
  if(missing.length)throw Error(`Key map must include ${missing.join(', ')}.`);
  return keys;
}
export function describeKeys(keys:Record<string,number>){return COMBOS.filter(combo=>combo in keys).map(combo=>`${combo}=${keys[combo]}`).join('  ');}
const editable=(target:EventTarget|null)=>target instanceof HTMLElement&&(target.isContentEditable||/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));

export class WorldModelClient{
  readonly canvas=document.createElement('canvas');
  keys:Record<string,number>;
  defaultSteps:number;
  steps:number;
  /** Latent code count reported by the server, once connected. */
  codes:number|null=null;
  idleKey:string|null;
  frames=0;
  fps:number|null=null;
  /** Requests currently allowed in flight; adapts to the measured wire time unless ?inflight= fixes it. */
  inflight:number;
  private readonly fixedDepth:number|null;
  /** Frames per second to aim for; null is as fast as the server allows. */
  targetFps:number|null;
  private nextSlot=0;
  clip:number|null=null;
  prompt=0;
  lastStep:WorldStep|null=null;
  onStep:(step:WorldStep)=>void=()=>{};
  onStatus:(text:string)=>void=()=>{};
  onError:(error:Error)=>void=()=>{};
  private context:CanvasRenderingContext2D;
  private listeners=new Set<(time:number)=>void>();
  private painted:Promise<void>=Promise.resolve();
  private presented:number[]=[];
  private startedAt=performance.now();
  private held=new Set<string>();
  private running=false;
  private looping=false;
  private disposed=false;
  constructor(readonly options:WorldModelOptions){
    this.keys={...(options.keys??DEFAULT_KEYS)};
    this.defaultSteps=this.steps=Math.max(1,Math.min(50,Math.round(options.steps??3)));
    this.idleKey=options.idleKey??null;
    // Depth is measured, not guessed: the page reaches the server through a proxy, so the URL says nothing
    // about whether it is on this machine. Each step reports the server's own time, and the depth needed to
    // cover the wire is round trip / server time: 1 for a local server, about 4 through an ssh tunnel.
    // Deeper than 4 only buys lag (measured: depth 6 gave 2.0 frames/s and 3 s of committed lag), since the server serializes behind one lock.
    this.fixedDepth=options.inflight===undefined?null:Math.min(4,Math.max(1,Math.round(options.inflight)));
    this.targetFps=options.targetFps??null;
    this.inflight=this.fixedDepth??1;
    this.canvas.width=options.width??320;this.canvas.height=options.height??180;
    this.context=this.canvas.getContext('2d')!;this.context.imageSmoothingEnabled=false;
    this.context.fillStyle='#000';this.context.fillRect(0,0,this.canvas.width,this.canvas.height);
  }
  get url(){return this.options.url.replace(/\/+$/,'');}
  /** Seconds since the client was created; the source time reported with each generated frame. */
  get time(){return (performance.now()-this.startedAt)/1000;}
  /** FrameProducer: listeners run after every new frame is drawn on `canvas`. */
  subscribe(listener:(time:number)=>void){this.listeners.add(listener);return ()=>{this.listeners.delete(listener);};}
  /** Reset the server to a fresh prompt clip (random unless given) and show its last prompt frame. */
  async reset(clip?:number){
    this.held.clear();
    const response=await this.request(`/reset${clip===undefined?'':`?clip=${clip}`}`);
    const info=await response.json() as {clip:number;prompt:number};
    this.clip=info.clip;this.prompt=info.prompt;this.fps=null;
    await this.show(await this.request(`/frame?t=${Date.now()}`));
    this.onStatus(`Prompt: validation clip ${info.clip}, ${info.prompt} frames. Hold A or D to turn; ${this.steps} MaskGIT passes.`);
    return info;
  }
  /** Read the server's default passes and code count (GET /keys, absent on older servers); the key map stays local. */
  async connect(){
    let response:Response|null=null;
    try{response=await fetch(`${this.url}/keys`,{cache:'no-store'});}catch(error){throw Error(`No world model at ${this.url} (${error instanceof Error?error.message:String(error)}). Start experiments/play_server.py.`);}
    if(response.ok){
      const info=await response.json() as {keys?:Record<string,number>;steps?:number;codes?:number};
      if(this.options.steps===undefined&&Number.isInteger(info.steps))this.defaultSteps=this.steps=Math.max(1,Math.min(50,info.steps!));
      if(Number.isInteger(info.codes))this.codes=info.codes!;
    }
    return this.reset();
  }
  /** A tunnelled server drops the odd connection, so a step is retried briefly before the game gives up. */
  private async request(path:string,attempts=3){
    let last='';
    for(let attempt=0;attempt<attempts;attempt++){
      if(this.disposed)throw Error('The world model client was disposed.');
      try{
        const response=await fetch(this.url+path,{cache:'no-store'});
        if(response.ok)return response;
        last=`${response.status}: ${(await response.text()).slice(0,200)}`;
        if(response.status<500)break;                       // a refusal will not fix itself
      }catch(error){last=error instanceof Error?error.message:String(error);}
      await new Promise(resolve=>setTimeout(resolve,120*(attempt+1)));
    }
    throw Error(`World model ${path.split('?')[0]} failed at ${this.url} (${last}). Start experiments/play_server.py, or check the tunnel if it is remote.`);
  }
  private async show(response:Response){
    const bitmap=await createImageBitmap(await response.blob());
    if(this.disposed){bitmap.close();return;}
    this.context.drawImage(bitmap,0,0,this.canvas.width,this.canvas.height);bitmap.close();
    this.frames++;
    // Frames per second over a two-second window. Gaps between individual presentations are useless here:
    // with requests overlapping, frames arrive in bursts of `inflight` and the instantaneous rate swings wildly.
    const now=performance.now();
    this.presented.push(now);
    while(this.presented.length&&now-this.presented[0]>2000)this.presented.shift();
    const span=now-this.presented[0];
    this.fps=this.presented.length>1&&span>0?(this.presented.length-1)*1000/span:null;
    const time=this.time;for(const listener of this.listeners)listener(time);
  }
  /** The latent action for the held keys: a raw digit wins, then the forward/back + turn combos. */
  combo():{label:string;code:number}|null{
    for(const key of this.held)if(/^[0-7]$/.test(key))return{label:`code ${key}`,code:Number(key)};
    const straight=this.held.has('W')?'W':this.held.has('S')?'S':'',turn=this.held.has('A')?'A':this.held.has('D')?'D':'',combo=straight+turn;
    return combo&&combo in this.keys?{label:combo,code:this.keys[combo]}:null;
  }
  private idle(){return this.idleKey&&this.idleKey in this.keys?{label:`idle (${this.idleKey})`,code:this.keys[this.idleKey]}:null;}
  private key(event:KeyboardEvent){return event.key.length===1?event.key.toUpperCase():(ARROWS[event.key]??'');}
  private keydown=(event:KeyboardEvent)=>{
    if(event.repeat||event.metaKey||event.ctrlKey||event.altKey||editable(event.target))return;
    const key=this.key(event);
    if((key&&'WASD'.includes(key))||/^[0-7]$/.test(key)){this.held.add(key);event.preventDefault();}
    else if(key==='R'){this.onStatus('New prompt…');void this.reset().catch(error=>this.fail(error));}
    else if(key==='Q'){this.steps=this.steps===25?this.defaultSteps:25;this.onStatus(`MaskGIT passes: ${this.steps}`);}
    else if(key==='['||key===']'){this.steps=Math.max(1,Math.min(50,this.steps+(key===']'?1:-1)));this.onStatus(`MaskGIT passes: ${this.steps}`);}
  };
  private keyup=(event:KeyboardEvent)=>{this.held.delete(this.key(event));};
  private blur=()=>{this.held.clear();};
  attachKeyboard(){document.addEventListener('keydown',this.keydown);document.addEventListener('keyup',this.keyup);window.addEventListener('blur',this.blur);}
  detachKeyboard(){document.removeEventListener('keydown',this.keydown);document.removeEventListener('keyup',this.keyup);window.removeEventListener('blur',this.blur);this.held.clear();}
  get heldKeys(){return [...this.held];}
  get isRunning(){return this.running;}
  /** While running, the world steps continuously: the held combo, otherwise the idle key (if any). */
  setRunning(value:boolean){this.running=value&&!this.disposed;if(this.running)void this.loop();}
  private async loop(){
    if(this.looping)return;this.looping=true;
    // Requests are issued up to `inflight` deep and drawn strictly in issue order, so the picture never jumps
    // backwards. The server serializes them behind its own lock, so they queue there rather than racing.
    const queue:Promise<void>[]=[];
    try{
      while(this.running&&!this.disposed){
        const action=this.combo()??this.idle();
        if(!action){
          if(queue.length)await queue.shift();
          else await new Promise(resolve=>setTimeout(resolve,30));
          continue;
        }
        if(this.targetFps&&this.targetFps>0){
          const interval=1000/this.targetFps,now=performance.now();
          if(this.nextSlot>now)await new Promise(resolve=>setTimeout(resolve,this.nextSlot-now));
          this.nextSlot=Math.max(performance.now(),this.nextSlot)+interval;
        }
        queue.push(this.step(action).then(()=>{}));
        while(queue.length>=this.inflight)await queue.shift();
      }
      while(queue.length)await queue.shift();
    }catch(error){this.fail(error);}
    finally{this.looping=false;}
  }
  private fail(error:unknown){this.running=false;this.onError(error instanceof Error?error:Error(String(error)));}
  /** Generate one frame for a latent action. The server serializes concurrent steps behind one lock. */
  async step(action:{label:string;code:number}){
    const started=performance.now();
    const ahead=this.painted;
    const response=await this.request(`/step?code=${action.code}&steps=${this.steps}`);
    // Wait for every earlier request of this batch to have been drawn, so frames appear in the order the
    // server generated them even when several were in flight.
    this.painted=ahead.then(async()=>{await this.show(response);});
    await this.painted;
    const header=(name:string)=>{const value=response.headers.get(name);return value===null||value===''?null:Number(value);};
    const roundTripMs=performance.now()-started;
    const step:WorldStep={frame:header('X-Frame')??this.frames,ms:header('X-Ms')??Math.round(roundTripMs),msDynamics:header('X-Ms-Dynamics'),msDecoder:header('X-Ms-Decoder'),context:header('X-Context')??0,rebuilt:header('X-Rebuilt')===1,pan:header('X-Pan')??0,code:header('X-Code')??action.code,label:action.label,roundTripMs,steps:this.steps};
    if(this.fixedDepth===null&&step.ms>0){
      // At a capped rate the queue only has to cover the wire within one frame interval, not saturate the server.
      const interval=this.targetFps&&this.targetFps>0?1000/this.targetFps:Math.max(1,step.ms);
      const needed=Math.min(4,Math.max(1,Math.ceil(roundTripMs/interval)));
      // Move one step at a time so a single slow response cannot swing the pipeline.
      this.inflight+=Math.sign(needed-this.inflight);
    }
    this.lastStep=step;this.onStep(step);this.onStatus(this.describe(step));
    return step;
  }
  describe(step:WorldStep){
    const server=step.msDynamics===null?`server ${step.ms} ms`:`server ${step.ms} ms = dynamics ${step.msDynamics} + decoder ${step.msDecoder}`;
    const interval=this.fps&&this.fps>0?1000/this.fps:step.roundTripMs;
    const capped=this.targetFps?` · capped at ${this.targetFps} frames/s`:'';
    const pipeline=this.inflight>1?` · ${this.inflight} in flight (~${Math.round(this.inflight*interval)} ms committed before a key change lands)`:'';
    return `${step.label} → code ${step.code} · frame ${step.frame} · ${(this.fps??0).toFixed(1)} generated frames/s (${step.roundTripMs.toFixed(0)} ms round trip, ${server})${capped}${pipeline} · ${step.steps} MaskGIT passes · context ${step.context} frames${step.rebuilt?' (cache restarted)':''} · turn ${step.pan>0?'+':''}${step.pan} px${step.pan>0?' right':step.pan<0?' left':''}`;
  }
  dispose(){this.disposed=true;this.running=false;this.detachKeyboard();this.listeners.clear();}
}
