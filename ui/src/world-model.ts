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
  width?:number;
  height?:number;
}
/** Latent codes of lam_spatial_a100_pan per key: 3/5/2 pan the scene left, 6/0/7 right, 1/4 no turn. A = 3 and D = 6
 * (the opposite of play_server.py's default) because A turned the player right in play testing on 2026-09-15.
 * Every code advances the player, since the recorded agent nearly always moved forward. */
export const DEFAULT_KEYS:Record<string,number>={W:1,S:4,A:3,D:6,WA:5,WD:7,SA:2,SD:0};
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
  clip:number|null=null;
  prompt=0;
  lastStep:WorldStep|null=null;
  onStep:(step:WorldStep)=>void=()=>{};
  onStatus:(text:string)=>void=()=>{};
  onError:(error:Error)=>void=()=>{};
  private context:CanvasRenderingContext2D;
  private listeners=new Set<(time:number)=>void>();
  private startedAt=performance.now();
  private held=new Set<string>();
  private running=false;
  private looping=false;
  private disposed=false;
  constructor(readonly options:WorldModelOptions){
    this.keys={...(options.keys??DEFAULT_KEYS)};
    this.defaultSteps=this.steps=Math.max(1,Math.min(50,Math.round(options.steps??3)));
    this.idleKey=options.idleKey??null;
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
  private async request(path:string){
    let response:Response;
    try{response=await fetch(this.url+path,{cache:'no-store'});}
    catch(error){throw Error(`No world model at ${this.url} (${error instanceof Error?error.message:String(error)}). Start experiments/play_server.py.`);}
    if(!response.ok)throw Error(`World model ${path.split('?')[0]} failed (${response.status}): ${(await response.text()).slice(0,200)}`);
    return response;
  }
  private async show(response:Response){
    const bitmap=await createImageBitmap(await response.blob());
    if(this.disposed){bitmap.close();return;}
    this.context.drawImage(bitmap,0,0,this.canvas.width,this.canvas.height);bitmap.close();
    this.frames++;const time=this.time;for(const listener of this.listeners)listener(time);
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
    try{
      while(this.running&&!this.disposed){
        const action=this.combo()??this.idle();
        if(!action){await new Promise(resolve=>setTimeout(resolve,30));continue;}
        await this.step(action);
      }
    }catch(error){this.fail(error);}
    finally{this.looping=false;}
  }
  private fail(error:unknown){this.running=false;this.onError(error instanceof Error?error:Error(String(error)));}
  /** Generate one frame for a latent action. The server serializes concurrent steps behind one lock. */
  async step(action:{label:string;code:number}){
    const started=performance.now();
    const response=await this.request(`/step?code=${action.code}&steps=${this.steps}`);
    await this.show(response);
    const header=(name:string)=>{const value=response.headers.get(name);return value===null||value===''?null:Number(value);};
    const roundTripMs=performance.now()-started;
    this.fps=this.fps===null?1000/roundTripMs:.8*this.fps+.2*1000/roundTripMs;
    const step:WorldStep={frame:header('X-Frame')??this.frames,ms:header('X-Ms')??Math.round(roundTripMs),msDynamics:header('X-Ms-Dynamics'),msDecoder:header('X-Ms-Decoder'),context:header('X-Context')??0,rebuilt:header('X-Rebuilt')===1,pan:header('X-Pan')??0,code:header('X-Code')??action.code,label:action.label,roundTripMs,steps:this.steps};
    this.lastStep=step;this.onStep(step);this.onStatus(this.describe(step));
    return step;
  }
  describe(step:WorldStep){
    const server=step.msDynamics===null?`server ${step.ms} ms`:`server ${step.ms} ms = dynamics ${step.msDynamics} + decoder ${step.msDecoder}`;
    return `${step.label} → code ${step.code} · frame ${step.frame} · ${(this.fps??0).toFixed(1)} generated frames/s (${step.roundTripMs.toFixed(0)} ms round trip, ${server}) · ${step.steps} MaskGIT passes · context ${step.context} frames${step.rebuilt?' (cache restarted)':''} · pan ${step.pan>0?'+':''}${step.pan} px`;
  }
  dispose(){this.disposed=true;this.running=false;this.detachKeyboard();this.listeners.clear();}
}
