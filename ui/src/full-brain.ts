import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export interface FullBrainChunk {
  file:string;
  sha256:string;
  vertices:number;
  edges:number;
  isolatedVertices:number;
  neuronCount:number;
  bounds:[number[],number[]];
}
export interface FullBrainMetadata {
  version:1;
  materialization:783;
  expectedNeurons:number;
  neuronCount:number;
  vertexCount:number;
  edgeCount:number;
  isolatedVertexCount:number;
  complete:boolean;
  missingRootIds:string[];
  bounds:[number[],number[]];
  chunks:FullBrainChunk[];
  neurons:{rootId:string;modelIndex:number;vertices:number;edges:number}[];
  source:{url:string;sha256:string;md5:string};
}

/** False-colour palettes for the spike light. Polynomial fits of the matplotlib
 * colormaps (Zucker); 0 keeps the original grayscale. The input is the same
 * density-normalized activity value in both views; colour never comes from the video. */
export const PALETTES=['grayscale','heat','viridis','plasma','magma','source'] as const;
/** 'source' keeps the spike light as brightness and takes only the hue from the frame the controller was given. */
const SOURCE=5;
export type Palette=typeof PALETTES[number];
const PALETTE_GLSL=`
vec3 poly(float t,vec3 c0,vec3 c1,vec3 c2,vec3 c3,vec3 c4,vec3 c5,vec3 c6){return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));}
vec3 shade(float v,int palette){
  v=clamp(v,0.0,1.0);
  if(palette==1)return clamp(poly(v,vec3(0.0002189403691192265,0.001651004631001012,-0.01948089843709184),vec3(0.1065134194856116,0.5639564367884091,3.932712388889277),vec3(11.60249308247187,-3.972853965665698,-15.9423941062914),vec3(-41.70399613139459,17.43639888205313,44.35414519872813),vec3(77.162935699427,-33.40235894210092,-81.80730925738993),vec3(-71.31942824499214,32.62606426397723,73.20951985803202),vec3(25.13112622477341,-12.24266895238567,-23.07032500287172)),0.0,1.0);
  if(palette==2)return clamp(poly(v,vec3(0.2777273272234177,0.005407344544966578,0.3340998053353061),vec3(0.1050930431085774,1.404613529898575,1.384590162594685),vec3(-0.3308618287255563,0.214847559468213,0.09509516302823659),vec3(-4.634230498983486,-5.799100973351585,-19.33244095627987),vec3(6.228269936347081,14.17993336680509,56.69055260068105),vec3(4.776384997670288,-13.74514537774601,-65.35303263337234),vec3(-5.435455855934631,4.645852612178535,26.3124352495832)),0.0,1.0);
  if(palette==3)return clamp(poly(v,vec3(0.05873234392399702,0.02333670892565664,0.5433401826748754),vec3(2.176514634195958,0.2383834171260182,0.7539604599784036),vec3(-2.689460476458034,-7.455851135738909,3.110799939717086),vec3(6.130348345893603,42.3461881477227,-28.51885465332158),vec3(-11.10743619062271,-82.66631109428045,60.13984767418263),vec3(10.02306557647065,71.41361770095349,-54.07218655560067),vec3(-3.658713842777788,-22.93153465461149,18.19190277267161)),0.0,1.0);
  if(palette==4)return clamp(poly(v,vec3(-0.002136485053939582,-0.000749655052795221,-0.005386127855323933),vec3(0.2516605407371642,0.6775232436837668,2.494026599312351),vec3(8.353717279216625,-3.577719514958484,0.3144679030132573),vec3(-27.66873308576866,14.26473078096533,-13.68929327010588),vec3(52.17613981234068,-27.94360607168351,12.94416944238394),vec3(-50.76852536473588,29.04658282127291,4.23415299384598),vec3(18.65570506591883,-11.48977351997711,-5.601961508734096)),0.0,1.0);
  return vec3(v);
}`;

/** Chunk transport and integrity run in workers because the acquisition loop is
 * main-thread bound, not I/O bound: a one-chunk main-thread prefetch saved 3%,
 * while a pool scales. Measured fetch + SHA-256 for the complete 270-chunk
 * anatomy: 12.1 s on the main thread, 7.3 s with 2 workers, 5.3 s with 4 and
 * 3.8 s with 8. GPU upload stays on the main thread and is unaffected. */
const LOADER_WORKERS=2;

/** Fetches and verifies released chunks ahead of the uploader. It never decides
 * what is complete: every chunk is still re-checked against its manifest entry
 * on the main thread before any buffer view is built from it. */
class ChunkLoaderPool{
  private workers:Worker[]=[];
  private waiting=new Map<number,{resolve:(raw:ArrayBuffer)=>void;reject:(error:Error)=>void}>();
  private turn=0;
  private failure:Error|null=null;
  constructor(size:number){
    for(let i=0;i<size;i++){
      const worker=new Worker(new URL('./skeleton-loader.worker.ts',import.meta.url),{type:'module'});
      worker.onmessage=event=>{
        const {id,raw,message}=event.data;
        const pending=this.waiting.get(id);if(!pending)return;this.waiting.delete(id);
        if(message)pending.reject(Error(message));else pending.resolve(raw);
      };
      worker.onerror=event=>this.fail(Error(event.message||'A skeleton chunk loader stopped.'));
      this.workers.push(worker);
    }
  }
  private fail(error:Error){
    this.failure=error;
    for(const pending of this.waiting.values())pending.reject(error);
    this.waiting.clear();
  }
  request(id:number,directory:string,chunk:FullBrainChunk):Promise<ArrayBuffer>{
    if(this.failure)return Promise.reject(this.failure);
    const worker=this.workers[this.turn++%this.workers.length];
    const pending=new Promise<ArrayBuffer>((resolve,reject)=>this.waiting.set(id,{resolve,reject}));
    worker.postMessage({id,dir:directory,file:chunk.file,vertices:chunk.vertices,edges:chunk.edges,isolatedVertices:chunk.isolatedVertices,sha256:chunk.sha256});
    // Each chunk is awaited in order below; mark it handled so an early failure
    // surfaces at its own await instead of as an unhandled rejection.
    pending.catch(()=>{});
    return pending;
  }
  dispose(){this.workers.forEach(worker=>worker.terminate());this.workers=[];this.waiting.clear();}
}

/** Display processing of the simulated light: a monotone level stretch to the frame's own range, gamma, an
 * unsharp local-contrast term, and a saturation factor. It reweights light the simulation produced; it adds
 * no video content and no per-branch colour. Unsupported texels carry -1 and are skipped, never averaged. */
const DISPLAY_GLSL=`
uniform vec2 levels;uniform float gamma;uniform float sharpen;uniform float saturation;uniform vec2 displayTexel;uniform float floorLevel;uniform vec2 pixelCells;uniform vec2 pixelOrigin;uniform float brightness;
float tone(float v){
  float x=levels.y>levels.x?clamp((v-levels.x)/(levels.y-levels.x),0.0,1.0):clamp(v,0.0,1.0);
  if(gamma!=1.0)x=pow(x,gamma);
  x=clamp(x*brightness,0.0,1.0);
  // Lift the black point off zero: a supported texel stays visible against the background even when the
  // stretch puts it at the bottom of the range, so the outer arbors never drop out of the silhouette.
  return floorLevel+(1.0-floorLevel)*x;
}
vec3 saturate3(vec3 c){float l=dot(c,vec3(0.2126,0.7152,0.0722));return clamp(l+(c-l)*saturation,0.0,1.0);}`;

/** Every released skeleton vertex and parent edge is uploaded unchanged apart
 * from a shared coordinate transform. This class has no source-image argument,
 * source texture, observation-site selection or neural-output fallback. A
 * front-view cable cache is evaluated exclusively from this neuron's activity.
 */
export class FullBrainView {
  readonly renderer:THREE.WebGLRenderer;
  readonly camera=new THREE.OrthographicCamera(-300,300,225,-225,.1,10000);
  readonly controls:OrbitControls;
  readonly activityTexture:THREE.DataTexture;
  private scene=new THREE.Scene();
  private screen=new THREE.Scene();
  private screenCamera=new THREE.Camera();
  private accumulation:THREE.WebGLRenderTarget;
  private material:THREE.ShaderMaterial;
  private inspectionMaterial:THREE.ShaderMaterial;
  private composite:THREE.ShaderMaterial;
  private screenGeometry=new THREE.PlaneGeometry(2,2);
  private geometries:THREE.BufferGeometry[]=[];
  private resizeObserver:ResizeObserver;
  private center=new THREE.Vector3();
  private radius=1000;
  private dirty=true;
  private lost=false;
  private loaded=false;
  private identityView=false;
  private frameSubmittedAt=0;
  private anatomyDrawCalls=0;
  private completedAt=0;
  private referenceFootprint:number[]|null=null;
  private lightWorker:Worker|null=null;
  private lightPending:{resolve:()=>void;reject:(error:Error)=>void}|null=null;
  private cachedScene=new THREE.Scene();
  private cachedTexture:THREE.DataTexture|null=null;
  private cachedMaterial:THREE.ShaderMaterial|null=null;
  private cachedPlane:THREE.PlaneGeometry|null=null;
  private cacheInfo:{resolution:number[];entries:number;branches:number;computeMs:number}|null=null;
  private drawMode='complete indexed geometry';
  private pending:Promise<void>|null=null;
  private demand:{directory:string;onProgress:(loaded:number,total:number)=>void}|null=null;
  private regionTexture:THREE.DataTexture;
  private regionUniform:{value:THREE.DataTexture};
  private regionMask:Float32Array|null=null;
  private regionName='all';
  private regionCounts:{neurons:number;vertices:number;edges:number}|null=null;
  private maskedActivity:Float32Array|null=null;
  private paletteUniform={value:0};
  private display={levels:{value:new THREE.Vector2(0,0)},gamma:{value:1},sharpen:{value:0},saturation:{value:1},displayTexel:{value:new THREE.Vector2(1,1)},floorLevel:{value:0},pixelCells:{value:new THREE.Vector2(0,0)},pixelOrigin:{value:new THREE.Vector2(0,0)},brightness:{value:1}};
  private autoLevels=false;
  private levelRange:[number,number]|null=null;
  private cacheGeometry:{center:number[];footprint:number[];resolution:number[]}|null=null;
  private palette:Palette='grayscale';
  private sourceTexture:THREE.DataTexture;
  private sourceUniform:{value:THREE.DataTexture};
  private referenceCenterUniform={value:new THREE.Vector2()};
  private referenceFootprintUniform={value:new THREE.Vector2(1,1)};
  readonly counts={neurons:0,vertices:0,edges:0,isolatedVertices:0,chunks:0};
  onContextLost:()=>void=()=>{};
  /** Reports a failure of the deferred geometry upload, which no caller awaits. */
  onGeometryError:(error:Error)=>void=()=>{};

  constructor(private element:HTMLElement,readonly meta:FullBrainMetadata){
    if(!meta.complete||meta.missingRootIds.length||meta.neuronCount!==meta.expectedNeurons)
      throw Error('The full-brain anatomy is incomplete. Missing neurons must be resolved before playback.');
    this.renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(devicePixelRatio);
    this.renderer.setClearColor(0x101619);
    this.renderer.domElement.setAttribute('role','img');
    this.renderer.domElement.setAttribute('aria-label','Complete FlyWire brain skeletons. Every branch follows its owning neuron. Drag to orbit, scroll to zoom, or use the Brain viewpoint menu.');
    this.renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();this.lost=true;this.onContextLost();});
    element.append(this.renderer.domElement);
    const box=new THREE.Box3(new THREE.Vector3().fromArray(meta.bounds[0]),new THREE.Vector3().fromArray(meta.bounds[1]));
    box.getCenter(this.center);this.radius=box.getSize(new THREE.Vector3()).length()*1.5;
    this.camera.position.copy(this.center).add(new THREE.Vector3(0,0,this.radius));
    this.camera.far=this.radius*5;this.camera.lookAt(this.center);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.target.copy(this.center);this.controls.enableDamping=false;
    this.controls.enablePan=true;this.controls.minZoom=.25;this.controls.maxZoom=20;
    this.controls.addEventListener('change',()=>{this.dirty=true;if(!this.atCachedFrontView())this.beginDemandLoad();});
    const width=2048,height=Math.ceil(meta.neuronCount/width);
    this.activityTexture=new THREE.DataTexture(new Float32Array(width*height),width,height,THREE.RedFormat,THREE.FloatType);
    this.activityTexture.needsUpdate=true;
    // 1 for every neuron the current region draws. Every neuron is still loaded,
    // checksummed and simulated; this only gates rasterization.
    this.regionTexture=new THREE.DataTexture(new Float32Array(width*height).fill(1),width,height,THREE.RedFormat,THREE.FloatType);
    this.regionTexture.needsUpdate=true;this.regionUniform={value:this.regionTexture};
    this.sourceTexture=new THREE.DataTexture(new Uint8Array([255,255,255,255]),1,1,THREE.RGBAFormat,THREE.UnsignedByteType);this.sourceTexture.needsUpdate=true;this.sourceUniform={value:this.sourceTexture};
    this.referenceCenterUniform.value.set(this.center.x,this.center.y);
    // Accumulates activity*chroma in RGB and the branch count in A. chroma is 1 except in the
    // source-colour mode, where it is the unit-luminance colour of the controller's own target frame.
    this.material=new THREE.ShaderMaterial({
      uniforms:{activityTexture:{value:this.activityTexture},activitySize:{value:new THREE.Vector2(width,height)},regionTexture:this.regionUniform,sourceTexture:this.sourceUniform,referenceCenter:this.referenceCenterUniform,referenceFootprint:this.referenceFootprintUniform,palette:this.paletteUniform},
      vertexShader:`attribute float neuronIndex; uniform sampler2D activityTexture; uniform sampler2D regionTexture; uniform vec2 activitySize; uniform sampler2D sourceTexture; uniform vec2 referenceCenter; uniform vec2 referenceFootprint; uniform int palette; varying float activity; varying vec3 chroma;
        void main(){vec2 slot=(vec2(mod(neuronIndex,activitySize.x),floor(neuronIndex/activitySize.x))+0.5)/activitySize;
          // Withheld neurons leave clip space, so they add neither light nor branch
          // count; masking activity alone would still dilute the per-pixel average.
          if(texture2D(regionTexture,slot).r<0.5){gl_Position=vec4(0.0,0.0,2.0,1.0);return;}
          activity=texture2D(activityTexture,slot).r;
          chroma=vec3(1.0);
          if(palette==${SOURCE}){vec2 uv=(position.xy-referenceCenter)/referenceFootprint+0.5;vec3 src=texture2D(sourceTexture,vec2(uv.x,1.0-uv.y)).rgb;float lum=dot(src,vec3(0.2126,0.7152,0.0722));chroma=lum>0.004?src/lum:vec3(1.0);}
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=1.0;}`,
      fragmentShader:`varying float activity; varying vec3 chroma; void main(){gl_FragColor=vec4(activity*chroma,1.0);}`,
      transparent:true,blending:THREE.AdditiveBlending,depthTest:false,depthWrite:false,toneMapped:false,
    });
    // A separately labelled anatomy inspector uses identity colours and depth.
    // It is never the activity output and has no access to the video either.
    this.inspectionMaterial=new THREE.ShaderMaterial({
      uniforms:{regionTexture:this.regionUniform,activitySize:{value:new THREE.Vector2(width,height)}},
      vertexShader:`attribute float neuronIndex;uniform sampler2D regionTexture;uniform vec2 activitySize;varying vec3 identityColor;
        void main(){if(texture2D(regionTexture,(vec2(mod(neuronIndex,activitySize.x),floor(neuronIndex/activitySize.x))+0.5)/activitySize).r<0.5){gl_Position=vec4(0.0,0.0,2.0,1.0);return;}
          float hue=fract(neuronIndex*0.61803398875);
          identityColor=0.58+0.35*cos(6.2831853*(hue+vec3(0.0,0.333333,0.666667)));
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=1.0;}`,
      fragmentShader:`varying vec3 identityColor;void main(){gl_FragColor=vec4(identityColor,1.0);}`,
      depthTest:true,depthWrite:true,toneMapped:false,
    });
    this.accumulation=new THREE.WebGLRenderTarget(1,1,{type:THREE.FloatType,depthBuffer:false,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter});
    this.composite=new THREE.ShaderMaterial({
      uniforms:{light:{value:this.accumulation.texture},exposure:{value:1},palette:this.paletteUniform,...this.display},
      vertexShader:`varying vec2 screenUV; void main(){screenUV=uv;gl_Position=vec4(position.xy,0.0,1.0);}`,
      fragmentShader:`uniform sampler2D light;uniform float exposure;uniform int palette;varying vec2 screenUV;${PALETTE_GLSL}${DISPLAY_GLSL}
        void main(){vec4 sum=texture2D(light,screenUV);
          vec3 color=vec3(0.062745,0.086275,0.098039);
          if(sum.a>0.0){
            float value=exposure*dot(sum.rgb,vec3(0.2126,0.7152,0.0722))/sum.a;
            if(sharpen>0.0){float blur=0.0,weight=0.0;
              for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){
                vec4 s=texture2D(light,screenUV+vec2(float(x),float(y))*displayTexel);
                if(s.a>0.0){blur+=exposure*dot(s.rgb,vec3(0.2126,0.7152,0.0722))/s.a;weight+=1.0;}}}
              if(weight>0.0)value+=sharpen*(value-blur/weight);}
            float t=tone(value);
            color=palette==${SOURCE}?saturate3(normalize(max(sum.rgb,vec3(1e-6)))*1.732*t):saturate3(shade(t,palette));}
          gl_FragColor=vec4(color,1.0);}`,
      depthTest:false,depthWrite:false,toneMapped:false,
    });
    this.screen.add(new THREE.Mesh(this.screenGeometry,this.composite));
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(element);this.resize();
  }

  /** Idempotent: a second call joins the upload already in progress. */
  load(directory:string,onProgress:(loaded:number,total:number)=>void=()=>{}):Promise<void>{
    this.pending??=this.loadChunks(directory,onProgress);
    return this.pending;
  }

  /** Start the 3D skeletons only once the view actually needs them. The front
   * view is drawn from the cable cache, which integrates every released branch
   * of every neuron, so the indexed geometry is required only to orbit or to
   * inspect anatomy. Neither the simulation nor the lighting changes: this is
   * purely when the geometry is uploaded, never which neurons exist. */
  loadOnDemand(directory:string,onProgress:(loaded:number,total:number)=>void=()=>{}){
    this.demand={directory,onProgress};
    // Settle the camera first: until the controls' first update the world matrix
    // reads a slightly tilted direction (about 0.0028 rad here) that would look
    // like an orbit and start the full upload for no reason.
    this.controls.update();
    if(this.identityView||!this.atCachedFrontView())this.beginDemandLoad();
  }
  private beginDemandLoad(){
    if(!this.demand||this.pending)return;
    const {directory,onProgress}=this.demand;
    void this.load(directory,onProgress).then(()=>{this.dirty=true;},error=>this.onGeometryError(error instanceof Error?error:Error(String(error))));
  }
  /** The camera direction the fixed cable cache was integrated for. */
  private atCachedFrontView(){
    const direction=this.camera.getWorldDirection(new THREE.Vector3());
    return Math.abs(direction.x)<1e-5&&Math.abs(direction.y)<1e-5&&direction.z<0;
  }

  private async loadChunks(directory:string,onProgress:(loaded:number,total:number)=>void){
    const base=new URL(directory,location.href).href,chunks=this.meta.chunks;
    const lookahead=Math.max(1,Math.min(LOADER_WORKERS,chunks.length));
    const pool=new ChunkLoaderPool(lookahead);
    const inflight=new Map<number,Promise<ArrayBuffer>>();
    const request=(index:number)=>{if(index<chunks.length)inflight.set(index,pool.request(index,base,chunks[index]));};
    try{
      let neuronCount=0;
      for(let i=0;i<lookahead;i++)request(i);
      for(let i=0;i<chunks.length;i++){
        const chunk=chunks[i];
        // Keep the next chunks in flight while this one is built and uploaded.
        let raw:ArrayBuffer|null=await inflight.get(i)!;
        inflight.delete(i);request(i+lookahead);
        // The loader verified this chunk against the same manifest entry; the
        // header is re-read here so every buffer view below still comes from
        // counts checked on this thread.
        const [version,vertices,edges,isolated]=new Uint32Array(raw,0,4);
        if(version!==785||vertices!==chunk.vertices||edges!==chunk.edges||isolated!==chunk.isolatedVertices||raw.byteLength!==16+vertices*20+edges*8+isolated*4)
          throw Error(`Incomplete full-anatomy chunk: ${chunk.file}`);
        const packed=new THREE.InterleavedBuffer(new Float32Array(raw,16,vertices*5),5);
        const geometry=new THREE.BufferGeometry();
        geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(packed,3,0));
        geometry.setAttribute('neuronIndex',new THREE.InterleavedBufferAttribute(packed,1,3));
        geometry.setAttribute('radius',new THREE.InterleavedBufferAttribute(packed,1,4));
        const index=new THREE.BufferAttribute(new Uint32Array(raw,16+vertices*20,edges*2),1);
        geometry.setIndex(index);
        geometry.boundingBox=new THREE.Box3(new THREE.Vector3().fromArray(chunk.bounds[0]),new THREE.Vector3().fromArray(chunk.bounds[1]));
        geometry.boundingSphere=geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
        const uploading=new THREE.Scene();
        const lines=new THREE.LineSegments(geometry,this.material);uploading.add(lines);this.geometries.push(geometry);
        if(isolated){
          const pointsGeometry=new THREE.BufferGeometry();
          pointsGeometry.setAttribute('position',geometry.getAttribute('position'));
          pointsGeometry.setAttribute('neuronIndex',geometry.getAttribute('neuronIndex'));
          const pointIndex=new THREE.BufferAttribute(new Uint32Array(raw,16+vertices*20+edges*8,isolated),1);
          pointsGeometry.setIndex(pointIndex);pointsGeometry.boundingSphere=geometry.boundingSphere;
          uploading.add(new THREE.Points(pointsGeometry,this.material));this.geometries.push(pointsGeometry);
          pointIndex.onUpload(()=>{pointIndex.array=new Uint32Array(0);});
        }
        // Keep the immutable buffers on the GPU; release their duplicate JS-side
        // storage after upload. Context loss fails explicitly and requires reload.
        packed.onUpload(()=>{packed.array=new Float32Array(0);});
        index.onUpload(()=>{index.array=new Uint32Array(0);});
        // Upload each chunk once. Redrawing all earlier chunks during acquisition
        // would make loading quadratic in the size of the complete anatomy.
        this.renderer.setRenderTarget(this.accumulation);this.renderer.render(uploading,this.camera);this.renderer.setRenderTarget(null);await this.waitForDraw();
        for(const child of [...uploading.children])this.scene.add(child);
        raw=null;
        neuronCount+=chunk.neuronCount;
        this.counts.neurons=neuronCount;this.counts.vertices+=vertices;this.counts.edges+=edges;this.counts.isolatedVertices+=isolated;this.counts.chunks++;
        onProgress(neuronCount,this.meta.neuronCount);
        await new Promise<void>(resolve=>setTimeout(resolve,0));
      }
      if(neuronCount!==this.meta.neuronCount||this.counts.vertices!==this.meta.vertexCount||this.counts.edges!==this.meta.edgeCount||this.counts.isolatedVertices!==this.meta.isolatedVertexCount)
        throw Error('Complete anatomy counts do not match the released brain.');
      this.loaded=true;this.dirty=true;
    }finally{pool.dispose();}
  }

  /** onProgress reports cache blocks as they are verified and uploaded. The
   * worker has always reported them; nothing consumed it before. */
  async loadLightCache(directory:string,onProgress:(loaded:number,total:number)=>void=()=>{}){
    if(this.lightWorker)throw Error('Anatomical cache already loaded.');
    const worker=this.lightWorker=new Worker(new URL('./anatomical-light.worker.ts',import.meta.url),{type:'module'});
    await new Promise<void>((resolve,reject)=>{
      worker.onerror=event=>{const error=Error(event.message);reject(error);this.lightPending?.reject(error);this.lightPending=null;};
      worker.onmessage=event=>{
        const m=event.data;
        if(m.type==='error'){const error=Error(m.message);reject(error);this.lightPending?.reject(error);this.lightPending=null;return;}
        if(m.type==='progress'){onProgress(m.loaded,m.total);return;}
        if(m.type==='ready'){
          if(m.neurons!==this.meta.neuronCount||m.branches!==this.meta.edgeCount){reject(Error('Anatomical cache coverage differs.'));return;}
          this.cacheInfo={resolution:m.resolution,entries:m.entries,branches:m.branches,computeMs:0};
          this.cachedTexture=new THREE.DataTexture(new Float32Array(m.resolution[0]*m.resolution[1]).fill(-1),m.resolution[0],m.resolution[1],THREE.RedFormat,THREE.FloatType);
          this.cachedTexture.needsUpdate=true;
          this.cachedMaterial=new THREE.ShaderMaterial({uniforms:{light:{value:this.cachedTexture},exposure:this.composite.uniforms.exposure,palette:this.paletteUniform,sourceTexture:this.sourceUniform,...this.display},
            vertexShader:`varying vec2 imageUV;void main(){imageUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
            fragmentShader:`uniform sampler2D light;uniform float exposure;uniform int palette;uniform sampler2D sourceTexture;varying vec2 imageUV;${PALETTE_GLSL}${DISPLAY_GLSL}
              // One cell of pixelCells is one pixel of the frame the controller was given, so the brain image
              // lands on the same grid as the game. Light inside a cell is averaged over supported texels.
              float cellLight(vec2 uv,vec2 step){float sum=0.0,weight=0.0;
                for(int y=0;y<2;y++){for(int x=0;x<2;x++){
                  float s=texture2D(light,uv+(vec2(float(x),float(y))-0.5)*step*0.5).r;
                  if(s>=0.0){sum+=s;weight+=1.0;}}}
                return weight>0.0?sum/weight:-1.0;}
              void main(){vec2 uv=vec2(imageUV.x,1.0-imageUV.y);
                bool blocky=pixelCells.x>0.0;
                vec2 step=blocky?1.0/pixelCells:displayTexel;
                if(blocky)uv=pixelOrigin+(floor((uv-pixelOrigin)*pixelCells)+0.5)/pixelCells;
                float raw=blocky?cellLight(uv,step):texture2D(light,uv).r;if(raw<0.0){discard;}
                float value=raw*exposure;
                if(sharpen>0.0){float blur=0.0,weight=0.0;
                  for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){
                    float s=blocky?cellLight(uv+vec2(float(x),float(y))*step,step):texture2D(light,uv+vec2(float(x),float(y))*displayTexel).r;
                    if(s>=0.0){blur+=s*exposure;weight+=1.0;}}}
                  if(weight>0.0)value+=sharpen*(value-blur/weight);}
                float t=tone(value);
                vec3 color=shade(t,palette);
                if(palette==${SOURCE}){vec3 src=texture2D(sourceTexture,uv).rgb;float lum=dot(src,vec3(0.2126,0.7152,0.0722));color=(lum>0.004?src/lum:vec3(1.0))*t;}
                gl_FragColor=vec4(saturate3(color),1.0);}`,
            depthTest:false,depthWrite:false,toneMapped:false});
          this.display.displayTexel.value.set(1/m.resolution[0],1/m.resolution[1]);
          this.cacheGeometry={center:m.center,footprint:m.footprintMicrometers,resolution:m.resolution};
          this.cachedPlane=new THREE.PlaneGeometry(m.footprintMicrometers[0],m.footprintMicrometers[1]);
          const plane=new THREE.Mesh(this.cachedPlane,this.cachedMaterial);plane.position.set(m.center[0],m.center[1],this.center.z);this.cachedScene.add(plane);resolve();
        }else if(m.type==='light'){
          if(!this.cachedTexture||!this.cacheInfo){this.lightPending?.reject(Error('Anatomical cache is not ready.'));return;}
          this.cachedTexture.image.data=m.values;this.cachedTexture.needsUpdate=true;this.cacheInfo.computeMs=m.computeMs;this.dirty=true;
          if(this.autoLevels)this.updateLevels(m.values as Float32Array);
          this.lightPending?.resolve();this.lightPending=null;
        }
      };
      worker.postMessage({type:'init',base:directory});
    });
  }

  /** The only input to the geometry cache is the full measured activity vector.
   * Camera changes still render every original 3D branch from the indexed data. */
  async prepareNeuralFrame(values:Float32Array<ArrayBuffer>){
    this.setNeuralActivity(values);
    values=this.applyRegion(values) as Float32Array<ArrayBuffer>;
    if(!this.lightWorker||!this.cacheInfo)throw Error('Complete anatomical cache is not ready.');
    if(this.lightPending)throw Error('An anatomical frame is already being projected.');
    await new Promise<void>((resolve,reject)=>{this.lightPending={resolve,reject};this.lightWorker!.postMessage({type:'activity',values});});
  }

  private resize(){
    const width=Math.max(1,this.element.clientWidth),height=Math.max(1,this.element.clientHeight);
    this.renderer.setSize(width,height);
    const size=new THREE.Vector3().fromArray(this.meta.bounds[1]).sub(new THREE.Vector3().fromArray(this.meta.bounds[0]));
    const span=this.referenceFootprint??[size.x*1.12,size.y*1.12];
    const half=Math.max(span[1]/2,span[0]*height/width/2);
    this.camera.left=-half*width/height;this.camera.right=half*width/height;this.camera.top=half;this.camera.bottom=-half;this.camera.updateProjectionMatrix();
    const pixels=this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.accumulation.setSize(pixels.x,pixels.y);this.dirty=true;
  }
  render(_now:number,_delta=0,onlyIfChanged=true){
    if(this.lost||this.lightPending&&!this.identityView||onlyIfChanged&&!this.dirty)return;
    this.controls.update();this.dirty=false;
    if(this.identityView){
      this.scene.overrideMaterial=this.inspectionMaterial;this.renderer.setRenderTarget(null);this.renderer.setClearColor(0x101619,1);this.renderer.render(this.scene,this.camera);this.scene.overrideMaterial=null;
      this.drawMode=this.loaded?'static complete indexed geometry':'static indexed geometry, upload incomplete';this.anatomyDrawCalls=this.renderer.info.render.calls;this.frameSubmittedAt=performance.now();return;
    }
    // This cache includes the complete cable of every neuron. It remains fixed
    // to the reference camera; an orbit uses the original 3D indexed geometry.
    if(this.cachedTexture&&this.atCachedFrontView()){
      this.renderer.setRenderTarget(null);this.renderer.setClearColor(0x101619,1);this.renderer.render(this.cachedScene,this.camera);
      this.drawMode='complete cable cache';this.anatomyDrawCalls=this.renderer.info.render.calls;this.frameSubmittedAt=performance.now();return;
    }
    this.renderer.setRenderTarget(this.accumulation);this.renderer.setClearColor(0,0);this.renderer.clear();this.renderer.render(this.scene,this.camera);
    this.anatomyDrawCalls=this.renderer.info.render.calls;
    this.drawMode=this.loaded?'complete indexed geometry':'indexed geometry, upload incomplete';
    this.renderer.setRenderTarget(null);this.renderer.setClearColor(0x101619,1);this.renderer.render(this.screen,this.screenCamera);
    this.frameSubmittedAt=performance.now();
  }
  setNeuralActivity(values:Float32Array){
    if(values.length!==this.meta.neuronCount)throw Error('Every released neuron requires its own activity value.');
    (this.activityTexture.image.data as Float32Array).set(this.applyRegion(values));this.activityTexture.needsUpdate=true;this.dirty=true;
  }
  /** Zero the activity of neurons the region withholds, into a reused scratch
   * buffer so the caller's own measurements are left untouched. */
  private applyRegion(values:Float32Array){
    if(!this.regionMask)return values;
    const masked=this.maskedActivity??=new Float32Array(values.length);
    for(let i=0;i<values.length;i++)masked[i]=values[i]*this.regionMask[i];
    return masked;
  }
  /** Draw only the neurons this region selects, and frame the camera on where
   * that region's cable actually is. A null mask restores the complete brain. */
  setRegion(name:string,mask:Float32Array|null,framingBounds?:[number[],number[]],counts?:{neurons:number;vertices:number;edges:number}){
    if(mask&&mask.length!==this.meta.neuronCount)throw Error('A region mask must cover every released neuron.');
    this.regionName=name;this.regionMask=mask;this.regionCounts=counts??null;this.maskedActivity=null;
    const data=this.regionTexture.image.data as Float32Array;
    data.fill(0);
    if(mask)data.set(mask);else data.fill(1,0,this.meta.neuronCount);
    this.regionTexture.needsUpdate=true;
    if(framingBounds){
      const box=new THREE.Box3(new THREE.Vector3().fromArray(framingBounds[0]),new THREE.Vector3().fromArray(framingBounds[1]));
      box.getCenter(this.center);this.radius=box.getSize(new THREE.Vector3()).length()*1.5;
      this.camera.far=this.radius*5;this.controls.target.copy(this.center);
      this.camera.position.copy(this.center).add(new THREE.Vector3(0,0,this.radius));this.camera.lookAt(this.center);
      const size=box.getSize(new THREE.Vector3());
      this.referenceFootprint=[size.x*1.08,size.y*1.08];this.controls.update();this.resize();
    }
    this.dirty=true;
  }
  getRegion(){return {name:this.regionName,counts:this.regionCounts};}
  /** Wait for actual GPU completion so a long queue cannot masquerade as
   * real-time neural rendering. Display scan-out remains outside this clock. */
  async waitForDraw(){
    const gl=this.renderer.getContext() as WebGL2RenderingContext;
    const fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
    if(!fence)throw Error('Could not measure full-brain GPU completion.');gl.flush();
    try{
      while(true){
        if(this.lost)throw Error('Graphics context lost during full-brain rendering.');
        const status=gl.clientWaitSync(fence,0,0);
        if(status===gl.ALREADY_SIGNALED||status===gl.CONDITION_SATISFIED)break;
        if(status===gl.WAIT_FAILED)throw Error('Full-brain GPU completion failed.');
        await new Promise<void>(resolve=>setTimeout(resolve,2));
      }
      this.completedAt=performance.now();
    }finally{gl.deleteSync(fence);}
  }
  setIdentityView(value:boolean){this.identityView=value;this.dirty=true;if(value)this.beginDemandLoad();}
  /** The controller's target raster covers this footprint around this centre (the operator record's values). */
  setReferenceProjection(footprintMicrometers:number[],center?:number[]){this.referenceFootprint=footprintMicrometers;this.referenceFootprintUniform.value.set(footprintMicrometers[0],footprintMicrometers[1]);if(center)this.referenceCenterUniform.value.set(center[0],center[1]);this.resize();}
  /** The frame the controller was given for the current neural frame; only its hue is used, only in the source-colour mode. */
  setSourceFrame(image:ImageData){
    if(this.sourceTexture.image.width!==image.width||this.sourceTexture.image.height!==image.height){
      this.sourceTexture.dispose();
      this.sourceTexture=new THREE.DataTexture(new Uint8Array(image.data),image.width,image.height,THREE.RGBAFormat,THREE.UnsignedByteType);
      this.sourceUniform.value=this.sourceTexture;
    }else{(this.sourceTexture.image.data as Uint8Array).set(image.data);}
    this.sourceTexture.needsUpdate=true;if(this.palette==='source')this.dirty=true;
  }
  setExposure(value:number){this.composite.uniforms.exposure.value=value;this.dirty=true;}
  /** Display processing of the simulated light only: level stretch, gamma, local contrast, saturation. */
  setDisplay(options:{autoLevels?:boolean;gamma?:number;sharpen?:number;saturation?:number;floor?:number;brightness?:number;pixelCells?:[number,number]|null;pixelOrigin?:[number,number]|null}){
    if(options.autoLevels!==undefined){this.autoLevels=options.autoLevels;if(!options.autoLevels){this.levelRange=null;this.display.levels.value.set(0,0);}}
    if(options.gamma!==undefined)this.display.gamma.value=options.gamma;
    if(options.sharpen!==undefined)this.display.sharpen.value=Math.max(0,options.sharpen);
    if(options.saturation!==undefined)this.display.saturation.value=Math.max(0,options.saturation);
    if(options.floor!==undefined)this.display.floorLevel.value=Math.min(.5,Math.max(0,options.floor));
    if(options.brightness!==undefined)this.display.brightness.value=Math.min(3,Math.max(.05,options.brightness));
    if(options.pixelCells!==undefined)this.display.pixelCells.value.set(...(options.pixelCells??[0,0]) as [number,number]);
    if(options.pixelOrigin!==undefined)this.display.pixelOrigin.value.set(...(options.pixelOrigin??[0,0]) as [number,number]);
    this.dirty=true;
  }
  getDisplay(){return {autoLevels:this.autoLevels,gamma:this.display.gamma.value,sharpen:this.display.sharpen.value,saturation:this.display.saturation.value,floor:this.display.floorLevel.value,brightness:this.display.brightness.value,pixelCells:this.display.pixelCells.value.toArray(),pixelOrigin:this.display.pixelOrigin.value.toArray(),levels:this.display.levels.value.toArray()};}
  /** Stretch to the light range of the pixels actually on screen: 2nd and 99.5th percentile of the lit
   * ones inside the current framing (0.5th and 99.5th percentile), smoothed over frames. Unsupported
   * texels (-1) are excluded, and floorLevel keeps the dimmest supported ones above the background. */
  private updateLevels(values:Float32Array){
    const cache=this.cacheGeometry;if(!cache)return;
    const [width,height]=cache.resolution;
    const halfWidth=(this.camera.right-this.camera.left)/2/this.camera.zoom,halfHeight=(this.camera.top-this.camera.bottom)/2/this.camera.zoom;
    const toColumn=(x:number)=>Math.round((x-(cache.center[0]-cache.footprint[0]/2))/cache.footprint[0]*width);
    const toRow=(y:number)=>Math.round(((cache.center[1]+cache.footprint[1]/2)-y)/cache.footprint[1]*height);
    const x0=Math.max(0,toColumn(this.camera.position.x-halfWidth)),x1=Math.min(width,toColumn(this.camera.position.x+halfWidth));
    const y0=Math.max(0,toRow(this.camera.position.y+halfHeight)),y1=Math.min(height,toRow(this.camera.position.y-halfHeight));
    if(x1-x0<2||y1-y0<2)return;
    const step=Math.max(1,Math.round(Math.sqrt((x1-x0)*(y1-y0)/40000)));
    const sample:number[]=[];
    for(let y=y0;y<y1;y+=step)for(let x=x0;x<x1;x+=step){const v=values[y*width+x];if(v>0)sample.push(v);}
    if(sample.length<64)return;
    sample.sort((a,b)=>a-b);
    const low=sample[Math.floor(sample.length*.005)],high=sample[Math.floor(sample.length*.995)];
    if(!(high>low))return;
    const exposure=this.composite.uniforms.exposure.value as number;
    const next:[number,number]=[low*exposure,high*exposure];
    // Smooth across frames so the picture does not pulse when a bright patch appears.
    this.levelRange=this.levelRange?[this.levelRange[0]*.8+next[0]*.2,this.levelRange[1]*.8+next[1]*.2]:next;
    this.display.levels.value.set(this.levelRange[0],this.levelRange[1]);
  }
  /** False-colour mapping of the same spike-light value; grayscale is the original display. */
  setPalette(palette:Palette){const index=PALETTES.indexOf(palette);if(index<0)throw Error(`Unknown palette: ${palette}`);this.palette=palette;this.paletteUniform.value=index;this.dirty=true;}
  getPalette(){return this.palette;}
  goTo(view:'front'|'full'|'image'|'oblique'|'side'){
    const direction=view==='side'?new THREE.Vector3(1,0,0):view==='oblique'?new THREE.Vector3(.7,.2,1).normalize():new THREE.Vector3(0,0,1);
    this.controls.target.copy(this.center);this.camera.position.copy(this.center).addScaledVector(direction,this.radius);this.camera.zoom=view==='image'?1.5:1;this.camera.updateProjectionMatrix();this.controls.update();this.dirty=true;
  }
  getInfo(){return {...this.counts,complete:this.loaded,geometryPending:!this.loaded&&this.pending!==null,identityView:this.identityView,palette:this.palette,display:this.getDisplay(),submittedAt:this.frameSubmittedAt,completedAt:this.completedAt,drawCalls:this.anatomyDrawCalls,drawMode:this.drawMode,cache:this.cacheInfo};}
  saveImage(){this.render(performance.now(),0,false);return new Promise<Blob>((resolve,reject)=>this.renderer.domElement.toBlob(blob=>blob?resolve(blob):reject(Error('Could not save the full-brain view.'))));}
  dispose(){this.lightWorker?.terminate();this.sourceTexture.dispose();this.cachedTexture?.dispose();this.cachedMaterial?.dispose();this.cachedPlane?.dispose();this.resizeObserver.disconnect();this.controls.dispose();this.geometries.forEach(g=>g.dispose());this.material.dispose();this.inspectionMaterial.dispose();this.composite.dispose();this.screenGeometry.dispose();this.accumulation.dispose();this.activityTexture.dispose();this.renderer.dispose();}
}
