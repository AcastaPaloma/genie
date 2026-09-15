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
  readonly counts={neurons:0,vertices:0,edges:0,isolatedVertices:0,chunks:0};
  onContextLost:()=>void=()=>{};

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
    this.controls.addEventListener('change',()=>{this.dirty=true;});
    const width=2048,height=Math.ceil(meta.neuronCount/width);
    this.activityTexture=new THREE.DataTexture(new Float32Array(width*height),width,height,THREE.RedFormat,THREE.FloatType);
    this.activityTexture.needsUpdate=true;
    this.material=new THREE.ShaderMaterial({
      uniforms:{activityTexture:{value:this.activityTexture},activitySize:{value:new THREE.Vector2(width,height)}},
      vertexShader:`attribute float neuronIndex; uniform sampler2D activityTexture; uniform vec2 activitySize; varying float activity; varying float owner;
        void main(){owner=neuronIndex;activity=texture2D(activityTexture,(vec2(mod(neuronIndex,activitySize.x),floor(neuronIndex/activitySize.x))+0.5)/activitySize).r;
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=1.0;}`,
      fragmentShader:`varying float activity; void main(){gl_FragColor=vec4(activity,1.0,0.0,1.0);}`,
      transparent:true,blending:THREE.AdditiveBlending,depthTest:false,depthWrite:false,toneMapped:false,
    });
    // A separately labelled anatomy inspector uses identity colours and depth.
    // It is never the activity output and has no access to the video either.
    this.inspectionMaterial=new THREE.ShaderMaterial({
      vertexShader:`attribute float neuronIndex;varying vec3 identityColor;
        void main(){float hue=fract(neuronIndex*0.61803398875);
          identityColor=0.58+0.35*cos(6.2831853*(hue+vec3(0.0,0.333333,0.666667)));
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=1.0;}`,
      fragmentShader:`varying vec3 identityColor;void main(){gl_FragColor=vec4(identityColor,1.0);}`,
      depthTest:true,depthWrite:true,toneMapped:false,
    });
    this.accumulation=new THREE.WebGLRenderTarget(1,1,{type:THREE.FloatType,depthBuffer:false,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter});
    this.composite=new THREE.ShaderMaterial({
      uniforms:{light:{value:this.accumulation.texture},exposure:{value:1}},
      vertexShader:`varying vec2 screenUV; void main(){screenUV=uv;gl_Position=vec4(position.xy,0.0,1.0);}`,
      fragmentShader:`uniform sampler2D light;uniform float exposure;varying vec2 screenUV;
        void main(){vec2 sum=texture2D(light,screenUV).rg;
          vec3 color=sum.y>0.0?vec3(clamp(exposure*sum.x/sum.y,0.0,1.0)):vec3(0.062745,0.086275,0.098039);
          gl_FragColor=vec4(color,1.0);}`,
      depthTest:false,depthWrite:false,toneMapped:false,
    });
    this.screen.add(new THREE.Mesh(this.screenGeometry,this.composite));
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(element);this.resize();
  }

  async load(directory:string,onProgress:(loaded:number,total:number)=>void){
    let neuronCount=0;
    for(const chunk of this.meta.chunks){
      const response=await fetch(directory+chunk.file);
      if(!response.ok)throw Error(`Full anatomy chunk could not be loaded (${response.status}): ${chunk.file}`);
      let raw:ArrayBuffer|null=await response.arrayBuffer();
      const header=new Uint32Array(raw,0,4);
      const [version,vertices,edges,isolated]=header;
      if(version!==785||vertices!==chunk.vertices||edges!==chunk.edges||isolated!==chunk.isolatedVertices||raw.byteLength!==16+vertices*20+edges*8+isolated*4)
        throw Error(`Incomplete full-anatomy chunk: ${chunk.file}`);
      const checksum=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),n=>n.toString(16).padStart(2,'0')).join('');
      if(checksum!==chunk.sha256)throw Error(`Full-anatomy checksum differs: ${chunk.file}`);
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
  }

  async loadLightCache(directory:string){
    if(this.lightWorker)throw Error('Anatomical cache already loaded.');
    const worker=this.lightWorker=new Worker(new URL('./anatomical-light.worker.ts',import.meta.url),{type:'module'});
    await new Promise<void>((resolve,reject)=>{
      worker.onerror=event=>{const error=Error(event.message);reject(error);this.lightPending?.reject(error);this.lightPending=null;};
      worker.onmessage=event=>{
        const m=event.data;
        if(m.type==='error'){const error=Error(m.message);reject(error);this.lightPending?.reject(error);this.lightPending=null;return;}
        if(m.type==='ready'){
          if(m.neurons!==this.meta.neuronCount||m.branches!==this.meta.edgeCount){reject(Error('Anatomical cache coverage differs.'));return;}
          this.cacheInfo={resolution:m.resolution,entries:m.entries,branches:m.branches,computeMs:0};
          this.cachedTexture=new THREE.DataTexture(new Float32Array(m.resolution[0]*m.resolution[1]).fill(-1),m.resolution[0],m.resolution[1],THREE.RedFormat,THREE.FloatType);
          this.cachedTexture.needsUpdate=true;
          this.cachedMaterial=new THREE.ShaderMaterial({uniforms:{light:{value:this.cachedTexture},exposure:this.composite.uniforms.exposure},
            vertexShader:`varying vec2 imageUV;void main(){imageUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
            fragmentShader:`uniform sampler2D light;uniform float exposure;varying vec2 imageUV;void main(){float value=texture2D(light,vec2(imageUV.x,1.0-imageUV.y)).r;if(value<0.0){discard;}gl_FragColor=vec4(vec3(clamp(value*exposure,0.0,1.0)),1.0);}`,
            depthTest:false,depthWrite:false,toneMapped:false});
          this.cachedPlane=new THREE.PlaneGeometry(m.footprintMicrometers[0],m.footprintMicrometers[1]);
          const plane=new THREE.Mesh(this.cachedPlane,this.cachedMaterial);plane.position.set(m.center[0],m.center[1],this.center.z);this.cachedScene.add(plane);resolve();
        }else if(m.type==='light'){
          if(!this.cachedTexture||!this.cacheInfo){this.lightPending?.reject(Error('Anatomical cache is not ready.'));return;}
          this.cachedTexture.image.data=m.values;this.cachedTexture.needsUpdate=true;this.cacheInfo.computeMs=m.computeMs;this.dirty=true;
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
      this.drawMode='static complete indexed geometry';this.anatomyDrawCalls=this.renderer.info.render.calls;this.frameSubmittedAt=performance.now();return;
    }
    // This cache includes the complete cable of every neuron. It remains fixed
    // to the reference camera; an orbit uses the original 3D indexed geometry.
    const direction=this.camera.getWorldDirection(new THREE.Vector3());
    if(this.cachedTexture&&Math.abs(direction.x)<1e-5&&Math.abs(direction.y)<1e-5&&direction.z<0){
      this.renderer.setRenderTarget(null);this.renderer.setClearColor(0x101619,1);this.renderer.render(this.cachedScene,this.camera);
      this.drawMode='complete cable cache';this.anatomyDrawCalls=this.renderer.info.render.calls;this.frameSubmittedAt=performance.now();return;
    }
    this.renderer.setRenderTarget(this.accumulation);this.renderer.setClearColor(0,0);this.renderer.clear();this.renderer.render(this.scene,this.camera);
    this.anatomyDrawCalls=this.renderer.info.render.calls;
    this.drawMode='complete indexed geometry';
    this.renderer.setRenderTarget(null);this.renderer.setClearColor(0x101619,1);this.renderer.render(this.screen,this.screenCamera);
    this.frameSubmittedAt=performance.now();
  }
  setNeuralActivity(values:Float32Array){
    if(values.length!==this.meta.neuronCount)throw Error('Every released neuron requires its own activity value.');
    (this.activityTexture.image.data as Float32Array).set(values);this.activityTexture.needsUpdate=true;this.dirty=true;
  }
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
  setIdentityView(value:boolean){this.identityView=value;this.dirty=true;}
  setReferenceProjection(footprintMicrometers:number[]){this.referenceFootprint=footprintMicrometers;this.resize();}
  setExposure(value:number){this.composite.uniforms.exposure.value=value;this.dirty=true;}
  goTo(view:'front'|'full'|'image'|'oblique'|'side'){
    const direction=view==='side'?new THREE.Vector3(1,0,0):view==='oblique'?new THREE.Vector3(.7,.2,1).normalize():new THREE.Vector3(0,0,1);
    this.controls.target.copy(this.center);this.camera.position.copy(this.center).addScaledVector(direction,this.radius);this.camera.zoom=view==='image'?1.5:1;this.camera.updateProjectionMatrix();this.controls.update();this.dirty=true;
  }
  getInfo(){return {...this.counts,complete:this.loaded,identityView:this.identityView,submittedAt:this.frameSubmittedAt,completedAt:this.completedAt,drawCalls:this.anatomyDrawCalls,drawMode:this.drawMode,cache:this.cacheInfo};}
  saveImage(){this.render(performance.now(),0,false);return new Promise<Blob>((resolve,reject)=>this.renderer.domElement.toBlob(blob=>blob?resolve(blob):reject(Error('Could not save the full-brain view.'))));}
  dispose(){this.lightWorker?.terminate();this.cachedTexture?.dispose();this.cachedMaterial?.dispose();this.cachedPlane?.dispose();this.resizeObserver.disconnect();this.controls.dispose();this.geometries.forEach(g=>g.dispose());this.material.dispose();this.inspectionMaterial.dispose();this.composite.dispose();this.screenGeometry.dispose();this.accumulation.dispose();this.activityTexture.dispose();this.renderer.dispose();}
}
