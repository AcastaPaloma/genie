import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {loadScene,buildMeshes,syncMeshes} from './vendor/flygym-scene.js';
import {Controller} from './vendor/flygym-controller.js';

/** Real MuJoCo rigid-body dynamics and the published NeuroMechFly body model.
 * The low-level CPG uses measured step trajectories. It is an explicit motor
 * adapter, not a claim that the VNC/muscles were reconstructed from FlyWire.
 */
export class FlyBodyView {
  private renderer:THREE.WebGLRenderer;
  private scene=new THREE.Scene();
  private camera=new THREE.PerspectiveCamera(40,1,.01,1000);
  private controls:OrbitControls;
  private observer:ResizeObserver;
  private physics:Awaited<ReturnType<typeof loadScene>>|null=null;
  private controller:Controller|null=null;
  private meshes:THREE.Group|null=null;
  private follow=new THREE.Vector3();
  private framingScale=1;
  private dirty=true;
  constructor(private element:HTMLElement) {
    this.renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
    this.renderer.setClearColor(0x101619);
    this.renderer.domElement.setAttribute('role','img');
    this.renderer.domElement.setAttribute('aria-label','Physically simulated NeuroMechFly body. Joint motion is driven by the neural output adapter.');
    element.append(this.renderer.domElement);
    this.camera.up.set(0,0,1);this.camera.position.set(-5,-7,4.5);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.addEventListener('change',()=>{this.dirty=true;});
    this.controls.target.set(0,0,.7);this.controls.enableDamping=true;
    this.controls.minDistance=2;this.controls.maxDistance=22;
    this.scene.add(new THREE.HemisphereLight(0xe7f2ff,0x79735e,2.7));
    const light=new THREE.DirectionalLight(0xffffff,3.5);light.position.set(-3,-5,10);this.scene.add(light);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x313b3e,roughness:1}));
    floor.position.z=-.01;this.scene.add(floor);
    const grid=new THREE.GridHelper(200,200,0x566569,0x3c474a);grid.rotation.x=Math.PI/2;grid.position.z=.002;this.scene.add(grid);
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(element);this.resize();
  }
  async load(onStage:(text:string)=>void) {
    this.physics=await loadScene({assetsDir:`${import.meta.env.BASE_URL}body/assets`,onStage});
    this.controller=new Controller(this.physics.meta);
    this.meshes=buildMeshes(this.physics.model,this.physics.meta);
    // Keep the supplied arena's fixtures and physical contacts intact.
    this.scene.add(this.meshes);this.render();
  }
  setTimestep(seconds:number) {
    if(!this.physics)return;
    const options=this.physics.model.opt;options.timestep=seconds;this.physics.model.opt=options;this.physics.meta.timestep=seconds;
    this.controller=new Controller(this.physics.meta);
  }
  applyPhysicalState(state:{qpos:Float64Array;positions:Float64Array;rotations:Float64Array;time:number}) {
    if(!this.physics)return;
    this.dirty=true;this.physics.data.qpos.set(state.qpos);this.physics.data.geom_xpos.set(state.positions);this.physics.data.geom_xmat.set(state.rotations);this.physics.data.time=state.time;
  }
  setView(view:string) {
    const directions:Record<string,THREE.Vector3>={angled:new THREE.Vector3(-5,-7,4.5),front:new THREE.Vector3(8,0,2.5),side:new THREE.Vector3(0,-8,2.5),top:new THREE.Vector3(0,-.01,9)};
    this.controls.target.copy(this.follow).add(new THREE.Vector3(0,0,.7));
    this.camera.position.copy(directions[view]??directions.angled).sub(new THREE.Vector3(0,0,.7)).multiplyScalar(this.framingScale).add(this.controls.target);
    this.controls.update();
  }
  step(ms:number,left:number,right:number) {
    if(!this.physics||!this.controller)return;
    const {mj,model,data,meta}=this.physics;
    const steps=Math.round(ms/(meta.timestep*1000));
    for(let i=0;i<steps;i++){
      this.controller.stepCPG(data.ctrl,left,right);
      mj.mj_step(model,data);
    }
    if(!Number.isFinite(data.qpos[0]))throw Error('The body simulation became unstable. Reset the experiment.');
  }
  render(onlyIfChanged=false) {
    if(this.physics&&this.meshes){
      syncMeshes(this.meshes,this.physics.data);
      const position=new THREE.Vector3(this.physics.data.qpos[0],this.physics.data.qpos[1],0);
      const movement=position.clone().sub(this.follow);
      this.camera.position.add(movement);this.controls.target.add(movement);this.follow.copy(position);
    }
    this.controls.update();if(onlyIfChanged&&!this.dirty)return;this.dirty=false;this.renderer.render(this.scene,this.camera);
  }
  getState(){
    if(!this.physics)return null;
    const {data}=this.physics;
    return {time:data.time,position:[data.qpos[0],data.qpos[1],data.qpos[2]],joints:Array.from(data.qpos.slice(7,49)) as number[]};
  }
  reset(){
    if(!this.physics)return;
    const {mj,model,data}=this.physics;
    mj.mj_resetDataKeyframe(model,data,0);mj.mj_forward(model,data);this.controller?.reset();
  }
  private resize(){this.dirty=true;
    const w=Math.max(1,this.element.clientWidth),h=Math.max(1,this.element.clientHeight);
    // Preserve the fly's horizontal framing in a tall, narrow split panel.
    // Scaling the existing orbit offset also preserves the user's zoom when
    // the panel resizes, instead of resetting their chosen viewpoint.
    const scale=Math.max(1,h/w);
    this.camera.position.sub(this.controls.target).multiplyScalar(scale/this.framingScale).add(this.controls.target);
    this.framingScale=scale;this.controls.maxDistance=22*scale;
    this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();
  }
  dispose(){
    this.observer.disconnect();this.controls.dispose();
    this.scene.traverse(node=>{if(node instanceof THREE.Mesh){node.geometry.dispose();const mats=Array.isArray(node.material)?node.material:[node.material];mats.forEach(m=>m.dispose());}});
    this.physics?.data.delete();this.physics?.model.delete();this.renderer.dispose();
  }
}
