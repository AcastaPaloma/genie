import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { BrainMetadata } from './types';
import { ActivityRenderer } from './activity-renderer';

const vertexShader = /* glsl */ `
  attribute float neuronIndex;
  uniform vec2 footprint;
  uniform vec2 projectionCenter;
  uniform float pointSize;
  uniform float pixelRatio;
  uniform float zoomScale;
  varying vec2 referenceUV;
  varying float neuron;
  varying float depth;
  void main() {
    referenceUV = (position.xy - projectionCenter) / footprint + 0.5;
    neuron = neuronIndex;
    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
    depth = viewPosition.z;
    gl_Position = projectionMatrix * viewPosition;
    gl_PointSize = pointSize * pixelRatio * sqrt(zoomScale);
  }
`;

const colorShader = /* glsl */ `
  uniform sampler2D frameTexture;
  uniform float exposure;
  uniform float anatomy;
  uniform float monochrome;
  uniform float depthMix;
  uniform float neuralMode;
  uniform sampler2D activityTexture;
  uniform vec2 activitySize;
  varying vec2 referenceUV;
  varying float neuron;
  varying float depth;
  vec3 neuronColor(float id) {
    float hue = fract(id * 0.61803398875);
    vec3 rgb = clamp(abs(fract(hue + vec3(0.0, 0.66667, 0.33333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(0.28), rgb, 0.65);
  }
  vec3 displayColor() {
    if (neuralMode > 0.5) {
      float activity = texture2D(activityTexture, (vec2(mod(floor(neuron),activitySize.x),floor(neuron/activitySize.x))+0.5)/activitySize).r;
      return vec3(0.013, 0.022, 0.028) + vec3(activity * exposure);
    }
    bool inside = all(greaterThanEqual(referenceUV, vec2(0.0))) && all(lessThanEqual(referenceUV, vec2(1.0)));
    vec3 projected = inside ? texture2D(frameTexture, referenceUV).rgb * exposure : vec3(0.022, 0.036, 0.045);
    projected = mix(projected, vec3(dot(projected, vec3(0.2126, 0.7152, 0.0722))), monochrome);
    vec3 color = mix(projected, neuronColor(neuron), anatomy);
    float cue = clamp(1.2 + (depth + 650.0) / 320.0, 0.5, 1.2);
    return color * mix(1.0, cue, depthMix);
  }
`;

const pointFragment = /* glsl */ `
  ${colorShader}
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    if (dot(uv, uv) > 0.25) discard;
    gl_FragColor = vec4(displayColor(), 1.0);
    #include <colorspace_fragment>
  }
`;

const lineFragment = /* glsl */ `
  ${colorShader}
  void main() {
    gl_FragColor = vec4(displayColor(), mix(0.16, 0.3, anatomy));
    #include <colorspace_fragment>
  }
`;

export class BrainView {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.OrthographicCamera(-200, 200, 150, -150, 1, 2000);
  readonly scene = new THREE.Scene();
  readonly controls: OrbitControls;
  readonly texture: THREE.CanvasTexture;
  readonly activityTexture: THREE.DataTexture;
  readonly meta: BrainMetadata;
  readonly uniforms: Record<string, THREE.IUniform>;
  readonly element: HTMLElement;
  private geometries: THREE.BufferGeometry[] = [];
  private materials: THREE.ShaderMaterial[] = [];
  private points: THREE.Points | null = null;
  private lines: THREE.LineSegments | null = null;
  private resizeObserver: ResizeObserver;
  private baseFootprint: THREE.Vector2;
  private transition: { start: number; from: THREE.Vector3; to: THREE.Vector3; zoom: number; targetZoom:number } | null = null;
  private radius = 650;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  private autoOrbit = false;
  private lost = false;
  private dirty = true;
  private activityRenderer:ActivityRenderer|null=null;
  private normalizeActivity=false;
  private imageDirection=new THREE.Vector3(0,0,1);
  onViewChange: (angle: number) => void = () => {};
  onContextLost: () => void = () => {};

  constructor(element: HTMLElement, source: HTMLCanvasElement, meta: BrainMetadata) {
    this.element = element;
    this.meta = meta;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setClearColor(0x101619);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D fly brain. Drag to orbit, scroll to zoom. Use the Front and Reveal depth buttons for keyboard camera control.');
    this.renderer.domElement.setAttribute('role', 'img');
    this.renderer.domElement.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      this.lost = true;
      this.onContextLost();
    });
    element.append(this.renderer.domElement);
    this.camera.position.set(0, 0, this.radius);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.enablePan = false;
    this.controls.minZoom = 0.55;
    this.controls.maxZoom = 3.5;
    this.controls.rotateSpeed = 0.55;
    this.controls.addEventListener('change',()=>{this.dirty=true;});
    this.controls.addEventListener('start', () => { this.transition = null; });
    this.texture = new THREE.CanvasTexture(source);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    const activityWidth = Math.min(2048,THREE.MathUtils.ceilPowerOfTwo(meta.neuronCount));
    const activityHeight = Math.ceil(meta.neuronCount/activityWidth);
    this.activityTexture = new THREE.DataTexture(new Float32Array(activityWidth*activityHeight), activityWidth, activityHeight, THREE.RedFormat, THREE.FloatType);
    this.activityTexture.needsUpdate = true;
    // Fixed central footprint: 95% of its 160 × 90 cells contain branch samples.
    // The full bounding box has large empty regions and would lose the HUD.
    this.baseFootprint = new THREE.Vector2(400, 225);
    this.uniforms = {
      footprint: { value: this.baseFootprint.clone() },
      projectionCenter: { value: new THREE.Vector2(0, 25) },
      frameTexture: { value: this.texture },
      pointSize: { value: 2.1 },
      pixelRatio: { value: this.renderer.getPixelRatio() },
      zoomScale: { value: 1 },
      exposure: { value: 1.3 },
      anatomy: { value: 0 },
      monochrome: { value: 0 },
      depthMix: { value: 0 },
      neuralMode: { value: 0 },
      activityTexture: { value: this.activityTexture },
      activitySize: { value: new THREE.Vector2(activityWidth,activityHeight) },
    };
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(element);
    this.resize();
  }

  async load(onProgress: (text: string) => void, directory=`${import.meta.env.BASE_URL}data/`, pointFile='brain-points.bin') {
    const buffers = await Promise.all([pointFile, 'brain-lines.bin'].map(async (file) => {
      if(file==='brain-lines.bin'&&this.meta.lineVertexCount===0)return new ArrayBuffer(0);
      const response = await fetch(`${directory}${file}`);
      if (!response.ok) throw new Error(`The anatomy file ${file} could not be loaded (${response.status}).`);
      return response.arrayBuffer();
    }));
    const expectedCounts = [this.meta.pointCount, this.meta.lineVertexCount];
    const geometries = buffers.map((buffer, index) => {
      if (buffer.byteLength !== expectedCounts[index] * 16) throw new Error('Anatomy data is incomplete. Rebuild the local assets and reload.');
      const interleaved = new THREE.InterleavedBuffer(new Float32Array(buffer), 4);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
      geometry.setAttribute('neuronIndex', new THREE.InterleavedBufferAttribute(interleaved, 1, 3));
      geometry.computeBoundingSphere();
      return geometry;
    });
    onProgress('Building the anatomical display…');
    this.geometries = geometries;
    const pointsMaterial = new THREE.ShaderMaterial({ vertexShader, fragmentShader: pointFragment, uniforms: this.uniforms, depthTest: true, depthWrite: true });
    const linesMaterial = new THREE.ShaderMaterial({ vertexShader, fragmentShader: lineFragment, uniforms: this.uniforms, transparent: true, depthWrite: false });
    this.materials = [pointsMaterial, linesMaterial];
    this.points = new THREE.Points(geometries[0], pointsMaterial);
    this.lines = new THREE.LineSegments(geometries[1], linesMaterial);
    this.scene.add(this.points, this.lines);
    this.renderer.compile(this.scene, this.camera);
    this.render(performance.now(), 0);
  }

  private resize() {
    this.dirty=true;
    const width = Math.max(this.element.clientWidth, 1);
    const height = Math.max(this.element.clientHeight, 1);
    this.renderer.setSize(width, height);
    const brainWidth = this.meta.bounds[1][0] - this.meta.bounds[0][0];
    const brainHeight = this.meta.bounds[1][1] - this.meta.bounds[0][1];
    const aspect = width / height;
    const halfHeight = Math.max(brainHeight * 0.64, brainWidth / aspect * 0.59);
    this.camera.left = -halfHeight * aspect;
    this.camera.right = halfHeight * aspect;
    this.camera.top = halfHeight;
    this.camera.bottom = -halfHeight;
    this.camera.updateProjectionMatrix();
  }

  render(now: number, delta: number, onlyIfChanged=false) {
    if (this.lost) return;
    if (this.transition) {
      this.dirty=true;
      const progress = Math.min(1, (now - this.transition.start) / (this.reducedMotion ? 1 : 1100));
      const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      this.camera.position.copy(this.transition.from).lerp(this.transition.to, eased).normalize().multiplyScalar(this.radius);
      this.camera.zoom = THREE.MathUtils.lerp(this.transition.zoom, this.transition.targetZoom, eased);
      this.camera.updateProjectionMatrix();
      if (progress === 1) this.transition = null;
    }
    this.controls.autoRotate = this.autoOrbit && !this.transition;
    this.controls.autoRotateSpeed = 0.65;
    this.controls.update(delta);
    const dot = THREE.MathUtils.clamp(this.camera.position.clone().normalize().dot(new THREE.Vector3(0, 0, 1)), -1, 1);
    const angle = Math.acos(dot) * 180 / Math.PI;
    this.uniforms.depthMix.value = Math.min(1, angle / 50) * 0.6;
    this.uniforms.zoomScale.value = this.camera.zoom;
    this.onViewChange(angle);
    if(onlyIfChanged&&!this.dirty)return;this.dirty=false;
    if(this.normalizeActivity&&this.points)this.activityRenderer!.render(this.renderer,this.scene,this.camera,this.points,this.lines);
    else this.renderer.render(this.scene, this.camera);
  }

  goTo(view: 'front' | 'oblique' | 'side' | 'image' | 'full') {
    this.autoOrbit = false;
    const directions = { front: new THREE.Vector3(0, 0, 1), image:this.imageDirection.clone(), full:this.imageDirection.clone(), oblique: new THREE.Vector3(0.84, 0.23, 0.55), side: new THREE.Vector3(1, 0, 0.025) };
    this.transition = { start: performance.now(), from: this.camera.position.clone(), to: directions[view].normalize().multiplyScalar(this.radius), zoom: this.camera.zoom, targetZoom:view==='image'?1.6:1 };
  }

  setImageView(normal:number[]) {
    this.dirty=true;
    this.imageDirection.fromArray(normal).normalize();
    this.camera.position.copy(this.imageDirection).multiplyScalar(this.radius);
    this.camera.zoom=1.6;this.camera.updateProjectionMatrix();this.controls.update();
  }

  setAutoOrbit(active: boolean) { this.autoOrbit = active; }
  setAnatomy(active: boolean) { this.uniforms.anatomy.value = active ? 1 : 0; }
  setMonochrome(active: boolean) { this.uniforms.monochrome.value = active ? 1 : 0; }
  setExposure(value: number) { this.uniforms.exposure.value = value; }
  setPointSize(value: number) { this.uniforms.pointSize.value = value; }
  setFootprint(value: number) { this.uniforms.footprint.value.copy(this.baseFootprint).multiplyScalar(value); }
  updateFrame() { this.texture.needsUpdate = true; }
  setActivityNormalization(enabled:boolean,cell=7.5) {
    this.dirty=true;
    if(enabled&&!this.activityRenderer)this.activityRenderer=new ActivityRenderer(this.activityTexture,this.uniforms.activitySize.value);
    this.normalizeActivity=enabled;if(this.activityRenderer)this.activityRenderer.cell=cell;
  }
  setActivityBlur(sigma:number){this.dirty=true;this.activityRenderer?.setBlur(sigma);}
  setNeuralActivity(values: Float32Array) {
    this.dirty=true;
    if (values.length !== this.meta.neuronCount) throw new Error('Activity and displayed anatomy do not match.');
    (this.activityTexture.image.data as Float32Array).set(values);
    this.activityTexture.needsUpdate = true;
    this.uniforms.neuralMode.value = 1;
    this.renderer.domElement.setAttribute('aria-label','Simulated neuronal spike light on real fly anatomy. Drag to orbit, or use the Brain viewpoint menu.');
  }
  async changePointGeometry(url:string,count:number) {
    this.dirty=true;
    const response=await fetch(url);if(!response.ok)throw Error(`Could not load anatomy (${response.status}).`);
    const raw=await response.arrayBuffer();if(raw.byteLength!==count*16)throw Error('Anatomy point count differs.');
    const buffer=new THREE.InterleavedBuffer(new Float32Array(raw),4),geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));geometry.setAttribute('neuronIndex',new THREE.InterleavedBufferAttribute(buffer,1,3));geometry.computeBoundingSphere();
    if(this.points){this.points.geometry=geometry;this.geometries[0].dispose();this.geometries[0]=geometry;this.dirty=true;}
  }
  getInfo() { return { points: this.meta.pointCount, drawCalls: this.renderer.info.render.calls, angle: this.camera.position.angleTo(new THREE.Vector3(0, 0, 1)) * 180 / Math.PI }; }

  saveImage() {
    this.render(performance.now(),0);
    return new Promise<Blob>((resolve, reject) => this.renderer.domElement.toBlob((blob) => blob ? resolve(blob) : reject(new Error('Could not capture the viewport.'))));
  }

  dispose() {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.geometries.forEach((g) => g.dispose());
    this.materials.forEach((m) => m.dispose());
    this.texture.dispose();
    this.activityTexture.dispose();
    this.activityRenderer?.dispose();
    this.renderer.dispose();
  }
}
