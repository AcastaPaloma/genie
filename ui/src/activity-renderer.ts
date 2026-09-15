import * as THREE from 'three';

/** Accumulate scalar neuron activity over actual 3D branches, then divide by
 * branch density. This renderer accepts anatomy and measured activity only.
 * It has no video texture, target image, or per-point target colors.
 */
export class ActivityRenderer {
  private target=new THREE.WebGLRenderTarget(1,1,{type:THREE.FloatType,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:false});
  private camera=new THREE.OrthographicCamera();
  private scene=new THREE.Scene();
  private screenCamera=new THREE.Camera();
  private geometry=new THREE.PlaneGeometry(2,2);
  private accumulation:THREE.ShaderMaterial;
  private composite:THREE.ShaderMaterial;
  private clear=new THREE.Color();

  cell=7.5;
  constructor(activity:THREE.DataTexture,size:THREE.Vector2) {
    this.accumulation=new THREE.ShaderMaterial({
      uniforms:{activityTexture:{value:activity},activitySize:{value:size}},
      vertexShader:`attribute float neuronIndex; uniform sampler2D activityTexture; uniform vec2 activitySize; varying float activity;
        void main(){activity=texture2D(activityTexture,(vec2(mod(floor(neuronIndex),activitySize.x),floor(neuronIndex/activitySize.x))+0.5)/activitySize).r;
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=1.0;}`,
      fragmentShader:`varying float activity; void main(){gl_FragColor=vec4(activity,1.0,0.0,1.0);}`,
      transparent:true,blending:THREE.AdditiveBlending,depthTest:false,depthWrite:false,toneMapped:false,
    });
    this.composite=new THREE.ShaderMaterial({
      uniforms:{accumulation:{value:this.target.texture},zoom:{value:1},blur:{value:0},texel:{value:new THREE.Vector2(1,1)}},
      vertexShader:`varying vec2 screenUV; void main(){screenUV=uv;gl_Position=vec4(position.xy,0.0,1.0);}`,
      fragmentShader:`uniform sampler2D accumulation; uniform float zoom; uniform float blur; uniform vec2 texel; varying vec2 screenUV;
        void main(){vec2 sum=texture2D(accumulation,(screenUV-0.5)/zoom+0.5).rg;
          float light=0.0;float density=0.0;
          for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){
            float weight=blur>0.0?exp(-float(x*x+y*y)/(2.0*blur*blur)):float(x==0&&y==0);
            vec2 sampleLight=texture2D(accumulation,(screenUV-0.5)/zoom+0.5+vec2(float(x),float(y))*texel).rg;
            if(sampleLight.y>0.0){light+=weight*sampleLight.x/sampleLight.y;density+=weight;}
          }}
          vec3 color=density>0.0?vec3(clamp(light/density,0.0,1.0)):vec3(0.062745,0.086275,0.098039);
          gl_FragColor=vec4(color,1.0);}`,
      depthTest:false,depthWrite:false,toneMapped:false,
    });
    this.scene.add(new THREE.Mesh(this.geometry,this.composite));
  }

  render(renderer:THREE.WebGLRenderer,scene:THREE.Scene,view:THREE.OrthographicCamera,points:THREE.Points,lines:THREE.LineSegments|null) {
    this.camera.copy(view);
    this.camera.zoom=1;this.composite.uniforms.zoom.value=view.zoom;
    // Align the reference-view raster to the 7.5 μm cells of the anatomical
    // operator (300 × 225 μm at 40 × 30 pixels). Orbit still moves real geometry.
    const cell=this.cell;
    this.camera.left=Math.floor(view.left/cell)*cell;this.camera.right=Math.ceil(view.right/cell)*cell;
    this.camera.bottom=-87.5+Math.floor((view.bottom+87.5)/cell)*cell;
    this.camera.top=-87.5+Math.ceil((view.top+87.5)/cell)*cell;
    this.camera.updateProjectionMatrix();
    const w=Math.round((this.camera.right-this.camera.left)/cell),h=Math.round((this.camera.top-this.camera.bottom)/cell);
    if(this.target.width!==w||this.target.height!==h)this.target.setSize(w,h);this.composite.uniforms.texel.value.set(1/w,1/h);
    const material=points.material,linesVisible=lines?.visible;
    points.material=this.accumulation;if(lines)lines.visible=false;
    renderer.getClearColor(this.clear);const alpha=renderer.getClearAlpha();
    renderer.setRenderTarget(this.target);renderer.setClearColor(0,0);renderer.clear();
    renderer.render(scene,this.camera);
    points.material=material;if(lines)lines.visible=linesVisible!;
    renderer.setRenderTarget(null);renderer.setClearColor(this.clear,alpha);
    renderer.render(this.scene,this.screenCamera);
  }
  setBlur(sigma:number){this.composite.uniforms.blur.value=Math.max(0,sigma);}
  dispose(){this.target.dispose();this.accumulation.dispose();this.composite.dispose();this.geometry.dispose();}
}
