/** A local browser video decoder. Source pixels only enter the stimulation
 * controller; this module has no access to neural geometry or activity. */
export class LiveSource {
  readonly video=document.createElement('video');
  private objectUrl:string|null=null;
  private generation=0;
  private canvas=document.createElement('canvas');
  private context:CanvasRenderingContext2D;
  lastCapture:ImageData|null=null;
  private displayCanvas=document.createElement('canvas');
  frameSerial=0;
  mediaTime=0;
  decodedAt=0;
  constructor(readonly preview:HTMLCanvasElement,readonly width:number,readonly height:number,private aperture?:{x:number;y:number;width:number;height:number}){
    this.video.muted=true;this.video.loop=true;this.video.playsInline=true;this.video.preload='auto';
    this.canvas.width=width;this.canvas.height=height;this.context=this.canvas.getContext('2d',{willReadFrequently:true})!;
    const presented=(_now:number,meta:VideoFrameCallbackMetadata)=>{this.frameSerial++;this.mediaTime=meta.mediaTime;this.decodedAt=performance.now();this.video.requestVideoFrameCallback(presented);};
    this.video.requestVideoFrameCallback(presented);
  }
  async load(source:string|File){
    const request=++this.generation;
    this.video.pause();
    const old=this.objectUrl;this.objectUrl=source instanceof File?URL.createObjectURL(source):null;
    this.video.src=this.objectUrl??source as string;
    if(old)URL.revokeObjectURL(old);
    await new Promise<void>((resolve,reject)=>{
      const clean=()=>{this.video.removeEventListener('loadeddata',loaded);this.video.removeEventListener('error',error);clearTimeout(timeout);};
      const loaded=()=>{clean();resolve();};const error=()=>{clean();reject(Error('This browser could not decode the video. Try H.264 MP4 or VP9 WebM.'));};
      const timeout=setTimeout(()=>{clean();reject(Error('The video did not finish loading.'));},30000);
      this.video.addEventListener('loadeddata',loaded,{once:true});this.video.addEventListener('error',error,{once:true});this.video.load();
    });
    if(request!==this.generation)return false;
    if(!Number.isFinite(this.video.duration)||this.video.duration<=0)throw Error('Choose a video with a finite duration.');
    this.frameSerial++;this.mediaTime=this.video.currentTime;this.decodedAt=performance.now();this.capture();return true;
  }
  capture(showPreview=true){
    const ctx=this.context,w=this.width,h=this.height;ctx.fillStyle='#000';ctx.fillRect(0,0,w,h);
    const area=this.aperture??{x:0,y:0,width:w,height:h};
    const scale=Math.min(area.width/this.video.videoWidth,area.height/this.video.videoHeight),sw=this.video.videoWidth*scale,sh=this.video.videoHeight*scale;
    ctx.drawImage(this.video,area.x+(area.width-sw)/2,area.y+(area.height-sh)/2,sw,sh);
    this.lastCapture=ctx.getImageData(0,0,w,h);if(showPreview)this.present(this.lastCapture);
    const pixels=this.lastCapture.data,values=new Float32Array(w*h);
    for(let i=0;i<values.length;i++)values[i]=(.2126*pixels[4*i]+.7152*pixels[4*i+1]+.0722*pixels[4*i+2])/255;
    return values;
  }
  present(frame:ImageData){this.displayCanvas.width=frame.width;this.displayCanvas.height=frame.height;this.displayCanvas.getContext('2d')!.putImageData(frame,0,0);this.preview.getContext('2d')!.drawImage(this.displayCanvas,0,0,this.preview.width,this.preview.height);}
  dispose(){this.generation++;this.video.pause();this.video.removeAttribute('src');this.video.load();if(this.objectUrl)URL.revokeObjectURL(this.objectUrl);}
}
