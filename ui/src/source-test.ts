import {LiveSource} from './live-source';

const result=document.getElementById('result')!,button=document.getElementById('run') as HTMLButtonElement;
const delay=(ms:number)=>new Promise<void>(resolve=>setTimeout(resolve,ms));
function assert(value:boolean,message:string){if(!value)throw Error(message);}
async function canvasRecording(){
 const canvas=document.createElement('canvas');canvas.width=192;canvas.height=320;
 const context=canvas.getContext('2d')!,stream=canvas.captureStream(20);
 const mimeType=['video/webm;codecs=vp9','video/webm;codecs=vp8'].find(type=>MediaRecorder.isTypeSupported(type));
 if(!mimeType)throw Error('This browser cannot create the WebM decoder fixture.');
 const recorder=new MediaRecorder(stream,{mimeType}),chunks:Blob[]=[];
 const stopped=new Promise<void>((resolve,reject)=>{recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};recorder.onstop=()=>resolve();recorder.onerror=()=>reject(Error('Fixture recording failed.'));});
 recorder.start();
 try{
  for(let frame=0;frame<24;frame++){
   context.fillStyle='#183147';context.fillRect(0,0,192,320);
   context.fillStyle=frame%2?'#e7e9d0':'#edd035';context.fillRect(12+frame*5,35,42,235);
   context.fillStyle='white';context.font='30px sans-serif';context.fillText(String(frame),16,310);
   await delay(50);
  }
  recorder.stop();await stopped;
 }finally{stream.getTracks().forEach(track=>track.stop());}
 return new File(chunks,'recorded-portrait.webm',{type:mimeType});
}
button.onclick=async()=>{
 button.disabled=true;delete document.body.dataset.done;delete document.body.dataset.error;
 const source=new LiveSource(document.getElementById('preview') as HTMLCanvasElement,320,240),checks:object[]=[];
 try{
  result.textContent='Loading a File object made from the public evaluation clip…';
  const response=await fetch('/data/live-evaluation/visual-holdout-640.mp4');if(!response.ok)throw Error('Public MP4 fixture unavailable.');
  const file=new File([await response.blob()],'evaluation.mp4',{type:'video/mp4'});
  assert(await source.load(file),'MP4 File load was superseded');
  const first=source.capture(false);assert(first.length===76800&&first.every(Number.isFinite),'File decoder produced invalid controller input');
  await source.video.play();await delay(300);source.video.pause();
  assert(source.frameSerial>2&&source.mediaTime>0,'File playback did not produce new decoded frames');
  checks.push({kind:'MP4 File',width:source.video.videoWidth,height:source.video.videoHeight,duration:source.video.duration,decodedFrames:source.frameSerial});
  result.textContent='Creating a short portrait WebM with MediaRecorder…';
  const recording=await canvasRecording();
  assert(await source.load(recording),'WebM File load was superseded');
  assert(source.video.videoWidth===192&&source.video.videoHeight===320,'Portrait decoder dimensions differ');
  const values=source.capture(false);let outside=0,inside=0;
  for(let y=0;y<240;y++)for(let x=0;x<320;x++){if(x<88||x>=232)outside=Math.max(outside,values[y*320+x]);else inside=Math.max(inside,values[y*320+x]);}
  assert(outside===0&&inside>.1,'Portrait File was not letterboxed into the full control target');
  checks.push({kind:'MediaRecorder WebM File',width:192,height:320,duration:source.video.duration,letterboxOutside:outside,insideMaximum:inside});
  let rejected=false;try{await source.load(new File(['invalid'],'broken.mp4',{type:'video/mp4'}));}catch{rejected=true;}
  assert(rejected,'Undecodable file was accepted');assert(await source.load(file),'The decoder did not recover after a rejected file');
  checks.push({kind:'Invalid file and recovery',passed:true});
  result.textContent=JSON.stringify({status:'passed',scope:'Actual File -> object URL -> browser decoder -> controller input; excludes native picker and neural image quality',checks},null,2);document.body.dataset.done='true';
 }catch(error){result.textContent=JSON.stringify({status:'failed',checks,duration:String(source.video.duration),error:String(error)},null,2);document.body.dataset.error='true';}
 finally{source.dispose();button.disabled=false;}
};
