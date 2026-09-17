/** Development check for the world-model stimulus path without the brain: the play server's generated frames
 * go through WorldModelClient, a canvas MediaStream and LiveSource.capture(), exactly as on the live page.
 * Counters are exposed as data attributes for tests/world-model.spec.ts. */
import {LiveSource} from './live-source';
import {WorldModelClient,parseKeys} from './world-model';

const app=document.getElementById('world-test')!,query=new URL(location.href).searchParams,base=import.meta.env.BASE_URL;
app.innerHTML=`<h1>World model source test</h1><p id="status">Connecting…</p>
<div style="display:flex;gap:16px;flex-wrap:wrap"><div><h2>Generated frame</h2><div id="generated"></div></div>
<div><h2>LiveSource capture · 320 × 240 control raster</h2><canvas id="preview" width="320" height="240" style="image-rendering:pixelated;border:1px solid #888"></canvas></div></div>
<p id="counts"></p><p>Hold A / D to turn, W / S for the no-turn codes, 0–7 raw codes, R new prompt, Q 3 / 25 passes, [ ] passes.</p>`;
const status=document.getElementById('status')!,counts=document.getElementById('counts')!;
const client=new WorldModelClient({url:query.get('world')??`${base}world`,keys:parseKeys(query.get('keys'))??undefined,steps:query.has('steps')?Number(query.get('steps')):undefined,idleKey:query.get('idle')==='none'?null:query.get('idle')??'W'});
client.canvas.style.cssText='border:1px solid #888;image-rendering:pixelated';document.getElementById('generated')!.append(client.canvas);
client.onStatus=text=>{status.textContent=text;};
client.onError=error=>{app.dataset.error=error.message;status.textContent=error.message;};
client.onStep=step=>{app.dataset.label=step.label;app.dataset.code=String(step.code);};
const source=new LiveSource(document.getElementById('preview') as HTMLCanvasElement,320,240);
let captures=0,lastSerial=-1,maximum=0;
(async()=>{
  try{
    await client.connect();
    if(!await source.load(client))throw Error('Source load was superseded.');
    await source.play();
    client.attachKeyboard();client.setRunning(true);
    app.dataset.ready='true';
    const loop=()=>{
      if(source.frameSerial!==lastSerial&&source.ready){
        lastSerial=source.frameSerial;const values=source.capture(true);captures++;
        let peak=0;for(let i=0;i<values.length;i++)if(values[i]>peak)peak=values[i];maximum=Math.max(maximum,peak);
      }
      app.dataset.generated=String(client.frames);app.dataset.serial=String(source.frameSerial);app.dataset.captures=String(captures);app.dataset.maximum=maximum.toFixed(3);
      counts.textContent=`${client.frames} generated · ${source.frameSerial} presented · ${captures} captured · brightest control pixel ${maximum.toFixed(2)} · ${source.sourceWidth} × ${source.sourceHeight} generated frames`;
      requestAnimationFrame(loop);
    };
    loop();
  }catch(error){app.dataset.error=error instanceof Error?error.message:String(error);status.textContent=app.dataset.error;}
})();
