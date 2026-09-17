// Diagnostic: does the front view render from the complete cable cache alone,
// with no 3D skeleton chunks uploaded? Every neuron is still represented: the
// cache integrates all 268,139,177 released branches.
import {FullBrainView,type FullBrainMetadata} from './full-brain';
const out=document.getElementById('result')!;
const directory='/data/full-brain-783/';
try{
  const meta:FullBrainMetadata=await (await fetch(directory+'brain.json')).json();
  const view=new FullBrainView(document.getElementById('brain')!,meta);
  const t=performance.now();
  await view.loadLightCache(directory);
  const cacheSeconds=+((performance.now()-t)/1000).toFixed(1);
  // No view.load(...) at all: zero skeleton chunks uploaded.
  const activity=new Float32Array(meta.neuronCount);
  for(let i=0;i<activity.length;i++)activity[i]=(i%97)/97;
  await view.prepareNeuralFrame(activity);
  view.render(performance.now(),0,false);
  await view.waitForDraw();
  const info=view.getInfo();
  out.textContent=JSON.stringify({cacheSeconds,drawMode:info.drawMode,chunksUploaded:info.chunks,
    verticesUploaded:info.vertices,branchesIntegratedByCache:info.cache?.branches,
    neuronsIntegratedByCache:meta.neuronCount,cacheComputeMs:+(info.cache?.computeMs??0).toFixed(1),
    drawCalls:info.drawCalls},null,1);
  view.dispose();document.body.dataset.done='true';
}catch(error){out.textContent=String(error);document.body.dataset.error='true';}
