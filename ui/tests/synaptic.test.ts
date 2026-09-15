import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {SpikingCircuit} from '../src/neural.ts';
import {SynapticImageController} from '../src/synaptic-controller.ts';

test('image control uses real incoming connections, improves a frame, and fails when those connections are removed or shuffled',()=>{
  const meta=JSON.parse(fs.readFileSync('public/data/presynaptic-controller.json','utf8'));
  const network=JSON.parse(fs.readFileSync('public/data/connectome.json','utf8'));
  const raw=gunzipSync(fs.readFileSync('public/data/connectome.bin.gz'));
  const circuit=new SpikingCircuit(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength),100);
  const ob=fs.readFileSync('public/data/synaptic-image-operator.bin');
  const operator=ob.buffer.slice(ob.byteOffset,ob.byteOffset+ob.byteLength);
  const controller=new SynapticImageController(meta,operator,network.outputs.map((n:{index:number})=>n.index));
  const rowById=new Map<number,number>(meta.outputIds.map((id:number,i:number)=>[id,i]));
  const real=new Map<string,number>();
  for(let input=0;input<meta.inputIds.length;input++){
    const id=meta.inputIds[input];assert.ok(!rowById.has(id));
    for(let e=circuit.offsets[id];e<circuit.offsets[id+1];e++){
      const row=rowById.get(circuit.targets[e]);if(row===undefined)continue;
      const key=`${input}/${row}`;real.set(key,(real.get(key)??0)+circuit.weights[e]);
    }
  }
  for(let e=0;e<meta.row.length;e++)assert.ok(Math.abs(meta.weight[e]*meta.strength[meta.column[e]]-(real.get(`${meta.column[e]}/${meta.row[e]}`)??0))<1e-6,'Controller coefficient must match measured signed contact counts.');
  const frame=JSON.parse(fs.readFileSync('public/data/synaptic-control-validation.json','utf8')).runs[0].frames[8];
  const target=frame.target as number[];
  const luma=new Float32Array(4800);for(let y=0;y<60;y++)for(let x=0;x<80;x++)luma[y*80+x]=target[Math.floor(y/2)*40+Math.floor(x/2)];
  const [P,,E]=new Uint32Array(operator,0,3),pixel=new Uint16Array(operator,12,E),neuron=new Uint16Array(operator,12+2*E,E),weight=new Float32Array(operator,12+4*E,E);
  function trial(){
    circuit.reset();controller.reset();controller.setFrame(luma);
    for(let i=0;i<2500;i++){controller.update(circuit);circuit.step(controller.inputIds,controller.inputRates);}
    const activity=controller.measuredActivity(circuit),prediction=new Float64Array(P);
    for(let e=0;e<E;e++)prediction[pixel[e]]+=weight[e]*activity[neuron[e]];
    return {activity,mse:prediction.reduce((s,x,p)=>s+(x-target[p])**2,0)/P};
  }
  const intact=trial(),mean=target.reduce((s,x)=>s+x,0)/P;
  const flatMse=target.reduce((s,x)=>s+(x-mean)**2,0)/P;
  assert.ok(intact.mse<flatMse*.8,`Actual spikes should improve the frame over flat brightness: ${intact.mse} vs ${flatMse}.`);
  circuit.synapsesEnabled=false;
  const disconnected=trial();assert.equal(Math.max(...disconnected.activity),0);assert.equal(circuit.deliveries,0);assert.ok(circuit.spikes>0);
  circuit.synapsesEnabled=true;
  for(let e=0;e<circuit.targets.length;e++){
    const row=rowById.get(circuit.targets[e]);if(row!==undefined)circuit.targets[e]=meta.outputIds[(row+523)%meta.outputIds.length];
  }
  const shuffled=trial();
  assert.ok(shuffled.mse>intact.mse*1.5,`Reassigning real incoming targets should damage the image: ${shuffled.mse} vs ${intact.mse}.`);
  console.log(JSON.stringify({intactMse:intact.mse,flatMse,disconnectedMse:disconnected.mse,shuffledMse:shuffled.mse}));
});
