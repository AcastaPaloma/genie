import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {SpikingCircuit,motorReadout} from '../src/neural.ts';

function tiny(weight=800){
  const data=new ArrayBuffer(12+4*4+2*4+2*2);
  new Uint32Array(data,0,3).set([783,3,2]);
  new Uint32Array(data,12,4).set([0,1,2,2]);
  new Uint32Array(data,28,2).set([1,2]);
  new Int16Array(data,36,2).set([weight,weight]);
  return new SpikingCircuit(data);
}
const emptyIds=new Int32Array(),emptyRates=new Float32Array();
test('spikes cross two real weighted edges after synaptic delay',()=>{
  const c=tiny();c.stimulate(0);c.step(emptyIds,emptyRates);
  assert.equal(c.spikes,1);assert.equal(c.deliveries,0);assert.equal(c.rateHz(1),0);
  for(let i=0;i<8;i++)c.step(emptyIds,emptyRates);
  assert.equal(c.deliveries,0,'1.8 ms delay has not elapsed');
  for(let i=0;i<50;i++)c.step(emptyIds,emptyRates);
  assert.ok(c.rateHz(1)>0&&c.rateHz(2)>0);
  assert.equal(c.deliveries,2);
});
test('inhibition and synaptic ablation prevent downstream firing',()=>{
  for(const c of [tiny(-800),Object.assign(tiny(),{synapsesEnabled:false})]){
    c.stimulate(0);for(let i=0;i<100;i++)c.step(emptyIds,emptyRates);
    assert.equal(c.rateHz(1),0);assert.equal(c.rateHz(2),0);
  }
});
test('published full graph recruits motor cells; cutting synapses removes the response',()=>{
  const meta=JSON.parse(fs.readFileSync('public/data/connectome.json','utf8'));
  const raw=gunzipSync(fs.readFileSync('public/data/connectome.bin.gz'));
  const c=new SpikingCircuit(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  const ids=Int32Array.from(meta.inputs,(n:{index:number})=>n.index),rates=new Float32Array(ids.length).fill(90);
  assert.equal(c.n,138639);assert.equal(c.edgeCount,15091983);
  for(let i=0;i<1000;i++)c.step(ids,rates);
  const original=c.renderActivity(meta.renderMap);
  const motor=motorReadout(meta,c);
  assert.ok(c.deliveries>1e6);
  assert.ok(motor.rates.some(n=>n.cellType==='DNa02'&&n.hz>10));
  assert.ok(Math.abs(motor.left)+Math.abs(motor.right)>0);
  c.reset();
  for(let i=0;i<1000;i++)c.step(ids,rates);
  assert.deepEqual(c.renderActivity(meta.renderMap),original,'seeded reset must replay the same spikes');
  c.reset();c.synapsesEnabled=false;
  for(let i=0;i<1000;i++)c.step(ids,rates);
  assert.ok(c.spikes>0,'external stimulation still spikes input cells');
  assert.equal(c.deliveries,0);
  assert.ok(motorReadout(meta,c).rates.every(n=>n.hz===0));
  assert.equal(motorReadout(meta,c).left,0);assert.equal(motorReadout(meta,c).right,0);
});
