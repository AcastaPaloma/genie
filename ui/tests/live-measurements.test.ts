import test from 'node:test';
import assert from 'node:assert/strict';
import {summarizeFrames,type CompletedFrame} from '../src/live-measurements.ts';

const frames=(count:number,segment=0):CompletedFrame[]=>Array.from({length:count},(_,i)=>({
  at:i*100,source:i/10,serial:i+1,segment,mode:'paired',latency:60,decodedLatency:75,compute:40,body:30,frame:i+1,
}));

test('counts completed source frames over N−1 intervals and exports raw evidence',()=>{
  const result=summarizeFrames(frames(120),0);
  assert.equal(result.completeWindow,true);
  assert.equal(result.newFramesPerSecond,10);
  assert.equal(result.distinctDecodedFrames,120);
  assert.equal(result.p95FrameIntervalMs,100);
  assert.equal(result.meanDecodedLatencyMs,75);
  assert.equal(result.samples.length,120);
});

test('never claims throughput for duplicate decoded frames or duplicate draws',()=>{
  for(const property of ['serial','frame'] as const){
    const samples=frames(120);samples[119][property]=samples[118][property];
    const result=summarizeFrames(samples,0);
    assert.equal(result.completeWindow,false);
    assert.equal(result.newFramesPerSecond,null);
  }
});

test('a new playback, intervention or view segment excludes the previous window',()=>{
  const samples=[...frames(120),...frames(3,1).map(frame=>({...frame,at:frame.at+60000}))];
  const result=summarizeFrames(samples,1);
  assert.equal(result.frames,3);
  assert.equal(result.completeWindow,false);
  assert.equal(result.newFramesPerSecond,10);
  const empty=summarizeFrames(samples,2);
  assert.equal(empty.newFramesPerSecond,null);
  assert.equal(empty.meanLatencyMs,null);
});
