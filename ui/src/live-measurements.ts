/** Measurements start only after the neural frame's GPU draw has completed. */
export interface CompletedFrame {
  at:number;
  source:number;
  serial:number;
  segment:number;
  mode:'paired'|'enlarged';
  latency:number;
  decodedLatency:number;
  compute:number;
  body:number;
  frame:number;
}

export function summarizeFrames(samples:CompletedFrame[],segment:number,limit=120){
  // Pause/resume, source changes, interventions and view changes split windows.
  // Neural history is independent of measurement history and remains intact.
  const frames=samples.filter(frame=>frame.segment===segment).slice(-limit);
  const intervals=frames.slice(1).map((frame,i)=>frame.at-frames[i].at);
  const mean=(values:number[])=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
  const p95=(values:number[])=>values.length?[...values].sort((a,b)=>a-b)[Math.ceil(values.length*.95)-1]:null;
  const elapsed=frames.length>1?frames.at(-1)!.at-frames[0].at:0;
  const distinctDecodedFrames=new Set(frames.map(frame=>frame.serial)).size;
  const distinctSourceTimes=new Set(frames.map(frame=>frame.source)).size;
  const distinctOutputs=new Set(frames.map(frame=>frame.frame)).size;
  const valid=intervals.every(value=>value>0)&&distinctOutputs===frames.length&&distinctDecodedFrames===frames.length;
  return {
    frames:frames.length,distinctDecodedFrames,distinctSourceTimes,
    completeWindow:frames.length===limit&&valid,
    newFramesPerSecond:elapsed>0&&valid?(frames.length-1)*1000/elapsed:null,
    meanLatencyMs:mean(frames.map(frame=>frame.latency)),p95LatencyMs:p95(frames.map(frame=>frame.latency)),
    meanDecodedLatencyMs:mean(frames.map(frame=>frame.decodedLatency)),p95DecodedLatencyMs:p95(frames.map(frame=>frame.decodedLatency)),
    p95FrameIntervalMs:p95(intervals),meanNeuralMs:mean(frames.map(frame=>frame.compute)),
    meanBodyMs:mean(frames.map(frame=>frame.body)),samples:frames,
  };
}
