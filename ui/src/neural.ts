/** Event-driven, single-compartment LIF model using published Shiu parameters.
 * Synapses are weighted neuron-to-neuron connections from FlyWire, not invented
 * screen-neighbor links. State persists across input frames. All times are ms.
 */
export interface CircuitMetadata {
  neuronCount: number; edgeCount: number; anatomicalSynapseCount: number;
  inputs: { index: number; rootId: string; cellType: string; side: string; u: number; v: number }[];
  outputs: { index: number; rootId: string; cellType: string; side: string }[];
  renderMap: number[]; missingRenderedNeurons: number;
}

export class SpikingCircuit {
  readonly n: number;
  readonly edgeCount: number;
  readonly offsets: Uint32Array;
  readonly targets: Uint32Array;
  readonly weights: Int16Array;
  readonly voltage: Float64Array;
  readonly current: Float64Array;
  readonly refractoryUntil: Int32Array;
  readonly traces: Float64Array;
  readonly traceTime: Float64Array;
  private active: number[] = [];
  private present: Uint8Array;
  private optical: Uint8Array;
  private queue: number[][] = Array.from({ length: 10 }, () => []);
  private rngState = 783;
  private tick = 0;
  private decayV = Math.exp(-0.2/20);
  private decayG = Math.exp(-0.2/5);
  private contribution = 5/(5-20)*(Math.exp(-0.2/5)-Math.exp(-0.2/20));
  synapsesEnabled = true;
  respectOpticalRefractory = false;
  spikes = 0;
  deliveries = 0;
  readonly dt = 0.2;
  readonly traceMs:number;

  constructor(buffer: ArrayBuffer, traceMs=50) {
    if(!(traceMs>0))throw Error('Spike-trace time constant must be positive.');
    this.traceMs=traceMs;
    const header = new Uint32Array(buffer, 0, 3);
    if (header[0] !== 783) throw new Error('Unknown connectome data format.');
    this.n = header[1]; this.edgeCount = header[2];
    const targetOffset = 12 + (this.n+1)*4;
    const weightOffset = targetOffset + this.edgeCount*4;
    if (buffer.byteLength !== weightOffset+this.edgeCount*2) throw new Error('Incomplete connectome file.');
    this.offsets = new Uint32Array(buffer, 12, this.n+1);
    this.targets = new Uint32Array(buffer, targetOffset, this.edgeCount);
    this.weights = new Int16Array(buffer, weightOffset, this.edgeCount);
    this.voltage = new Float64Array(this.n).fill(-52);
    this.current = new Float64Array(this.n);
    this.refractoryUntil = new Int32Array(this.n);
    this.traces = new Float64Array(this.n);
    this.traceTime = new Float64Array(this.n);
    this.present = new Uint8Array(this.n);
    this.optical = new Uint8Array(this.n);
  }

  get timeMs() { return this.tick*this.dt; }
  get activeCount() { return this.active.length; }
  private random() {
    let x = this.rngState;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.rngState = x >>> 0;
    return this.rngState/4294967296;
  }
  private activate(id: number) {
    if (!this.present[id]) { this.present[id] = 1; this.active.push(id); }
  }
  stimulate(id: number) {
    if (id < 0 || id >= this.n) return;
    // Same strong external voltage kick as w_syn * f_poi in the source model.
    this.voltage[id] += 68.75;
    this.activate(id);
  }

  step(inputIds: Int32Array, inputRatesHz: Float32Array, pulseIds: number[] = []) {
    const now = this.timeMs;
    // Optical targets have no refractory period in the published protocol.
    for (let j=0; j<inputIds.length; j++) {
      this.optical[inputIds[j]]=1;
      if (this.random() < inputRatesHz[j]*this.dt/1000) {
        const id=inputIds[j];
        if(this.respectOpticalRefractory&&this.tick<this.refractoryUntil[id])continue;
        if(!this.respectOpticalRefractory)this.refractoryUntil[id]=0;
        this.stimulate(id);
      }
    }
    for (const id of pulseIds) { this.refractoryUntil[id]=0; this.stimulate(id); }
    const due = this.queue[this.tick%10];
    if (this.synapsesEnabled) {
      for (const pre of due) {
        for (let e=this.offsets[pre]; e<this.offsets[pre+1]; e++) {
          const w=this.weights[e];
          if (w===0) continue;
          const target=this.targets[e];
          this.current[target] += w*0.275;
          this.activate(target); this.deliveries++;
        }
      }
    }
    due.length=0;
    const scheduled=this.queue[(this.tick+9)%10]; // 9 × 0.2 ms = 1.8 ms delay.
    for (let j=this.active.length-1; j>=0; j--) {
      const i=this.active[j];
      if (this.tick >= this.refractoryUntil[i]) {
        const g=this.current[i];
        this.voltage[i]=-52+(this.voltage[i]+52)*this.decayV+g*this.contribution;
        this.current[i]=g*this.decayG;
        if (this.voltage[i]>-45) {
          this.voltage[i]=-52; this.current[i]=0;
          this.refractoryUntil[i]=this.tick+(!this.respectOpticalRefractory&&this.optical[i]?0:11);
          this.traces[i]=this.traces[i]*Math.exp(-(now-this.traceTime[i])/this.traceMs)+1;
          this.traceTime[i]=now;
          scheduled.push(i); this.spikes++;
        }
      }
      // Sleeping neurons are exactly passive except for a <1e-8 mV tail.
      if (Math.abs(this.voltage[i]+52)<1e-8 && Math.abs(this.current[i])<1e-8 && this.tick>=this.refractoryUntil[i]) {
        this.present[i]=0;
        this.active[j]=this.active[this.active.length-1]; this.active.pop();
      }
    }
    this.tick++;
  }

  rateHz(id: number) {
    if (id<0) return 0;
    return this.traces[id]*Math.exp(-(this.timeMs-this.traceTime[id])/this.traceMs)*(1000/this.traceMs);
  }
  renderActivity(mapping: number[]) {
    return Float32Array.from(mapping, (id) => 1-Math.exp(-this.rateHz(id)/65));
  }
  reset(seed=783) {
    this.rngState=seed; this.tick=0; this.spikes=0; this.deliveries=0;
    this.voltage.fill(-52); this.current.fill(0); this.refractoryUntil.fill(0);
    this.traces.fill(0); this.traceTime.fill(0); this.present.fill(0);this.optical.fill(0);
    this.active.length=0; this.queue.forEach((q)=>{q.length=0;});
  }
  setOpticalTargets(ids:Int32Array){this.optical.fill(0);for(const id of ids)this.optical[id]=1;}
  snapshot() {
    return {tick:this.tick,rng:this.rngState,spikes:this.spikes,deliveries:this.deliveries,
      voltage:this.voltage.slice(),current:this.current.slice(),refractory:this.refractoryUntil.slice(),
      traces:this.traces.slice(),traceTime:this.traceTime.slice(),present:this.present.slice(),
      optical:this.optical.slice(),active:this.active.slice(),queue:this.queue.map(q=>q.slice())};
  }
  restore(s: ReturnType<SpikingCircuit['snapshot']>) {
    this.tick=s.tick;this.rngState=s.rng;this.spikes=s.spikes;this.deliveries=s.deliveries;
    this.voltage.set(s.voltage);this.current.set(s.current);this.refractoryUntil.set(s.refractory);
    this.traces.set(s.traces);this.traceTime.set(s.traceTime);this.present.set(s.present);this.optical.set(s.optical);
    this.active=s.active.slice();this.queue=s.queue.map(q=>q.slice());
  }
}

export function motorReadout(meta: CircuitMetadata, circuit: Pick<SpikingCircuit,'rateHz'>) {
  const rates=meta.outputs.map((n)=>({...n, hz:circuit.rateHz(n.index)}));
  const mean=(type:string,side?:string)=>{
    const group=rates.filter((n)=>n.cellType===type && (!side || n.side===side));
    return group.reduce((s,n)=>s+n.hz,0)/Math.max(1,group.length);
  };
  // An explicit engineered DN-to-CPG interface, not reconstructed VNC wiring.
  const forward=1-Math.exp(-mean('DNp09')/35);
  const backward=1-Math.exp(-mean('MDN')/35);
  const turn=Math.tanh((mean('DNa02','left')-mean('DNa02','right'))/45);
  const drive=forward-backward;
  const left=Math.max(-1.2,Math.min(1.2,drive-0.55*turn));
  const right=Math.max(-1.2,Math.min(1.2,drive+0.55*turn));
  return {left,right,forward,backward,turn,rates};
}
