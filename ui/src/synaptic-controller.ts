import type {SpikingCircuit} from './neural';

export interface SynapticControlMetadata {
  inputIds:number[];outputIds:number[];row:number[];column:number[];weight:number[];strength:number[];
}

/** Sparse bounded coordinate descent. Values are control variables, never
 * observations. The live renderer receives separately measured spike traces.
 */
class BoundedLinearSolver {
  readonly values:Float64Array;
  readonly residual:Float64Array;
  private pointers:Uint32Array;
  private rows:Uint32Array;
  private weights:Float64Array;
  private norm:Float64Array;
  private ridge:number;
  constructor(row:ArrayLike<number>,column:ArrayLike<number>,weight:ArrayLike<number>,n:number,p:number,ridge=1e-6){
    this.ridge=ridge;
    this.values=new Float64Array(n);this.residual=new Float64Array(p);
    this.pointers=new Uint32Array(n+1);
    for(let e=0;e<column.length;e++)this.pointers[column[e]+1]++;
    for(let c=0;c<n;c++)this.pointers[c+1]+=this.pointers[c];
    const cursor=this.pointers.slice();this.rows=new Uint32Array(row.length);this.weights=new Float64Array(row.length);this.norm=new Float64Array(n).fill(ridge);
    for(let e=0;e<row.length;e++){const c=column[e],i=cursor[c]++;this.rows[i]=row[e];this.weights[i]=weight[e];this.norm[c]+=weight[e]**2;}
  }
  project(){
    const y=new Float64Array(this.residual.length);
    for(let c=0;c<this.values.length;c++)for(let e=this.pointers[c];e<this.pointers[c+1];e++)y[this.rows[e]]+=this.weights[e]*this.values[c];
    return y;
  }
  solve(target:ArrayLike<number>,upper:ArrayLike<number>,iterations:number){
    const projected=this.project();
    for(let p=0;p<this.residual.length;p++)this.residual[p]=projected[p]-target[p];
    for(let iter=0;iter<iterations;iter++)for(let c=0;c<this.values.length;c++){
      let grad=this.ridge*this.values[c];
      for(let e=this.pointers[c];e<this.pointers[c+1];e++)grad+=this.weights[e]*this.residual[this.rows[e]];
      const next=Math.max(0,Math.min(upper[c],this.values[c]-grad/this.norm[c])),delta=next-this.values[c];
      this.values[c]=next;
      for(let e=this.pointers[c];e<this.pointers[c+1];e++)this.residual[this.rows[e]]+=this.weights[e]*delta;
    }
  }
}

/** Feedback optical controller using actual incoming FlyWire connections.
 * Displayed and motor-output cells are excluded from its stimulation population.
 */
export class SynapticImageController {
  readonly inputIds:Int32Array;
  readonly outputIds:Int32Array;
  readonly inputRates:Float32Array;
  readonly rateScale=60;
  readonly maxInputHz=4000;
  private incoming:BoundedLinearSolver;
  private image:BoundedLinearSolver;
  private brightnessUpper:Float64Array;
  private currentUpper:Float64Array;
  private demand:Float64Array;
  private integral:Float64Array;
  private strength:Float64Array;
  private lastControlTime=-Infinity;
  private frameTarget=new Float64Array(1200);
  fitMse=0;

  constructor(meta:SynapticControlMetadata,operator:ArrayBuffer,motorIds:number[]=[]){
    this.inputIds=Int32Array.from(meta.inputIds);this.outputIds=Int32Array.from(meta.outputIds);
    const forbidden=new Set([...this.outputIds,...motorIds]);
    if(this.inputIds.some(id=>forbidden.has(id)))throw Error('Displayed or motor-output neurons cannot be directly stimulated by the image controller.');
    this.inputRates=new Float32Array(this.inputIds.length);
    this.strength=Float64Array.from(meta.strength);
    this.incoming=new BoundedLinearSolver(meta.row,meta.column,meta.weight,this.inputIds.length,this.outputIds.length,.0001);
    this.currentUpper=Float64Array.from(this.strength,w=>this.maxInputHz*.001375*w);
    this.demand=new Float64Array(this.outputIds.length);this.integral=new Float64Array(this.outputIds.length);
    const [p,n,e]=new Uint32Array(operator,0,3);
    if(p!==1200||n!==this.outputIds.length||operator.byteLength!==12+8*e)throw Error('Invalid anatomical image operator.');
    this.image=new BoundedLinearSolver(new Uint16Array(operator,12,e),new Uint16Array(operator,12+2*e,e),new Float32Array(operator,12+4*e,e),n,p);
    this.brightnessUpper=new Float64Array(n).fill(1);
  }

  setFrame(luminance:ArrayLike<number>){
    if(luminance.length!==4800)throw Error('Expected an 80 × 60 stimulus.');
    for(let y=0;y<30;y++)for(let x=0;x<40;x++){
      const p=y*2*80+x*2;this.frameTarget[y*40+x]=(luminance[p]+luminance[p+1]+luminance[p+80]+luminance[p+81])/4;
    }
    this.image.solve(this.frameTarget,this.brightnessUpper,80);
    const prediction=this.image.project();this.fitMse=prediction.reduce((s,x,i)=>s+(x-this.frameTarget[i])**2,0)/prediction.length;
  }

  update(circuit:SpikingCircuit,muted=false){
    if(muted){this.inputRates.fill(0);this.lastControlTime=circuit.timeMs;return;}
    if(circuit.timeMs-this.lastControlTime<5-1e-6)return;
    const initial=!Number.isFinite(this.lastControlTime),elapsed=initial ? .005 : (circuit.timeMs-this.lastControlTime)/1000;
    this.lastControlTime=circuit.timeMs;
    for(let j=0;j<this.outputIds.length;j++){
      const desired=this.rateScale*this.image.values[j],actual=circuit.rateHz(this.outputIds[j]);
      const time=desired>0?Math.max(.1,1000/desired-2.2):1e6;
      const feedforward=desired>0?7/(1-20/15*Math.exp(-time/20)+5/15*Math.exp(-time/5)):0;
      const error=desired-actual;
      this.integral[j]=Math.max(-250,Math.min(250,this.integral[j]+error*elapsed*4));
      this.demand[j]=Math.max(-300,Math.min(200,feedforward+this.integral[j]+.5*error));
    }
    this.incoming.solve(this.demand,this.currentUpper,initial?30:3);
    for(let i=0;i<this.inputRates.length;i++)this.inputRates[i]=this.incoming.values[i]/(.001375*this.strength[i]);
  }

  measuredActivity(circuit:SpikingCircuit){return Float32Array.from(this.outputIds,id=>Math.min(1,circuit.rateHz(id)/this.rateScale));}
  reset(){this.incoming.values.fill(0);this.inputRates.fill(0);this.integral.fill(0);this.lastControlTime=-Infinity;}
}
