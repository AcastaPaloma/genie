/** A versioned scientific configuration, shared by the live UI and simulator. */
export interface LiveProfile {
  id:string;
  anatomyDirectory:string;
  view:string;
  operator:string;
  controller:string;
  observationPoints:string;
  parameters:{rateScale:number;traceMs:number;feedbackTicks:number;initialIterations:number;maintenance:number;gradientLimit:number;inputRatePenalty:number;inputMode:2;compensate:true;robust:true};
  bodyTimestep:number;
  disclosure:string;
}
