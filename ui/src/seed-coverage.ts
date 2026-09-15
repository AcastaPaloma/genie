/** Target-independent greedy seed selection. Every seed is an existing,
 * excitatory presynaptic neuron and its complete arbor remains displayed.
 * The objective spreads measured excitatory contacts across the anatomical
 * population instead of repeatedly selecting inputs to the same few cells.
 */
export function coverageSeeds(graph:{n:number;offsets:Uint32Array;targets:Uint32Array;weights:Int16Array},candidates:number[],importance:Float64Array,count:number){
 const coverage=new Float64Array(graph.n),selected:number[]=[];
 type Candidate={id:number;score:number};const heap:Candidate[]=[];
 const score=(id:number)=>{
  let total=0;for(let e=graph.offsets[id];e<graph.offsets[id+1];e++){
   const post=graph.targets[e],weight=Math.max(0,graph.weights[e]);
   total+=Math.sqrt(importance[post])*Math.log1p(weight/(12+coverage[post]));
  }return total/Math.sqrt(Math.max(.05,importance[id]));
 };
 const push=(value:Candidate)=>{let i=heap.length;heap.push(value);while(i){const parent=(i-1)>>1;if(heap[parent].score>=value.score)break;heap[i]=heap[parent];i=parent;}heap[i]=value;};
 const pop=()=>{const first=heap[0],last=heap.pop()!;if(heap.length){let i=0;while(2*i+1<heap.length){let child=2*i+1;if(child+1<heap.length&&heap[child+1].score>heap[child].score)child++;if(heap[child].score<=last.score)break;heap[i]=heap[child];i=child;}heap[i]=last;}return first;};
 for(const id of candidates)push({id,score:score(id)});
 while(heap.length&&selected.length<count){
  const item=pop();item.score=score(item.id);
  if(heap.length&&item.score<heap[0].score-1e-10){push(item);continue;}
  selected.push(item.id);for(let e=graph.offsets[item.id];e<graph.offsets[item.id+1];e++)coverage[graph.targets[e]]+=Math.max(0,graph.weights[e]);
 }
 return Uint32Array.from(selected);
}
