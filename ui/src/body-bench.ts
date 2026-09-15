const output=document.getElementById('result')!;
const rows:{dt:number;meanComputeMs:number;minZ:number;maxZ:number;finite:boolean;finalTime:number;finalPosition:number[]}[]=[];
try{
 for(const dt of [.00025,.0005]){
  const worker=new Worker(new URL('./body.worker.ts',import.meta.url),{type:'module'});
  const send=(message:object)=>new Promise<any>((resolve,reject)=>{worker.onmessage=e=>e.data.type==='error'?reject(Error(e.data.message)):resolve(e.data);worker.onerror=e=>reject(Error(e.message));worker.postMessage(message);});
  await send({type:'init',runtime:new URL('/body/runtime.js',location.href).href,assets:'/body/assets',timestep:dt});
  let elapsed=0,minZ=Infinity,maxZ=-Infinity,last:any;
  for(let i=0;i<300;i++){
   const phase=i%100,left=phase<30?1:phase<50?-.7:phase<75?1.2:.15,right=phase<30?1:phase<50?-.7:phase<75?.15:1.2;
   last=await send({type:'step',epoch:0,ms:100,left,right});elapsed+=last.computeMs;minZ=Math.min(minZ,last.qpos[2]);maxZ=Math.max(maxZ,last.qpos[2]);
   output.textContent=JSON.stringify({progress:`${dt}: ${i+1}/300`,computeMs:last.computeMs,z:last.qpos[2],completed:rows});
  }
  rows.push({dt,meanComputeMs:elapsed/300,minZ,maxZ,finite:true,finalTime:last.time,finalPosition:Array.from(last.qpos.slice(0,7))});worker.terminate();
 }
 output.textContent=JSON.stringify({complete:true,rows});document.body.dataset.done='true';
}catch(error){output.textContent=String(error);document.body.dataset.error='true';}
export {};
