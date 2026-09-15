import {chromium} from '@playwright/test';import fs from 'node:fs';
const browser=await chromium.launch({headless:true});
try{const page=await browser.newPage();await page.goto('http://127.0.0.1:5173/body-test.html');const records=await page.evaluate(async()=>{
 const runs=[];for(const dt of [.0001,.00025]){
  const worker=new Worker('/src/body.worker.ts',{type:'module'});const message=()=>new Promise((resolve,reject)=>{worker.onmessage=e=>e.data.type==='error'?reject(Error(e.data.message)):resolve(e.data);worker.onerror=e=>reject(Error(e.message));});
  let pending=message();worker.postMessage({type:'init',runtime:location.origin+'/body/runtime.js',assets:'/body/assets',timestep:dt});await pending;
  const rows=[];for(let i=0;i<300;i++){const phase=i%100,left=phase<30?1:phase<50?-.7:phase<75?1.2:.15,right=phase<30?1:phase<50?-.7:phase<75?.15:1.2;pending=message();worker.postMessage({type:'step',epoch:0,ms:100,left,right});const state=await pending;rows.push({time:state.time,computeMs:state.computeMs,qpos:Array.from(state.qpos)});}worker.terminate();runs.push({dt,rows});
 }return runs;
});fs.writeFileSync('.cache/large-control/body-timestep-validation.json',JSON.stringify(records));console.log(JSON.stringify(records.map(r=>({dt:r.dt,steps:r.rows.length,time:r.rows.at(-1).time,meanCompute:r.rows.reduce((s,x)=>s+x.computeMs,0)/r.rows.length,zRange:[Math.min(...r.rows.map(x=>x.qpos[2])),Math.max(...r.rows.map(x=>x.qpos[2]))],finite:r.rows.every(x=>x.qpos.every(Number.isFinite))}))));}finally{await browser.close();}
