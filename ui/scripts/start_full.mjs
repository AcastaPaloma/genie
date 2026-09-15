import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
let physics;
try{const r=await fetch('http://127.0.0.1:8769/info',{signal:AbortSignal.timeout(500)});if(!r.ok)throw Error('Native body unavailable');}
catch{
 physics=spawn(process.env.NEUROFRAME_PYTHON??path.resolve(root,'../.venv/bin/python'),[path.join(root,'scripts/native_body.py')],{cwd:root,stdio:'inherit'});
 physics.on('error',error=>{console.error('Install the native body dependencies with: ../.venv/bin/python -m pip install -r requirements-body.txt');console.error(error.message);process.exit(1);});
 let ready=false;
 for(let attempt=0;attempt<100;attempt++){
  if(physics.exitCode!==null)throw Error('Native body exited. Install requirements-body.txt before starting the full application.');
  try{const r=await fetch('http://127.0.0.1:8769/info',{signal:AbortSignal.timeout(300)});if(r.ok){ready=true;break;}}catch{}
  await new Promise(resolve=>setTimeout(resolve,150));
 }
 if(!ready){physics.kill();throw Error('Native body did not become ready.');}
}
const vite=spawn(process.execPath,[path.join(root,'node_modules/vite/bin/vite.js'),'--host','127.0.0.1','--port','5173','--strictPort'],{cwd:root,stdio:'inherit'});
const close=()=>{vite.kill();physics?.kill();};
process.on('SIGINT',close);process.on('SIGTERM',close);vite.on('exit',code=>{physics?.kill();process.exitCode=code??0;});
