import {defineConfig} from 'vite';
// The Genie world model (experiments/play_server.py) is a separate local process. /world is proxied to it so the
// page can fetch generated frames without cross-origin headers (the regex key keeps /world-test.html local); WORLD_MODEL_URL retargets it (for example a
// tunnelled GPU box: ssh -L 8008:localhost:8008 piano).
const world={target:process.env.WORLD_MODEL_URL??'http://127.0.0.1:8008',changeOrigin:true,rewrite:(path:string)=>path.replace(/^\/world/,'')};
export default defineConfig({server:{watch:{usePolling:false,ignored:[/\/\.cache(?:\/|$)/]},proxy:{'^/world/':world}},preview:{proxy:{'^/world/':world}},build:{rolldownOptions:{input:{home:'index.html',viewer:'projection.html',embodied:'embodied.html',live:'live.html',play:'play.html',sites:'sites.html',worldTest:'world-test.html'}}}});
