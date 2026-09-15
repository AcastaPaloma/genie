import {defineConfig} from 'vite';
export default defineConfig({server:{watch:{usePolling:false,ignored:[/\/\.cache(?:\/|$)/]}},build:{rolldownOptions:{input:{home:'index.html',viewer:'projection.html',embodied:'embodied.html',live:'live.html',sites:'sites.html'}}}});
