import { defineConfig } from "vite";
import { Agent } from "node:http";
// The Genie world model (experiments/play_server.py) is a separate local process. /world is proxied to it so the
// page can fetch generated frames without cross-origin headers (the regex key keeps /world-test.html local); WORLD_MODEL_URL retargets it (for example a
// tunnelled GPU box: ssh -L 8008:localhost:8008 piano).
// Without an agent the proxy opens a NEW upstream socket per request. Against a local server that is free, but
// through an ssh tunnel every frame then pays a TCP handshake: measured 2026-09-17 against the A100 behind a
// 172 ms tunnel, one request cost 355 ms through the proxy versus 132 ms straight to the tunnel. Keep-alive
// reuses sockets, and maxSockets caps how deep a pipelined client can queue upstream.
const world = {
  target: process.env.WORLD_MODEL_URL ?? "http://127.0.0.1:8008",
  changeOrigin: true,
  agent: new Agent({ keepAlive: true, keepAliveMsecs: 10000, maxSockets: 8 }),
  rewrite: (path: string) => path.replace(/^\/world/, ""),
};
export default defineConfig({
  server: {
    watch: { usePolling: false, ignored: [/\/\.cache(?:\/|$)/] },
    proxy: { "^/world/": world },
  },
  preview: { proxy: { "^/world/": world } },
  build: {
    rolldownOptions: {
      input: {
        home: "index.html",
        viewer: "projection.html",
        embodied: "embodied.html",
        live: "live.html",
        play: "play.html",
        sites: "sites.html",
        worldTest: "world-test.html",
      },
    },
  },
});
