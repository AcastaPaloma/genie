/** Standalone bench for the body's air mode: the NeuroMechFly body, the same physics worker play.html uses,
 * and a key to cut gravity. No brain and no world model, so the flail can be tuned and screenshotted without
 * taking over a play session. G toggles air, R resets, [ and ] change the walking drive. */
import { FlyBodyView } from "./body";

const base = import.meta.env.BASE_URL,
  app = document.getElementById("fly-air-test")!;
app.innerHTML = `<h1>Fly air mode</h1><p id="status">Loading the body…</p>
<div id="fly" style="width:min(100%,900px);height:520px;border:1px solid #444"></div>
<p><kbd>G</kbd> air on/off · <kbd>R</kbd> reset · <kbd>[</kbd> <kbd>]</kbd> drive · the legs keep walking whatever the gravity is</p>`;
const status = document.getElementById("status")!;
const body = new FlyBodyView(document.getElementById("fly")!);
const physics = new Worker(new URL("./body.worker.ts", import.meta.url), {
  type: "module",
});

let epoch = 0,
  busy = false,
  drive = 0.8,
  airborne = false,
  steps = 0,
  last = performance.now(),
  height = 0,
  backend = "";
const describe = () => {
  status.textContent = `${backend} · ${airborne ? "AIR (gravity off, hovering)" : "ground (gravity 100%)"} · drive ${drive.toFixed(2)} · height ${height.toFixed(2)} mm · ${steps} body steps`;
  app.dataset.air = String(airborne);
  app.dataset.height = height.toFixed(3);
  app.dataset.steps = String(steps);
};

physics.onmessage = (event) => {
  const m = event.data;
  if (m.type === "error") {
    status.textContent = m.message;
    app.dataset.error = m.message;
    busy = false;
    return;
  }
  if (m.type === "ready") {
    backend = m.backend;
    app.dataset.ready = "true";
    describe();
    return;
  }
  if (m.type === "air") {
    airborne = m.on;
    if (!m.supported) {
      app.dataset.error = m.message;
      status.textContent = m.message;
    }
    describe();
    return;
  }
  busy = false;
  if (m.epoch !== epoch) return;
  body.applyPhysicalState(m);
  body.render();
  height = m.qpos[2];
  steps++;
  describe();
};
physics.onerror = (event) => {
  app.dataset.error = event.message;
  status.textContent = event.message;
};

await body.load((text) => {
  status.textContent = text;
});
// The view keeps its own MuJoCo instance for rendering; expose it so the gravity API can be probed from a
// test harness without another edit cycle. Bench page only.
(globalThis as any).bench = body as any;
// No native backend here: air mode needs the WASM model, whose gravity this page rewrites.
physics.postMessage({
  type: "init",
  runtime: new URL(`${base}body/runtime.js`, location.href).href,
  assets: `${base}body/assets`,
  timestep: 0.00025,
});
body.setView("side");

window.setInterval(() => {
  if (busy || !app.dataset.ready) return;
  const now = performance.now(),
    ms = Math.max(10, Math.min(200, Math.round(now - last)));
  last = now;
  busy = true;
  physics.postMessage({ type: "step", epoch, ms, left: drive, right: drive });
}, 40);

document.addEventListener("keydown", (event) => {
  const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;
  if (key === "G") {
    physics.postMessage({ type: "air", on: !airborne, epoch });
  } else if (key === "R") {
    epoch++;
    busy = false;
    steps = 0;
    physics.postMessage({ type: "reset", epoch });
  } else if (key === "[" || key === "]") {
    drive = Math.max(0, Math.min(1.3, drive + (key === "]" ? 0.1 : -0.1)));
    describe();
  }
});
