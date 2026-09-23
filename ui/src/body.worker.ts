import { loadScene } from "./vendor/flygym-scene";
import { Controller } from "./vendor/flygym-controller";

let physics: Awaited<ReturnType<typeof loadScene>>, controller: Controller;
let busy = false;
let native: {
  base: string;
  session: string;
  nu: number;
  timestep: number;
} | null = null;
/** Air mode is a display toy, not biomechanics: gravity is scaled down and the legs keep running their
 * measured step trajectories, so the fly tumbles and flails instead of walking. The model's own gravity is
 * -9810 mm/s^2 and the fly is about 3 mm long. It is lifted well clear of the floor because in zero gravity a
 * tumbling leg that clips the ground kicks the whole fly away for good: at 6 mm it escaped upward at 30 mm/s. */
const AIR = { gravity: 0, lift: 15, spin: [4, -3, 6], hold: 0.12 };
let airborne = false;
function setGravity(scale: number) {
  // MjOption.gravity is a read-only property but the array it returns is a live view into the model, so it
  // is written in place; assigning options.gravity throws. The opt round-trip matches the timestep path.
  const options = physics.model.opt;
  options.gravity[0] = 0;
  options.gravity[1] = 0;
  options.gravity[2] = -9810 * scale;
  physics.model.opt = options;
}
/** Break contact with the floor and set the fly tumbling. With gravity at zero and nothing touching it, its
 * linear velocity is zeroed so it hovers in frame (body.ts follows x and y but not z) while the conserved
 * spin and the still-running gait make it flail in place. */
function launch() {
  const qpos = physics.data.qpos,
    qvel = physics.data.qvel;
  qpos[2] += AIR.lift;
  // Every velocity, not just the root's: leaving the legs' stance momentum in place let internal forces
  // redistribute it and the whole fly coasted upward out of frame at ~22 mm/s.
  qvel.fill(0);
  [qvel[3], qvel[4], qvel[5]] = AIR.spin;
}
async function hash(raw: ArrayBuffer) {
  return Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", raw)),
    (v) => v.toString(16).padStart(2, "0"),
  ).join("");
}
async function nativeState(
  action: string,
  data: ArrayBuffer | undefined,
  epoch: number,
  type: string,
  start: number,
) {
  const response = await fetch(
    `${native!.base}/session/${native!.session}/${action}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: data,
    },
  );
  if (!response.ok) throw Error(`Native body failed: ${await response.text()}`);
  const values = new Float64Array(await response.arrayBuffer()),
    [time, physicsMs, nq, ngeom] = values;
  if (values.length !== 4 + nq + 12 * ngeom)
    throw Error("Native body dimensions differ.");
  const qpos = values.slice(4, 4 + nq),
    positions = values.slice(4 + nq, 4 + nq + 3 * ngeom),
    rotations = values.slice(4 + nq + 3 * ngeom);
  self.postMessage(
    {
      type,
      epoch,
      time,
      qpos,
      positions,
      rotations,
      physicsMs,
      computeMs: performance.now() - start,
    },
    { transfer: [qpos.buffer, positions.buffer, rotations.buffer] },
  );
}
self.onmessage = async (event) => {
  const m = event.data;
  try {
    if (m.type === "init") {
      if (m.native) {
        let info: any = null;
        try {
          const r = await fetch(m.native + "/info", {
            signal: AbortSignal.timeout(1500),
          });
          if (r.ok) info = await r.json();
        } catch {
          /* The static application can still run its explicit WASM backend. */
        }
        if (info) {
          const [xml, raw] = await Promise.all([
            fetch(m.assets + "/model/fly.xml").then((r) => r.arrayBuffer()),
            fetch(m.assets + "/model_meta.json").then((r) => r.arrayBuffer()),
          ]);
          if (
            info.version !== "3.9.0" ||
            info.timestep !== m.timestep ||
            (await hash(xml)) !== info.modelSha256 ||
            (await hash(raw)) !== info.metadataSha256
          )
            throw Error(
              "Native physics version, model or timestep differs from the browser model.",
            );
          const meta = JSON.parse(new TextDecoder().decode(raw));
          meta.timestep = m.timestep;
          controller = new Controller(meta);
          const r = await fetch(m.native + "/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timestep: m.timestep }),
          });
          if (!r.ok)
            throw Error("Could not create an independent native body session.");
          const session = await r.json();
          native = {
            base: m.native,
            session: session.session,
            nu: session.nu,
            timestep: m.timestep,
          };
          self.postMessage({
            type: "ready",
            backend: "Native MuJoCo 3.9.0 · 0.25 ms timestep",
          });
          return;
        }
      }
      await import(/* @vite-ignore */ m.runtime);
      physics = await loadScene({ assetsDir: m.assets });
      const options = physics.model.opt;
      options.timestep = m.timestep;
      physics.model.opt = options;
      physics.meta.timestep = m.timestep;
      controller = new Controller(physics.meta);
      self.postMessage({
        type: "ready",
        backend: `MuJoCo ${physics.mj.mj_versionString()} WASM · ${m.timestep * 1000} ms timestep`,
      });
    } else if (m.type === "reset") {
      if (native) {
        controller.reset();
        await nativeState(
          "reset",
          undefined,
          m.epoch,
          "reset",
          performance.now(),
        );
        return;
      }
      if (airborne) {
        airborne = false;
        setGravity(1);
      } // the keyframe restores the body, not the world
      physics.mj.mj_resetDataKeyframe(physics.model, physics.data, 0);
      physics.mj.mj_forward(physics.model, physics.data);
      controller.reset();
      sendState("reset", m.epoch, 0);
    } else if (m.type === "air") {
      // Display toy: gravity is scaled down and the same gait keeps running, so the fly flails in the air.
      if (native) {
        self.postMessage({
          type: "air",
          on: false,
          available: false,
          message:
            "Air mode needs the browser physics backend: the native physics session has no gravity control.",
        });
        return;
      }
      // Only the transition launches: a page that re-sends on:true would otherwise add another lift each
      // time and the fly would climb away.
      const wasAirborne = airborne;
      airborne = !!m.on;
      setGravity(airborne ? AIR.gravity : 1);
      if (airborne && !wasAirborne) launch();
      self.postMessage({
        type: "air",
        on: airborne,
        available: true,
        epoch: m.epoch,
      });
    } else if (m.type === "step") {
      if (busy) throw Error("Body work cannot overlap.");
      busy = true;
      if (native) {
        if (m.ms !== 100)
          throw Error("Native body frames must span exactly 100 ms.");
        const start = performance.now(),
          steps = Math.round(m.ms / (native.timestep * 1000)),
          controls = new Float64Array(steps * native.nu);
        for (let i = 0; i < steps; i++)
          controller.stepCPG(
            controls.subarray(i * native.nu, (i + 1) * native.nu),
            m.left,
            m.right,
          );
        await nativeState("step", controls.buffer, m.epoch, "state", start);
        busy = false;
        return;
      }
      if (airborne) {
        // The centre of mass cannot move without contact, but the root shifts as the legs flail, so the fly
        // sinks a couple of mm/s and would eventually clip the floor and be kicked away. Nudge the root back
        // toward the launch height instead of injecting velocity, which would build up energy.
        const qpos = physics.data.qpos,
          qvel = physics.data.qvel,
          t = physics.data.time;
        qpos[2] += (AIR.lift + 1.2 - qpos[2]) * AIR.hold;
        // Nudging the position alone was not enough: under play.html's stronger leg drive and longer physics
        // steps the root gained vertical speed and climbed away at ~16 mm/s regardless. Nothing is touching
        // the fly, so holding the vertical velocity at zero fights no contact force and cannot build energy.
        qvel[2] = 0;
        // A conserved spin settles into one steady axis and reads as a fly lying still on its side. Setting
        // the angular velocity to a bounded, slowly turning pattern keeps the tumble changing axis, which is
        // what makes it read as flailing. Set, never accumulated, so no energy builds up.
        qvel[3] = AIR.spin[0] * Math.sin(t * 1.7);
        qvel[4] = AIR.spin[1] * Math.cos(t * 2.3);
        qvel[5] = AIR.spin[2] * Math.sin(t * 1.1 + 1);
      }
      const start = performance.now(),
        before = physics.data.time;
      const steps = Math.round(m.ms / (physics.meta.timestep * 1000));
      for (let i = 0; i < steps; i++) {
        controller.stepCPG(physics.data.ctrl, m.left, m.right);
        physics.mj.mj_step(physics.model, physics.data);
      }
      if (Math.abs(physics.data.time - before - m.ms / 1000) > 1e-6)
        throw Error("Body timestep did not advance the requested time.");
      if (!Array.from(physics.data.qpos).every((x) => Number.isFinite(x)))
        throw Error("The body physics became unstable.");
      busy = false;
      sendState("state", m.epoch, performance.now() - start);
    }
  } catch (error) {
    busy = false;
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
function sendState(type: string, epoch: number, computeMs: number) {
  const qpos = Float64Array.from(physics.data.qpos),
    positions = Float64Array.from(physics.data.geom_xpos),
    rotations = Float64Array.from(physics.data.geom_xmat);
  self.postMessage(
    {
      type,
      epoch,
      computeMs,
      time: physics.data.time,
      qpos,
      positions,
      rotations,
    },
    { transfer: [qpos.buffer, positions.buffer, rotations.buffer] },
  );
}
