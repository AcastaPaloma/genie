// Load the official MuJoCo WASM module as a static browser module, preserving
// its import.meta.url for the neighboring .wasm binary. No Vite rewriting.
import createMujoco from './vendor/mujoco/mujoco.js';
globalThis.neuroframeMujoco = createMujoco();
