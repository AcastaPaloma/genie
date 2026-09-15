# Neuroframe

Play a local DOOM clip through a simulated fly connectome and watch its spike
activity form the image on real 3D neuronal anatomy. The same simulation drives
an articulated NeuroMechFly body in MuJoCo.

**Open [Live synaptic cinema](http://127.0.0.1:4173/sites.html).** The earlier
experiment uses a **160 × 120 grayscale target and 18,901 real display neurons**,
with all **138,639 model neurons and 15,091,983 measured weighted connections**
simulated. It is a grainy reconstruction, not an HD video screen.

```sh
cd ui
npm ci
npm run build
npm run preview
```

Use Node 22.18+ and a desktop browser with working WebGPU, WebGL 2, WebAssembly,
and hardware acceleration. The measured backend is Chromium / ANGLE Metal on
an Apple M5 Pro, 16 GPU cores, 24 GB memory. No SSH connection or training server
is needed. Uploaded videos stay on this computer; decoding is local.

## Use it

Choose an example or a local video, then start the experiment. MP4/H.264 and
WebM support depends on the browser. The source is letterboxed to preserve its
aspect ratio, converted to grayscale, and supplied to an online controller.
Neural history persists across frames, seeks, and clip changes. Reset clears
brain and body and pauses playback.

- **Image close-up** shows the fixed reference view. Orbit or choose another
  viewpoint to reveal the unchanged 3D coordinates.
- **Fixed observation sites** shows one actual branch location per displayed
  neuron. These are not somas. They were selected using anatomy and connectivity
  only, independently of every input clip.
- **All sampled branches** illuminates that neuron's entire sampled arbor with
  the same activity. Image detail spreads out along those branches.
- **Soften point speckle** applies a fixed .6-cell blur to emitted light. Disable
  it to inspect raw sites. It does not use the source image.
- **Synaptic transmission off + Reset** is a causal test: the display stays dark
  while externally driven upstream neurons continue firing.
- **Video stimulation off** stops external drive without resetting neural state.
- **Save brain view** exports the current anatomical viewport.

The smaller source preview is paired with the completed neural output. Scrubbing
while paused updates the preview but leaves the brain unchanged. Background tabs
pause. Reduced-motion preferences, or `live.html?paused`, start paused.

## What actually produces the light

The controller commands pulse trains on **106,621 separate upstream cells**.
The 18,901 displayed cells and identified descending motor outputs cannot be
actuators. Real measured edges transmit spikes to them; each displayed cell's
brightness comes only from its own simulated spike history. There is no
source-texture overlay, per-point target coloring, prerecorded neural-output
playback, or target-to-renderer shortcut in this route.

This uses **ideal external voltage clamps**, including cancellation of incoming
currents on actuator cells, with rates up to 4,000 Hz. It controls a large fraction
of the model, not only sensory neurons. That is a strong engineering assumption,
not natural fly vision, an unmodified biological stimulation protocol, or a
procedure demonstrated on a living fly. Selecting display neurons with negative
outgoing weights reduces uncontrolled positive recurrence; the published
neurotransmitter sign assumptions are approximate.

The LIF model retains .2 ms integration, 1.8 ms synaptic delay, and ordinary-cell
refractoriness. Display light uses a 40 ms exponential spike trace, normalized at
300 Hz. Coverage is **98.4%** of the reference raster. All geometry stays in its
original relative anatomical position; only centering, unit conversion and a
coordinate reflection are applied. Full derivation and limitations are in
[LIVE_METHOD.md](LIVE_METHOD.md).

The body uses an explicitly engineered descending-neuron → gait-controller
adapter. MuJoCo computes joints and contacts at .25 ms steps. It follows the same
neural output sequence one processing stage behind. The VNC, natural muscles,
natural vision and body-to-brain feedback are not reconstructed. The body and
brain specimens are different flies. No fallback movement animation is used.

## Measured performance and evidence

The production build's complete local-file path processed **10.84 new neural
frames/s** during a 20-second run of a native 35-FPS Freedoom input, with body
physics and rendering active. Mean capture-to-neural-display submission was
**54 ms**; mean delay from the decoded frame becoming available was **75 ms**
(95th percentile **101 ms**).
These measurements exclude physical monitor scanout and file-decoder startup;
the neural observation trace adds temporal smoothing. They are local engineering
measurements, not a guarantee for another GPU, codec, or clip.

The app displays new neural outputs/s separately from model time. Source playback
follows the wall clock; the controller takes the newest available frame without
building a backlog. A 10-FPS source limits distinct updates to about 10/s. Screen
refreshes are not counted as new neural outputs.

[Source versus simulated light](public/data/live-validation/comparison.png),
[sequence metrics](public/data/live-validation/sequence-summary.json), and
[VALIDATION.md](VALIDATION.md) record the tests and their limits. More target
pixels do not guarantee faithful fine detail; contrast, grain, missing sites and
spike-trace persistence remain visible.

## Build and validate

```sh
npm run build
npm run preview
npm test
```

Serve through HTTP, not `file://`. `dist/` is a static build. For a subdirectory,
pass Vite's `--base` option directly: `npx vite build --base=/your-path/` after
`npx tsc --noEmit`. Dev-only GPU/sequence harnesses are excluded from the
production entrypoints. See [VALIDATION.md](VALIDATION.md) for the additional
hardware-dependent validation commands.

For development, `npm run dev` serves the same entrypoint at
`http://127.0.0.1:5173/live.html`. Keep other experiment tabs paused when measuring
performance; they share the same GPU.

## Earlier experiments

- `/embodied.html`: the earlier 1,500-neuron, 40 × 30 whole-arbor controller;
  slower than wall-clock real time.
- `/`: per-point anatomical projection mapping. It is explicitly labeled as
  projection, and does not simulate synaptic image formation.

The earlier setup, tests and data preparation instructions are preserved in
[docs/BASELINE.md](docs/BASELINE.md). Their numerical results apply to those routes,
not the new live display. Data licensing and provenance are in
[DATA_SOURCES.md](DATA_SOURCES.md); the visual system is in [DESIGN.md](DESIGN.md).

## Rebuild the live anatomy exactly

The final directory contains `parent-anatomy.json`, which pins the complete
ordered 29,998-neuron input pool, source skeleton hashes and coordinate transform.
With the original annotation and model-mapping caches prepared by the baseline
asset scripts, run from `ui/`:

```sh
../.venv/bin/python scripts/prepare_large_anatomy.py --manifest public/data/selected-inhibitory-selective-160/parent-anatomy.json --output inhibitory --samples 256
../.venv/bin/python scripts/prepare_observation_sites.py --directory inhibitory --width 160 --angle -45
../.venv/bin/python scripts/prepare_selected_observers.py --directory inhibitory --width 160 --score external-selectivity
```

The manifest rebuild fails if a source hash or packaged geometry hash changes.
Generated parent and unsuccessful experiment assets have been moved to
`.cache/archived-public-data-20260915/`; the shipped public assets total about
249 MiB including the body. Only final runtime assets and evidence are deployed.
Native game inputs can be regenerated with `prepare_live_clips.py` and
`prepare_live_stress_clip.py`; the latter is a validation input, not neural output.
