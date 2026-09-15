# Historical baseline routes

This document records the earlier projection and 40 x 30 experiments. For the current local-video route and current asset sizes, see [the main README](../README.md).

# Neuroframe

Recorded DOOM frames, drawn with real 3D FlyWire neuron branches. Drag the brain or use **Reveal the depth**: the viewing camera moves while the image projection stays fixed.

The new **Neural activity + body** page runs the full published Shiu v783 connection graph, renders simulated spike traces, and couples descending-neuron output to a physical NeuroMechFly body. The default **Synaptic image controller** now produces coarse, recognizable grayscale frames through simulated spikes in real connections. Each displayed neuron receives brightness only from its own spike history. [See source versus actual spikes](../public/data/synaptic-control-comparison.png), [recorded playback](../public/data/synaptic-playback.gif), and [methods and limitations](../RESEARCH.md). This is strong artificial control of a simplified model, not natural fly vision or a biological demonstration.

## Run

Use Node.js 22.18+ or a newer supported release, and a browser with WebGL 2, WebAssembly, web workers and hardware acceleration. Node's native TypeScript support is used by the neural tests and calibration script.

```sh
cd ui
npm ci
npm run dev
```

Open the local URL printed by Vite, normally **http://127.0.0.1:5173**. The included assets are enough to run the app; Python, SSH, a GPU training server, and FlyWire credentials are not needed at runtime.

- **Projection display:** `http://127.0.0.1:5173/`
- **Spiking brain + physical body:** `http://127.0.0.1:5173/embodied.html`

```sh
npm run build       # Type-check and build dist/
npm run preview     # Serve the production build locally
```

Deploy `dist/` to a static host. For a subdirectory deployment, build with `npm run build -- --base=/your-path/`. Use an HTTP server; opening index.html through file:// will not load the assets correctly.

## Explore

- Play, pause, step, scrub, or pick a frame in the filmstrip. Choose from four 128-frame dataset clips and 5–20 FPS playback.
- Drag to orbit, scroll to zoom, or use **Reveal the depth**, **Orbit**, and **Front**. `R` returns to the reference view.
- `Space` toggles playback; arrow keys step through frames when a form control is not focused.
- Compare with the separate source monitor. Try checkerboard and solid inputs, grayscale, brightness, point size, and footprint controls.
- **Show anatomy only** assigns a distinct false color to each neuron. **Save image** exports just the brain viewport as PNG.
- Reduced-motion preferences disable autoplay and shorten camera transitions. Mobile places the source/settings below the viewport.

## Neural activity and body controls

Play or pause the shared simulation clock. Select a DOOM clip, scrub the stimulus, and orbit either specimen by dragging or using its viewpoint menu. Neural history persists through clip/frame changes; **Reset** clears brain, body and movie together and pauses.

The default controller fits one target brightness per whole neuron, then uses feedback to stimulate **8,534 separate upstream cells** through the measured graph. None of the 1,500 displayed neurons or identified motor outputs receives direct video stimulation. The renderer uses measured spikes, never desired activities. **Image close-up** returns to the fixed 15° reference view; orbiting reveals the actual anatomy and changes the apparent image.

The image is 40 × 30 grayscale. Branch activity is summed and divided by anatomical density, with a 100 ms spike trace and a 60 Hz brightness scale. Optical input rates can reach 4,000 Hz under the reference model's zero-refractory optical protocol. These are computational control choices, not a validated procedure for a living fly. The 1,024-cell luminance encoder and earlier 37-group controller remain available for comparison; neither reproduced the requested image well.

- **Synaptic transmission:** interrupt downstream propagation. Reset with this off for a clean intervention; previous activity otherwise takes time to decay.
- **Video stimulation:** stop the external video drive while retaining the network state.
- **Pulse forward command neurons:** directly stimulate DNp09 for 150 simulated ms, independently of the video, as a positive control.

The full model contains **138,639 neurons and 15,091,983 weighted connections**. The same 1,500-neuron geometry subset is displayed, with one spike-trace brightness per neuron. The neural shader does not use the video texture to color branches.

The body is the official scan-derived NeuroMechFly browser model in MuJoCo. Selected descending-neuron rates modulate an engineered gait adapter and the upstream oscillator/recorded-step controller. Joints, contact and adhesion are physical simulation; VNC wiring, muscles, natural vision and body-to-brain feedback are not reconstructed. There is no fallback movement animation. This is a virtual fly, not a living animal.

The neural/body mode targets **0.1× real time**, with 10 dataset FPS **in simulation time** (roughly one new frame per wall-clock second). Render FPS and simulation speed are separate measured readouts. The complete packaged assets occupy approximately 92 MB; the connection data expand in worker memory. A desktop browser is recommended for the full experiment; reduced motion starts it paused.

## What is real in the projection display

The packaged anatomy contains **1,500 actual FlyWire neurons**, sampled from annotated intrinsic neurons at materialization 783. The renderer holds **768,000 points sampled along real skeleton edges**, plus up to 100 original line segments per neuron. Coordinates retain anatomical shape and relative position; preparation only centers, scales nanometers to micrometers, and flips the Y convention. This is a subset, not the entire fly brain.

The **512 DOOM frames** come from this repository's test dataset, records 0, 12, 35, and 71. Their native resolution is 80 × 60. They are resized with nearest-neighbor sampling to 120 × 90 and padded with 20 black pixels on each side in a 160 × 90 texture, preserving the 4:3 content. There is no VAE inference in this viewer. Clip titles are editorial labels, not dataset annotations.

This page is **per-point projection mapping**, not simulated neuronal activity or whole-neuron inverse rendering. One neuron's branches can receive different colors. The scene contains only point and line geometry: no flat DOOM image is composited over the brain. Synaptic connectivity is used only in the separate neural/body experiment.

In the fixed reference view, `UV = (anatomicalXY - [0, 25]) / [400, 225] + 0.5`, with coordinates in micrometers. The default central footprint has branch samples in approximately **95.3%** of its 160 × 90 cells. That is geometric support, not reconstruction accuracy. Gaps, occlusion, and coarse dataset resolution remain visible; larger footprints lose more image content. Rotation changes the view matrix, not these UVs.

## Performance and checks

The UI reports measured render FPS and actual DOOM texture updates per second separately. Playback defaults to 10 FPS; source acquisition FPS is unspecified. Playback skips ahead if rendering falls behind, and pauses work in background tabs.

On the development Mac's Apple M5 Pro, Chromium with ANGLE Metal measured approximately **60 render FPS and 10 frame updates/s** at 1440 × 1000 with all 768,000 points. This is a local observation, not a guarantee on other devices. Headless software rendering was much slower; enable browser hardware acceleration for interactive use.

```sh
npx playwright install chromium
npm test
```

Four neural tests cover excitatory delay, inhibition, seeded reset, connection integrity, image quality, and full-graph disconnection/shuffling interventions. Five browser tests cover both pages, physical body integration, controls, mobile layout and reduced motion. A paused-source scrub also verifies that changing the video alone cannot change the neural image. The macOS browser configuration uses ANGLE Metal. See [VALIDATION.md](../VALIDATION.md) for completed checks and the finish verdict.

## Rebuild the data assets

The app includes approximately 92 MB of assets for both modes. Rebuilding is optional. Asset preparation uses NumPy, Pillow, certifi and array-record; connectome packaging additionally uses pyarrow, and the image-basis experiment uses scipy:

```sh
# From the repository root:
.venv/bin/python ui/scripts/prepare_assets.py
.venv/bin/python ui/scripts/prepare_assets.py --only clips
.venv/bin/python ui/scripts/prepare_assets.py --only brain --neurons 1500
.venv/bin/python ui/scripts/prepare_connectome.py
.venv/bin/python ui/scripts/prepare_body.py
.venv/bin/python ui/scripts/test_neuron_image_basis.py
node ui/scripts/calibrate_controller.ts
.venv/bin/python ui/scripts/render_controller_results.py

# Current synaptic image controller and repeatable 64-frame validation:
CONTROL_ONLY_INPUTS=1 .venv/bin/python ui/scripts/prepare_synaptic_control.py
.venv/bin/python ui/scripts/prepare_synaptic_display.py
node ui/scripts/validate_synaptic_sequence.ts
.venv/bin/python ui/scripts/render_synaptic_results.py
```

The clip step expects `data/test/data_0000.array_record`. The anatomy step downloads the public annotation table and skeletons over HTTPS, caches originals under `ui/.cache/`, and samples deterministically with seed 783. Only load trusted ArrayRecord data: its records use Python pickle serialization.

`public/data/brain.json` records string root IDs, original skeleton hashes, annotation hash, coordinate transform, sampling method, binary hashes, and source links. `public/data/clips.json` records dataset revision, record indices, raw-video hashes, dimensions, and frame preprocessing. The two binary geometry files contain little-endian float32 `[x, y, z, neuronIndex]` entries; consecutive line vertices form segments.

Data sources, attribution, licensing, and modifications are in [DATA_SOURCES.md](../DATA_SOURCES.md). The visual system is recorded in [DESIGN.md](../DESIGN.md).

The brain model source is pinned to `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`. The body manifest pins the official generated assets and their individual hashes; `prepare_body.py` verifies or restores them. The published connection table is converted to compressed outgoing adjacency lists, preserving signed contact counts and neuron index identities. The gzip loader supports servers that send either compressed bytes or transparently decoded content.

The current controller assets retain a connectome hash, operator hash, anatomical hash, camera-selection protocol and control-cell identities. The published sequence report scores all 1,200 pixels after 300 ms warmup per clip, preserves neural state across each 16-frame sequence, and includes the disconnection result. Its GIF replays recorded simulation at 10 FPS in simulation time; it is not a claim of 10 FPS live computation.
