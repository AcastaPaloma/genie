# Neuroframe

A local experiment using the **complete public FlyWire v783 brain**: all 139,255
official neurons, every available skeleton branch, and all 15,091,983 measured
neuron-to-neuron connections (54,492,922 synaptic contacts).

The home page and `live.html` both open [the full-brain experiment](http://127.0.0.1:4173/). The older projection-mapping prototype is archived at `projection.html`.
Its online controller accepts browser-decodable video and changes external
conductances and currents. The simulator produces spikes, and each neuron's spike history lights
its entire released skeleton. **The complete front view and connected native body
have sustained about 10 new neural displays/s on the measured M5 Pro machine;
recognizable high-detail video still fails visual review.** The
older selected-site measurements do not apply to the new renderer.

```sh
cd ui
npm ci
../.venv/bin/python -m pip install -r requirements-body.txt
npm run dev:full
```

Use a desktop browser with WebGPU, WebGL 2 and hardware acceleration. Full
geometry is several gigabytes; there is no automatic fallback to a neuron subset.
The native MuJoCo service binds only to `127.0.0.1:8769`, retains the same 3.9.0 engine, model and 0.25 ms timestep, and receives only gait controls. Its backend is identified beside the body. Without it, the static application uses the browser WASM body, which was slower in the measured run.

The interface fails explicitly if the complete data or GPU resources are missing.
Local files are decoded on the computer and are not uploaded.

## Regions

The main view draws the **central brain**: every released neuron except the optic
lobes (super_class `optic`) and the photoreceptor afferents (R1-6, R7, R8, ocellar
retinula cells). That is 50,328 of the 139,255 neurons and 169,589,408 of the
268,278,432 vertices, selected by published annotation, never by a coordinate cut.
The two optic lobes are dense enough to fill the frame and hide the central
neuropil, which is why they are withheld by default.

Withholding is a **display** choice. The complete archive is still downloaded,
checksummed and uploaded, every neuron is still simulated with all its measured
connections, and withheld neurons keep spiking; they are simply not drawn and
contribute no light. `?region=all` draws the complete brain and `?region=optic`
draws only the lobes. Membership is measured by
`scripts/prepare_brain_regions.py` into `regions.json`.

## Use the full brain

The initial view uses **labelled static anatomy inspection** so inactive branches
are visible. Start the experiment to switch to simulated spike activity. Choose
an example or any local video the browser can decode. Video is letterboxed and
converted to a 640 × 480 grayscale target in the current default trial. That is
an input size, not a claim of effective image detail. The previous 320 × 240
setting remains at `?controller=visual-inputs&controlWidth=320`.

Use **Enlarge brain** for a full-window view with playback controls; Escape or
**Return to body** restores the paired view. The same anatomical canvas and activity
remain active, and the body stays connected. Orbit, zoom, or choose a viewpoint
to inspect the unchanged 3D positions. Every
parent link belonging to an official neuron remains present; there are no selected display
cells, one-point observation sites, or newly sampled branches in this route.

Playback preserves neural history across frames, seeks and clip changes. Reset
clears both simulations. Background tabs pause. The first view starts paused;
`?autoplay` starts neural playback after loading unless reduced motion is set.

Disable synaptic transmission, reset, then play to test causality. Directly
stimulated cells may continue firing and remain visible. Cells receiving only
synaptic input must remain silent. Their spike counts are reported separately.
Video stimulation off sets the external currents to zero without clearing state.

## Play the DOOM world model on the brain

The live page accepts a third stimulus: frames generated on the fly by the trained Genie world model
(`experiments/play_server.py`: the frozen tokenizer, latent action model and dynamics model, run through the
cached fast paths in `experiments/fast_infer.py`). Start the model server from the repository root, then
the UI, which proxies `/world` to it:

```sh
python3 experiments/play_server.py --port 8008 --steps 3 --dtype float16 \
  --checkpoint data/checkpoints/dynamics_newvae_weights.pt \
  --tokenizer data/checkpoints/vae_new_43db_weights.pt \
  --lam data/checkpoints/lam_spatial_a100_pan.pt
cd ui && npm run dev:full
```

Those three paths are the current pair and are not the script's defaults, which still name the
earlier 37 dB tokenizer. On 24 validation clips the current pair reaches 25.55 dB on the next frame
against a 25.47 dB copy-the-last-frame baseline (the earlier pair: 24.96 dB, below that baseline) and
holds 24.1 dB six frames into a rollout (earlier: 23.3 dB). It also never collapsed at 2, 3, 4, 6, 8 or 25
MaskGIT passes, where the earlier pair collapsed on 2 of 24 clips at 4 passes. Its exact token-ID accuracy
is lower, 18% against 23%, because token accuracy is measured in each tokenizer's own codes and is not
comparable between them; judge a pair on picture quality against the copy-last baseline instead.

Press **Play world model** under *Your stimulus*, then **Start experiment**.

**Remote generation and pipelining.** The world model can run on a GPU box reached through an ssh tunnel
(`ssh -f -N -L 8008:localhost:8021 piano`, then `WORLD_MODEL_URL=http://127.0.0.1:8008 npm run dev`). On an
A100 a frame takes 39-47 ms against 220 ms on the Mac, but the tunnel costs about 172 ms per round trip, so
one request at a time gives no gain. The client therefore keeps several step requests in flight and presents
the frames strictly in request order. The depth is measured, not assumed, since the page reaches the server
through a proxy and cannot tell local from remote: it targets round trip / server time, capped at 4, which
lands on 1 for a local server and 4 through a tunnel. `?inflight=` fixes it manually.

Measured end to end with nothing else driving the server: 3.4 generated frames/s before, 20.7-23.3 after,
with the brain rising from 4 to 9-10 displays/s because the world model no longer competes for the local
GPU. Round trip 159-171 ms, server time 35-36 ms, adaptive depth settling at 4. The cost is stated in the status
line: at depth 4 about 170-200 ms of play is already committed when you change key. Deeper than 4 adds lag
without frames, because the server serializes behind one lock and saturates near 23 frames/s. Transient
connection drops are retried three times before the game stops, since tunnels do drop.

**Air mode.** `G` lifts the fly off the floor: the body worker sets gravity to zero, kicks it upward and
spins it, while the gait controller keeps running its measured step trajectories, so the legs flail with
nothing underneath them. The fly label says so while it is on, `R` and toggling `G` again land it, and the
page stops auto-righting the fly while it is airborne, since a low-gravity tumble looks exactly like having
fallen over. It needs the browser physics backend; the native MuJoCo service has no gravity control and the
label says so.

**Frame-rate cap.** The A100 can generate about 23 frames/s, which turned out to be faster than the picture
reads: the brain's light is a 40 ms trace and the scene changes faster than the eye follows it. The play page
therefore aims for 5 generated frames/s by default (`?fps=`, `?fps=max` for as fast
as the server allows), because the A100 can produce about 23/s and that is faster than the picture reads.
**K** and **L** change it live. Capping also shortens the control lag: the client then queues only deep
enough to cover the wire within one frame interval, so at 5 frames/s it runs one request at a time and
commits about 170 ms rather than 4 frames.

**One driver at a time.** `play_server.py` serializes every client behind a single lock and keeps one
generation state: one context cache, one prompt clip. A second client does not merely halve the frame rate,
it interleaves its own actions into the same world, and a benchmark running alongside a player will make
both look slow and behave strangely. Multi-client play would need per-connection cache state and the GPU
memory for it.

**Screens.** `T` cycles the play page between all three panels, the brain alone, the generated game alone,
and the fly alone; `?screen=` picks one at load. Everything keeps running while hidden: the world model
still generates, the simulation still integrates every frame, and the body still steps.

**What the eight latent codes do**, measured on the deployed pair by holding each code for five steps from
four prompts, as mean camera turn per step (positive = the camera turns right) with its spread across scenes:
code 0 +11.3 (sd 1.3), 6 +10.6 (2.6), 7 +3.7 (9.9), 4 +1.2 (8.4), 1 -1.3 (8.9), 2 -4.7 (8.2), 3 -4.9 (9.0),
5 -7.5 (6.2). The spread above is step to step within a fixed prompt set; the per-code mean itself moves with the scene,
so code 0 measured +11.3 on one prompt set and +2.0 on another. **Read the table as an ordering of the
codes, not as a turn rate**: 0 and 6 are the rightmost, 5 the leftmost, 1 and 4 the middle, but the
magnitude depends on the room. The default key map puts the two steadiest codes on the plain keys
(`D` = 0, `A` = 5). The turn figure in the heads-up display is the last step only, so it can read the
opposite sign for one frame when the context cache restarts; judge a key by several held steps.
No code fires the weapon: the weapon band moves by at most 0.02 for any of them, because ATTACK is a single
rare action id that an 8-code alphabet never spent a slot on. Every code also walks the player forward, so
releasing all keys is the only way to stand still. Generated frames leave the
client through a canvas `MediaStream`, so the same `LiveSource` capture and the same controller receive
them as any decoded video: the brain forms each frame the model produces. Hold **A** / **D** to turn (this
LAM's latent actions are a camera-pan quantizer; **W** / **S** are its two no-turn codes; **0**–**7** send
raw codes). **R** takes a new prompt clip, **Q** toggles 3 / 25 MaskGIT passes, **[** **]** change passes.
With nothing held the world holds its frame (every latent code walks forward, so this is the only way
to stand still); `?idle=W` keeps it walking. Reset also resets the world to a fresh prompt. A is code 3
and D is code 6, the opposite of `play_server.py`'s default map, because A turned the player right in play
testing; `?keys=` overrides.

**`/play.html` is the arcade version**: the generated frame and the brain side by side, the physical
fly across the bottom, no clips, file input or inspector. It connects to the world model by itself and
starts as soon as the anatomy is uploaded. Keys: A / D turn, R new prompt (also resets brain and body),
Q and [ ] passes, Space pause, P cycle colours, V brain viewpoint, B fly viewpoint; `?palette=`, `?steps=`,
`?world=` work as on live.html. Its fly uses an **arcade motor adapter**, labelled on the page: the descending-neuron readout amplified by
`?bodyGain=` (default 9), a twitch scaled by the brain's mean spike rate (`?twitch=`, default 2, 0 disables)
that beats two frequencies against each other and quickens with the firing rate, physics paced by the wall
clock, and a `?bodySpeed=` multiplier (default 3) that runs the body clock faster than real time. **,** and
**.** change both gains live. The page also asks the controller for a much higher firing rate than
live.html, `?rate=` (default 840 against live.html's 180). Measured over 12-second runs: mean rate 19 Hz ->
48 Hz, spikes 1.05M/s -> 2.6M/s, distance travelled 11.5 -> about 40 units.

Two limits found while tuning. The mean firing rate saturates near 50 Hz: asking for 1400 or 2800 gives no
more than 840 does, because thresholds and refractory periods bound it. And driving the gait adapter past
about 1.3 stops it walking, so the fly shuffles on the spot; extra speed has to come from `bodySpeed`, not
from more drive. `?rate=180&bodyGain=4&twitch=1&bodySpeed=1&drive=1.2` restores the earlier behaviour.

On `play.html` the frame is **placed where the anatomy can show it** rather than letterboxed across the
whole raster, and the camera frames exactly that area so the entire picture is on screen. Share of each
band's pixels that have any branch behind them, measured on `whole-arbor-640` (content rows 0-59 scene,
60-77 weapon, 78-89 status bar), with the share of the brain's cable inside the rectangle:

| `?frame=` | placement | scene | weapon | status bar | worst row | cable |
|---|---|---|---|---|---|---|
| `full` | 640 x 360 at (0, 60), letterboxed | 59.7% | 58.0% | 5.1% | 0.0% | 100% |
| `band` | 480 x 270 at (78, 60) | 58.4% | 98.8% | 85.2% | 0.0% | 75.6% |
| `tight` (default) | 320 x 180 at (172, 132) | 97.8% | 98.6% | 92.7% | 79.2% | 46.7% |
| `strict` | 224 x 126 at (351, 181) | 100% | 100% | 100% | 100% | 23.5% |

**Worst row is the number to watch.** `band` and `full` each contain a row of the picture with no cable
behind it at all, which no average reveals and which is what a viewer reads as the top being cut off.

`tight` is not a quality compromise: it beats `band` by 39 points on the moving scene and 7 on the status
bar. What `band` buys is a claim about the brain, not the picture, since `tight` leaves 53% of the cable
outside the picture area, so more than half the anatomy no longer carries image content.

The letterboxed mapping puts the lower rows of the frame below the brain's ventral edge, where there is no
cable to light, which is why its status bar cannot appear. `tight` puts the whole picture on supported
anatomy at the cost of using fewer of the brain's branches; `band` keeps more of the brain at the cost of
half the scene. The picture's edges fade out over 8 raster pixels instead of ending hard. `?focus=bottom`
or `whole` restore the earlier region framings.

The brain image can also be snapped to the game's own pixel grid, cycled with **Z** and selected with
`?pixels=`: `game pixels` (default) uses one cell per pixel of the generated frame, since the controller
letterboxes the 160 x 90 frame into its 640 x 480 raster, so one game pixel is exactly 4 x 4 control cells;
`half pixels` uses a 2x finer grid; `full detail` keeps the raw 1280 x 960 cache. Each cell averages the
supported light texels inside it, and the grid in use is stated in the brain label.

Compare contrast between grid modes at the cell scale, not per device pixel: inside a cell neighbouring
pixels are identical by construction (96.4% of them in game-pixel mode), so a naive per-pixel gradient
reports 0.0050 against 0.0264 for full detail. Measured at a common 19-pixel cell the ordering reverses,
0.0321 raw full detail, 0.0443 refined full detail, 0.0491 half pixels, 0.0509 game pixels: averaging
concentrates structure at the cell scale rather than removing it.

On `play.html` the brain view also runs **display processing of the simulated light**, stated in the
brain label and cycled with **X**: `refined` (default), `punchy`, `raw`. Overall brightness is separate, on **-** and **=** (or `?brightness=`,
default 0.65), because the picture area concentrates the light and the right level depends on the room and
the screen. It stretches each frame to the
light range of the pixels actually on screen (2nd to 99.5th percentile, smoothed across frames), applies
gamma, an unsharp local-contrast term that skips unsupported texels, and a saturation factor. It is a
monotone reweighting of light the simulation produced: no video content and no per-branch colour enter it,
and `?look=raw` or the X key turns it off. Measured on the default bottom framing it raises median
brightness from 0.22 to 0.28 of white, the 99th percentile from 0.62 to 0.99, local contrast from 0.014 to
0.025 and colour saturation from 0.05 to 0.13, clipping 1.3% of lit pixels.

The brain header has a **colour** menu: the spike light is shown through a false-colour palette
(heat by default; viridis, plasma, magma, or the original grayscale, also `?palette=grayscale`), or in
**source colours**: brightness is still the density-normalized spike light, and only the hue is taken from
the frame the controller was given, at the same raster position (`play.html` defaults to this). The
palettes use no video at all. Saved brain images use the current mode.

Options: `?keys=W=1,S=4,…` mirrors a server started with `--keys`; `?steps=N` sets the initial passes;
`?world=URL` targets a server directly (it must send CORS headers; the proxy path needs none);
`WORLD_MODEL_URL=http://127.0.0.1:8018 npm run dev` retargets the proxy, and `ssh -L 8008:localhost:8008
piano` lets the GPU box generate. `/world-test.html` exercises the source path without the brain, and
`WORLD_MODEL_URL=… npx playwright test tests/world-model.spec.ts` runs it headless (it skips with no server).

Measured on an M3 Max (MPS): the cached world model generates about 4 frames/s in float32 and about
5 frames/s with `--dtype float16` (around 200 ms per frame, three MaskGIT passes). The brain side processes
about 10 frames/s on the measured M5 Pro, so generation is the slower stage; a key press reaches the neural
display after roughly one generated frame plus one neural frame. The generated pictures share the dynamics
model's current limits (see `experiments/README.md`): the world drifts, and only turning is controllable.

## What is real and what is modelled

The geometry and contact counts are measured FlyWire data. The public skeletons
were generated from LOD-1 meshes and downsampled by their publishers. We retain
all vertices and parent links in that release without additional downsampling.
This is complete released centerline anatomy, not the original electron-microscopy
volume, membrane meshes, or a complete biological simulation.

The Shiu model contains 138,639 of the official cells. Reconciliation against the
complete public connectivity table finds **616 additional official neurons with
no listed incoming or outgoing connections**. They are included as isolated
model cells; no missing synapses are invented. The connection tables otherwise
match exactly. All original presynaptic sign assumptions are preserved.

The default experimental feedback controller drives annotated visual afferents
and adjusts conductances on the remaining non-motor cells. It has no constant
seed currents. The input population is fixed independently of the video. Identified descending motor outputs
receive no direct stimulation.
All stimulated cells remain visible. Ordinary thresholding, refractoriness and
incoming synaptic currents remain active on every cell. The controller never
writes voltages, spikes, traces, or rendered brightness directly.

The front view uses a 1280 × 960 cache integrating all 268,139,177 released branches. This is a complete, transparent cable-length projection evaluated from current spike activity; it is not a source-video raster. Orbiting uses the original complete 3D indexed geometry, with a different line-rasterization approximation and a higher draw cost.

The entire arbor shares its neuron's 40 ms exponential spike trace. The renderer
has no source video argument or texture. Whole-neuron activation couples distant
image regions, so a complete brain can produce a less recognizable image than the
older selected-point display. The diagnostic independent-brightness fit is clearly
separate from neural playback and is never a runtime fallback.

The physical NeuroMechFly body uses MuJoCo and an engineered descending-neuron
→ gait adapter. Natural vision, the ventral nerve cord, muscles and body-to-brain
feedback are not reconstructed. The brain and body come from different specimens.
No living fly is connected.

## Reproduce the complete import

With the existing baseline caches prepared, run from the repository root:

```sh
.venv/bin/python ui/scripts/download_full_brain.py
.venv/bin/python ui/scripts/prepare_full_brain.py
.venv/bin/python ui/scripts/prepare_full_connectome.py
.venv/bin/python ui/scripts/prepare_full_basis.py --width 320
.venv/bin/python ui/scripts/prepare_full_basis.py --width 640   # the live and play pages' default control width
.venv/bin/python ui/scripts/prepare_full_basis.py --width 1280
.venv/bin/python ui/scripts/prepare_render_cache.py
.venv/bin/python ui/scripts/evaluate_full_anatomy.py
```

The baseline cache prerequisites are the pinned Shiu `Completeness_783.csv` and
`Connectivity_783.parquet` from `prepare_connectome.py`, plus the shared coordinate
origin in `public/data/brain.json`. Public acquisition verifies the publishers'
file sizes and MD5s, then records SHA-256 hashes. The complete runtime manifest
lists every official root ID, model index, neuron/chunk counts and chunk hashes.
Missing neurons, broken parent links or changed source hashes fail the import.
The source archive's extra roots outside the official list are recorded explicitly.

Data and audits live in `ui/.cache/full-brain/`; complete runtime files are in
`ui/public/data/full-brain-783/`. See [FULL_BRAIN_PROGRESS.md](FULL_BRAIN_PROGRESS.md)
for measured progress and remaining limitations.

## Validation and earlier experiments

`npm run build` checks TypeScript and creates a static build. `npm run test:neural`
checks delayed propagation, inhibition, disconnection and the earlier controller.
The development-only `gpu-test.html` additionally checks CPU/GPU agreement,
normal tonic-current refractoriness, downstream silence after cutting transmission,
controller buffer execution and full-size indirect event dispatch.

New neural frames/s counts distinct decoded frames through GPU draw completion,
with body physics active. Capture latency includes that completion, but physical
monitor scan-out is outside the measurement. Screen refreshes are not counted.
Keep other experiment tabs paused during benchmarks.

- `/sites.html`: the earlier 18,901-neuron selected-site display; historical
  measurements and assumptions are in [docs/SELECTED_SITES.md](docs/SELECTED_SITES.md).
- `/embodied.html`: the earlier 1,500-neuron, 40 × 30 whole-arbor experiment.
- `/projection.html`: explicitly labelled per-point image projection, without synaptic control.

Primary sources: [official neurons and connectivity](https://zenodo.org/records/10676866),
[whole-brain skeleton archive](https://zenodo.org/records/10877326),
[publication annotations](https://github.com/flyconnectome/flywire_annotations/tree/v2.1.0),
[reference neural model](https://github.com/philshiu/Drosophila_brain_model).
Licences and earlier preparation steps are in [DATA_SOURCES.md](DATA_SOURCES.md)
and [docs/BASELINE.md](docs/BASELINE.md).

See [QUALITY_BAR.md](QUALITY_BAR.md) for the user's visual and temporal completion criteria. A passing causal intervention is not a passing reconstruction.

## Event-gated full-brain trial

The earlier event controller remains at `?controller=events`, and
`?controller=tonic` preserves the earlier baseline. It simulates every
official neuron and renders every released branch. A fixed 700-cell excitatory
population receives tonic current; 138,543 other cells receive strong external
conductance control, and the 12 identified motor outputs remain unactuated.
Conductance prepares cells below threshold and is released when a requested spike
and a positive measured synaptic current coincide. Brightness remains the owning
neuron’s 40 ms spike trace, with a fixed 1.5× display exposure.

**The image information comes from extensive external control.** The measured
connectome supplies and constrains the emitted spikes. This is not a biological
stimulation protocol or evidence of natural visual processing. The conductance is
very strong (16,384 times the baseline leak in the software model). Cutting
transmission after reset silences all non-seed cells in the tested sequences.

The trial improves spatial correspondence in the tested clips, but high-detail,
recognizable arbitrary-video playback is still unproven. Three-tick integration
retains the original 0.2 ms step and 1.8 ms synaptic delay. GPU validation checks
one-, three-, and nine-tick batches against the same CPU reference.

`scripts/audit_spike_reachability.py` records a necessary connectivity bound;
`full-bench.html` accepts `hold=1.9` for stationary-target diagnosis and `audit`
to export requested and measured rates for every neuron. `fit=0.95` tests a
geometry-derived supported target window. That smaller framing is diagnostic
only and is not used by the full UI.

The event controller corrects its requests from the emitted spike image and
supports underactive cells through the signed measured graph. Upstream support
requires current image demand, so stale requests cannot sustain background
firing after a dark cut. On 60 tuning frames, supported MSE falls from 0.005545
to 0.005244; the first two matched corridor frames preserve stronger edges.
The measured full application sustains 10.56 new displays/s with 66.2 ms mean
processing latency and 76.3 ms p95 on this Mac, including complete anatomical
draws and the connected native body. Decoder-to-draw latency averages 81.1 ms
(121.9 ms p95), excluding display scan-out. That speed does not establish visual
quality. `event-support-live-performance.json` records the measured window.

`Latest evaluation · 640 × 480 / 35 FPS` contains maps 04, 08 and 12 generated
after the new controller was frozen. The earlier evaluation is also retained.
The trial links a **recorded diagnostic** showing the new source and actual
simulated spikes through two scene cuts. It is never used as runtime output.
Temporal correspondence passes this check; scene edges remain weak and the
recognition criterion still fails. See `scripts/evaluate_frozen_sequence.py`,
`scripts/audit_anatomical_limit.py`, and [FULL_BRAIN_PROGRESS.md](FULL_BRAIN_PROGRESS.md)
for the measurements and their scope.

The numerical harness also accepts `controlWidth=640`. Its exact operator includes
every branch. It costs more compute and does not improve matched edge detail at
either common 320-pixel or 1280-pixel observation resolution, so the main controller
stays at 320 × 240 while the complete anatomical renderer remains 1280 × 960.
See `scripts/compare_control_resolution.py` and `scripts/compare_event_support.py`.


## Anatomical visual-input trial

The earlier visual-input controller remains at `?controller=visual-seeded`. It retains the
complete anatomy and measured graph while supplying bounded external currents to
all **11,391 annotated visual afferents**, alongside 700 constant seed cells.
Another 127,152 cells receive the existing conductance control, and the 12 motor
outputs remain unactuated. This does not reconstruct a biological retina: the
external controller still imposes the image information.

The trial uses a denser requested rate scale of 180 Hz. No membrane, spike or
brightness is overwritten. Direct-input and synapse-dependent spikes are counted
separately in the UI. The displayed optical fractions integrate complete arbors
in the 320 × 240 control projection; they describe emitted light, not information
origin or a guarantee about another camera angle.

On the 60 tuning frames, supported MSE improves from 0.005244 to 0.003382, but one
of the three matched final corridor frames has worse edges. This mixed result
kept that version separate from the then-default event controller. `visual-input-comparison.png`
shows actual spike light for both controllers at the same exposure.

A frozen, new 640 × 480 / 35 FPS input uses maps 13, 18 and 24. All 120 input and
output frames are distinct, with persistent history across both cuts. The best
of the tested temporal lags is zero in each scene. Mean image correlations are
0.760 / 0.747 / 0.654 and edge cosines 0.402 / 0.424 / 0.313. Fine recognition
still fails; these are different maps from previous holdouts, not a before/after
comparison. The UI links `recorded-visual-comparison.mp4`, an actual-spike diagnostic
that is never read by the runtime. Archived controller sources and input hashes
are checked by `scripts/evaluate_frozen_sequence.py`.

In that holdout, 82.9% of projected light comes from cells requiring incoming
events, 16.1% from directly stimulated visual inputs and 1.0% from constant seeds.
After transmission is cut and state reset, the visual inputs and seeds still
fire; every other cell has zero spikes and there are zero synaptic deliveries.
A separate three-cell GPU test checks connected propagation, disconnected direct
input activity, and complete silence with stimulation also disabled.

`scripts/audit_input_capacity.py` supplies a conditional anatomical bound that
fixes observed seed brightness and constrains unreachable cells to zero. It does
not simulate attainable rates. On the stationary map02 target, adding visual
afferents reduces the optimistic bound from approximately 0.002596 to 0.001946.
Inhibition, current limits, timing and spike noise can only make the actual
controller harder to solve; the bound is never used as rendered neural output.


The production visual-input UI measures 10.73 new neural displays/s in both the
paired and enlarged views on the M5 Pro. Mean processing latency is 74.7 / 74.2 ms
(p95 83.7 / 82.6 ms), with the full anatomy and native body active. Each measured
window contains 120 distinct source times. Decoder-to-draw means are 91.4 / 90.5 ms;
physical display scan-out is excluded. See `event-visual-live-performance.json`.
These timing checks do not establish recognizable fine detail.


## Visual inputs without constant seeds (320-control stage)

The home page, `live.html`, and `?controller=visual-inputs` now open the improved
visual-input trial. It keeps every neuron and branch, directly stimulates the
11,391 annotated visual afferents, and applies external conductance control to
127,852 other cells. The 12 identified motor outputs remain unactuated. There
are **no constant seed currents**. All non-visual cells need incoming events to
fire in the tested model, while the external controls still impose the image.

The external preparation is now 0.01 mV below the unchanged firing threshold
(`gateBias=6.99`, versus the original 6.8 mV bias above rest). The extremely strong
conductance is 327,680 times the baseline leak, increased to retain suppression
when a cell should stay dark. This is engineered control, not a physiological
stimulation protocol. It leaves the measured contacts, membrane equation,
threshold, refractory period and own-spike brightness readout intact.

On the three matched tuning frames, edge similarity improves from 0.313 / 0.364 /
0.350 to 0.362 / 0.411 / 0.374. Mean supported error over all 60 tuning frames
falls from 0.003382 to 0.003205. A dark cut leaves mean light around 1.8e-12 after
one simulated second, with zero new gated spikes in the last 100 ms. The new
`weak-input-comparison.png` shows measured output at equal exposure. These are
partial improvements; fine-detail recognition still fails.

After freezing this controller, new maps 06, 20 and 29 were generated. The
120-frame neural evaluation decodes the source through a **browser File object
and object URL**, with all frames distinct and history preserved through both
cuts. Image correlations are 0.764 / 0.758 / 0.812 and edge cosines 0.510 / 0.513 /
0.535. These different maps cannot establish a before/after improvement. The
linked `recorded-weak-input-comparison.mp4` is a recorded diagnostic, never a
runtime source of neural activity. Across this reel, 85.6% of complete-projection
light comes from cells requiring incoming events; 14.4% comes from directly
stimulated visual inputs. This describes light contribution, not information origin.

`source-test.html` on the development server separately checks H.264 MP4 File
playback, a generated portrait WebM File, aspect-preserving letterboxing, and
recovery after invalid input. It passed using the actual browser decoder.
The native picker itself remains unverified because its Open button was disabled
in the automation session. File-path decoding is now tested separately from that
picker issue; no extension permissions were widened.


That 320-control production version measures 10.76 new displays/s in the paired view
and 10.79 in the enlarged view, with mean processing latency 73.4 / 72.7 ms and
p95 83.5 / 81.8 ms. All 270 anatomy chunks and the 1280 × 960 complete-branch
cache are present, with native body physics active. Each window contains 120
distinct source times. Decoder-to-draw means are 88.5 / 88.0 ms, excluding physical
scan-out. Results are in `weak-input-live-performance.json`; speed is still
separate from the unmet fine-detail recognition criterion.


## September 15 checkpoint

The current 640 × 480 controller has a frozen 120-frame File-object evaluation
with maps 05, 09 and 16. All decoded targets and emitted frames were distinct;
85.7% of complete-projection light came from cells requiring incoming events.
Cutting transmission after reset silenced those cells. Fine edge recognition
still fails, and production performance at this finer setting is not yet
validated. An initial live readout was below the 10-frame/s target.

**Save measurements** exports the latest 120 completed neural GPU draws, source
frame identities, latency, geometry counts, physics state and causal counters as
local JSON. Pause/resume, source and viewpoint changes, and interventions start
new measurement windows without resetting neural history. Partial windows are
explicitly identified. The export's statistics pass unit checks; its download
interaction still needs a complete browser verification.

Generated datasets, anatomy/operator files, videos, checkpoints and validation
caches are excluded from Git. Recreate them using the preparation commands above;
a fresh source checkout does not contain the multi-gigabyte brain assets.
