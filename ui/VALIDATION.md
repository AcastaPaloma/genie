> Historical selected-site experiment. These measurements and method details do not describe the complete-brain main view. See FULL_BRAIN_PROGRESS.md and QUALITY_BAR.md.

# Validation — live synaptic cinema

Measured 2026-09-15 on Apple M5 Pro (16 GPU cores, 24 GB), Chromium/ANGLE Metal,
WebGPU adapter architecture `metal-3`. These are engineering checks of a software
model. They do not validate natural fly physiology or a living-fly intervention.

## Live-file path

A native **35-FPS, 320 × 240 Freedoom** clip was selected through the same local
file input available to the user in the production build at
`http://127.0.0.1:4173/live.html`. With the 160 × 120 / 18,901-neuron profile,
full graph, body physics and both renderers active, the 20-second run produced
218 neural outputs. Its rolling 120-output timing window measured:

| Quantity | Mean | 95th percentile |
|---|---:|---:|
| New neural frames/s | 10.84 | — |
| Capture → neural display submission | 53.8 ms | 56.7 ms |
| Decoded frame available → submission | 74.5 ms | 100.6 ms |
| Neural computation | 52.6 ms | 55.3 ms |
| Body computation per 100 simulated ms | 21.3 ms | 23.7 ms |

The decoder-availability measurement begins at the browser video-frame callback;
it includes waiting for scheduling and neural processing. It excludes decoder
startup, file open, physical monitor scanout and the 40 ms spike-trace smoothing.
Body computation runs in a separate worker, one sequence stage behind. Screen
refresh rate is never substituted for new neural outputs. Ordinary 10-FPS clips
limit distinct input updates to approximately 10/s. An earlier development-server
run measured 10.87 outputs/s, 76.3 ms capture-to-submission and 95.6 ms
decoded-to-submission. Both records are preserved; the table above is the final
production measurement.


Two complete-page held-out runs preserved neural history across their clip switch:

| Source | New neural frames/s | Capture → submission | Decoded → submission |
|---|---:|---:|---:|
| map02, native 10 FPS | 10.000 | 50.7 ms | 53.3 ms |
| map03, native 10 FPS | 10.004 | 52.2 ms | 54.9 ms |

A first run measured 7–8 FPS while the old `embodied.html` experiment had restarted
after development hot reload and was still running at 30 render FPS. After that
competing experiment was stopped, the unchanged live profile achieved the results
above. Both runs are retained in
[runtime and causal evidence](public/data/live-validation/runtime-and-causality.json).
The ≥10-FPS capacity claim is supported by the independent native 35-FPS file test;
a native 10-FPS source itself cannot supply more than about ten distinct frames/s.

## Moving-image fidelity

Each clip contains 96 native 320 × 240 frames at 10 FPS. The controller sees
160 × 120 grayscale. A clip begins from rest, without warmup; all 96 successive
100-ms simulation windows retain neural history. Map01 was used for development.
Map02 and map03 were excluded from controller, camera and observation selection.
No parameter changes were made after viewing their results.

| Clip | Use | Raw light MSE | Fixed-blur light PSNR | Mean correlation | Detail correlation above 40 × 30 |
|---|---|---:|---:|---:|---:|
| map01 | Development | .01687 | 19.89 dB | .733 | .426 |
| map02 | Held out | .00872 | 23.32 dB | .752 | .443 |
| map03 | Held out | .01053 | 22.52 dB | .808 | .456 |

All target pixels are scored, including geometrically unsupported cells. The
fixed light filter is .6 cells with a 3 × 3 kernel and anatomical-density
normalization. It has no access to target pixels. Fine-detail correlation compares
residuals after subtracting a 40 × 30 downsample/upsample image from both target
and measured light. It demonstrates retained detail, not perfect resolution.
The reconstruction remains visibly grainy.

For all three sequences, comparison against the current frame has lower MSE than
comparison against references offset by ±1 or ±2 frames. That resolves lag only
at 100-ms frame intervals; it is not a zero-latency claim.

A deliberately hard three-pixel checkerboard, reversed every 100 simulated ms,
also retained contrast above the old raster's Nyquist frequency: raw output
averaged .488 on desired bright sites and .219 on desired dark sites, with
correlation .492. Its MSE was .1124, showing substantial amplitude error despite
resolving some of this detail. This synthetic test was not used to tune parameters.

[Comparison image](public/data/live-validation/comparison.png) and
[full metric definitions/results](public/data/live-validation/sequence-summary.json).
The underlying arrays are validation records, never live playback sources.

## Causality and identity

- Cutting transmission and resetting leaves every displayed cell at zero activity,
  with zero synaptic deliveries. More than one million upstream spikes still
  occurred in each 100-ms cut test. The live UI intervention passed too.
- A seeded permutation of postsynaptic identities among display cells preserves
  edge weights, presynaptic degree, and actuator identities but disrupts wiring.
  On the same first 32 map02 inputs, raw MSE rises from **.007961 to .054655**
  (6.87×). This intervention is dev-only and never changes the shipped graph.
- All 18,901 observation points exactly match a sample on their declared parent
  neuron's real skeleton. Source root IDs, model indices and geometry ordering
  agree with controller output IDs. Packaged geometry hashes match their manifest.
- Actuator IDs are disjoint from displayed and identified motor-output neurons.
- Paused source seeking leaves both the neural image bytes and spike count
  unchanged. The activity renderer receives measured traces and anatomy only.
- CPU/GPU delayed-chain checks at 40 and 100 ms observation traces match spike
  counts and agree within .03 Hz. Reset reproduces cells and per-cell counts.
  The clamped-current optimization gives byte-identical state and identical
  delivery counts. The full-size synthetic chain passes event-counter checks.

## Body and UI

MuJoCo ran 30 simulated seconds of forward, backward and alternating turning
commands at both .1 ms and .25 ms. Both remained finite; the .25-ms version took
32.5 ms per 100 simulated ms on average versus 81.1 ms at .1 ms. Body height ranged
1.049–1.360 model units at .25 ms. Trajectories differ; this is a stability check,
not evidence of numerical trajectory equivalence or biological fidelity.

Local MP4 loading, unsupported-video error and recovery, preserved state on
pause/seek, raw-light toggle, camera presets, whole-arbor geometry switching while
paused, PNG export, and 390-pixel layout passed. Reduced motion starts paused.
Desktop and mobile renders were reviewed. Impeccable finish review disposition:
**ship**, with no material fixes.

`npm test`: **4 neural tests + 5 browser tests passed**, including the earlier
projection and whole-arbor routes. Historical validation is retained in
[docs/VALIDATION_BASELINE.md](docs/VALIDATION_BASELINE.md).

`npm run build` passed TypeScript and Vite compilation. The static build totals
about 272 MiB. Vite reports the shared Three.js chunk above its 500-kB advisory
threshold; this is not a build failure. The full live UI validation also passed
against the production build, including all controls and the timing test above.

## Reproduction

From `ui/`, with the dev server at `127.0.0.1:5173`:

```sh
npm test
node scripts/validate_live_ui.mjs
node scripts/validate_body_timestep.mjs
node scripts/validate_live_sequence.mjs live160-map02 'control&large&sites&width=160&selected=selected-inhibitory-selective-160&pixel&clamp&compensate&robust&rate=300&interval=50&trace=40&gradient=50&maintenance=4&initial=32&energy=.00003&steps=96&video=/data/live-clips/map02.mp4'
```

To repeat the production UI check, run `npm run build` and `npm run preview`, then:

```sh
node scripts/validate_live_ui.mjs 'http://127.0.0.1:4173/live.html?paused'
```

Repeat the last command for map01/map03; append `&shuffle` with 32 steps for the
wiring intervention, or replace `video=...` with `pattern=3` for the synthetic
detail test. `scripts/render_live_results.py` builds the comparison from recorded
arrays, never from desired neuron activities. `gpu-test.html` runs the separate
GPU numerical checks. Run GPU benchmarks sequentially, with other experiment
tabs paused, and avoid code changes during measurement.

The sequence harness streams records to a localhost-only collector. Large arrays
are kept out of browser DOM/layout; rendering a 100-MB JSON text node caused an
earlier benchmark export to crash, while the live application did not retain
those arrays. Complete measurement records are under `.cache/large-control/`.
