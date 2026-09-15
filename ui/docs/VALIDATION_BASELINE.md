# Validation — 2026-09-14

- `npm run build`: passed TypeScript and the two-page Vite production build. The shared Three.js chunk exceeds Vite's default 500 KB advisory threshold (143 KB gzipped).
- `npm test`: **4 neural tests passed** (9.24 seconds) and **5 browser tests passed** (9.7 seconds), Chromium with ANGLE Metal. Projection desktop 1440 × 1000, embodied desktop 1600 × 1100, mobile 390 × 844.
- Integrity check: 1,500 distinct neuron identities, 768,000 finite point entries, source skeletons totaling 3,060,716 vertices, and all 512 frame tiles preserve 4:3 content with black side padding.
- Default reference footprint: **95.32%** of 160 × 90 cells contain one or more point samples. This is occupancy, not perceptual image quality.
- Performance observation: Apple M5 Pro, Chromium ANGLE Metal, all points loaded, default 10 FPS playback; approximately **60 render FPS / 10.0 frame updates per second**. Headless software rendering was about 2 FPS before switching the browser test to the hardware backend.
- Static design detector: no findings in the changed UI targets.

Screenshots produced by the browser tests are stored locally under `.cache/screenshots/` (ignored by git).

## Neural model and body verification

- Unit intervention checks: delayed excitatory propagation across two edges; inhibitory suppression; synaptic ablation; deterministic replay after seeded reset.
- Actual full graph: 138,639 neurons, 15,091,983 weighted connections. Stimulation recruits identified descending cells; cutting connections eliminates their response while externally stimulated cells still spike.
- All 1,500 displayed neuron identities map to this graph. All 44 official body assets passed SHA-256 verification against the committed manifest. Published source Git blob hashes were verified against the pinned author repository tree.
- Browser integration: live spike traces and body state, pause/resume, reset, disabled synaptic delivery, direct DNp09 pulse, encoder disclosure, paused-source scrub with a byte-identical neural canvas until simulation resumes, reduced-motion initial pause, mobile overflow checks, and existing projection interactions all passed. No JavaScript errors in the tested paths.
- Production preview at port 4173: the separate public MuJoCo module/WASM and gzip connection table load successfully. The source direction contract survives the build.
- Earlier luminance-encoder production body comparison (50 ms trace; retained as integration evidence, not current-controller measurements): three seeded trials were sampled at exactly **200 ms simulated time**. Relative to synapses disabled, a direct DNp09 pulse changed sampled joint angles by **0.1052 rad RMS**; normal video stimulation with the graph enabled changed them by **0.03447 rad RMS**. Both differ from passive settling with no neural motor drive. These are 42 sampled internal joint coordinates, not the 42 actuator list. See [raw results and protocol](public/data/body-causal-validation.json).
- Earlier live user Chrome observation with the luminance encoder: approximately **0.09× simulation speed and 119–120 render FPS** after more than four simulated seconds. This is a local M5 Pro measurement; neural content advances at roughly one source frame per wall-clock second, not 120 frames/s.

These checks establish implementation behavior and causal coupling, not living-fly accuracy. The DN-to-gait adapter is engineered, the visual encoder is artificial, and body feedback is absent.

## Image-control experiments

- Historical independent whole-neuron brightness fitting: checkerboard 7.47 dB PSNR; three DOOM frames 17.83, 28.56 and 28.74 dB. This ignores dynamics and scores supported pixels only.
- Historical 37-group stimulation controller: checkerboard 3.22 dB; three DOOM frames 3.60, 12.31 and 12.23 dB. It failed to reconstruct recognizable DOOM and remains a comparison option.
- **Current default:** 8,534 disjoint presynaptic controls, full published graph, one measured 100 ms spike trace per displayed neuron. No direct stimulation of displayed or identified motor-output neurons. One geometry contribution per whole neuron; 40 × 30 grayscale, density normalization, fixed −15° camera. The current interactive renderer and numerical operator share that spatial basis. Geometry supports 99.5% of the footprint.
- **64 consecutive-frame evaluations:** four clips, 300 ms seeded warmup per clip, then 16 frames at 100 ms each with persistent state. Median PSNR 24.96 / 24.67 / 16.96 / 24.74 dB. Mean image error is 58.9% / 62.6% / 76.2% / 61.2% below a per-frame flat-brightness baseline. The fire clip is visibly poorer; gun/HUD details remain blurred. All 1,200 pixels are scored. Controller/view selection used the same dataset, so this is engineering validation, not held-out generalization.
- **Disconnection:** all 1,500 displayed outputs remain exactly zero after reset and 300 ms of stimulation with synapses off; 647,944 upstream spikes still occur.
- **Wiring intervention:** a separate 500 ms frame test gives intact MSE 0.003663, flat baseline 0.007303, disconnected MSE 0.060002, and reassigned-target MSE 0.030496 (8.33× intact). All 24,427 controller coefficients are checked against the actual signed contact counts.
- **No source bypass:** changing the source while paused leaves the neural canvas byte-identical. Resuming the simulator changes it. The dedicated activity render pass has no video/target texture.
- The [comparison](public/data/synaptic-control-comparison.png), [recorded playback](public/data/synaptic-playback.gif), and [complete numerical report](public/data/synaptic-control-validation.json) contain actual simulated outputs. GIF playback uses simulation time, not live throughput. The four 1.6-second sequences took 10.3–12.7 wall seconds each on the development M5 Pro, excluding warmup and body simulation.
- Strong optical rates up to 4,000 Hz and zero refractory period on optical targets are disclosed. These results do not validate natural vision, living-fly stimulation, or biological behavior. See [RESEARCH.md](RESEARCH.md).

## Finish review

Disposition: **ship**.

| Prior finding | Result |
| --- | --- |
| Projection coverage | Resolved — front capture now includes the weapon and lower HUD within the central anatomical footprint. |
| Source aspect and labeling | Resolved — desktop and mobile previews preserve 4:3 content with visible side padding and the “Dataset” label. |
| Mobile accessible names | Resolved — source confirms persistent `aria-label` values for Save image and How it works. |

Remaining: clear. No material regressions from the fix batch.

## Embodied-page finish review

Disposition: **ship**.

| Prior finding | Result |
| --- | --- |
| Stimulus scaling | Resolved — source fills the monitor on desktop/mobile, preserving aspect and pixel edges. |
| Keyboard viewpoints | Resolved — both specimens have visible, labeled native camera menus. |
| Product persistence | Resolved — PRODUCT.md describes both experiments and scopes projection-only constraints. |

Remaining: clear. No material regressions from the fix batch. DESIGN.md retains the original visual world and records the new surface's patterns.

## Synaptic-controller finish review

Disposition: **ship**. The bounded review found no material regressions. Desktop and mobile retain the original visual system; the controller, reference viewpoint, measured-spike renderer, strong stimulation limits and recorded evidence are clearly distinguished. DESIGN.md and its sidecar now record the new mode and viewpoint patterns.

The final production preview loaded the compressed full graph, new operator/controller assets, activity renderer and MuJoCo body together. A corridor clip reached 500 simulated ms with nonzero synaptic deliveries and a progressing physical body state; no JavaScript errors were reported. The live Chrome page is left on the synaptic controller with the corridor clip selected.
