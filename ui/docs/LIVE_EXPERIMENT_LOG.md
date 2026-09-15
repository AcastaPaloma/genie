> Historical working log from 2026-09-14. Its open issues and numbers describe earlier experiments. See [current progress](../LIVE_PROGRESS.md) and [final validation](../VALIDATION.md).

# Live synaptic display — goal remains open

Working record, 2026-09-14. This document describes an unfinished experiment;
`VALIDATION.md` still records the earlier 1,500-neuron, 40×30 baseline.

The active goal requires arbitrary local DOOM clips, recognizable detail above
the baseline, and at least 10 *new neural outputs* per second including the
connected simulated body. Display brightness must come only from the owning
neuron's actual simulated spike history. It must not be replaced by target
pixels, a decoded image, or recorded neural output.

## Implemented and under test

- `live.html`: local browser video decoding and latest-frame scheduling, WebGPU
  full graph in one worker, MuJoCo body physics in another. User videos remain
  local. Distinct decoded-frame throughput is separate from screen refresh.
- Full measured model: 138,639 cells, 15,091,983 weighted directed edges.
- 43,992 real FlyWire neuron geometries / 11,261,952 branch samples downloaded
  with provenance. A fixed, video-independent real branch observation site per
  neuron provides 99.77% support of a 128×96 raster. The full-arbor view uses the
  same per-neuron activity. Observation sites are not somas.
- Strong, explicitly artificial upstream voltage clamps generate pulse trains
  on a disjoint set of 59,227 cells; displayed and identified motor cells are
  never clamped. Incoming currents on actuators are canceled by those clamps.
  This is not natural vision or an unmodified Shiu stimulation protocol.
- LIF dynamics retain .2 ms steps, 1.8 ms synaptic delay and the original model
  threshold, reset, time constants and ordinary-cell refractory period.
- Batched GPU integration, subgroup spike queues, 64-bit event accounting,
  and elimination of additions that are immediately canceled by ideal clamps.
  All measured synaptic events are still counted. No graph weights are changed.
- Body timestep .25 ms is experimental and still needs a longer comparison
  against the .1 ms baseline. Body follows the same sequence, one worker stage
  behind the neural display. DN-to-gait control remains engineered.

## What has not passed

**Do not call this goal complete.** The actual moving live page remains too
noisy and its late throughput has been about 8–9 new frames/s on this M5 Pro.
Static frames held for seconds reconstruct substantially better, which is not
proof of responsive video performance. Native Freedoom map01 is the development
sequence. Map02 and map03 are reserved for held-out sequence tests and have not
yet passed them.

The shader's small fixed light blur averages emitted light only. It is a display
convention, not a measured biological optical response. Raw emission can be
inspected. Greater target raster size alone is not evidence of resolvable detail.

## Current diagnosis and experiments

The initial mean-current controller omitted current accumulating during the
refractory period. Its request for 240 Hz produced about 357 Hz in a constant-
input isolated-cell check. The corrected discrete inverse gives about 238 Hz.
That correction is in the pixel controller, but did not by itself solve moving
image fidelity.

The controller's coupled inverse and spike variability remain the main limits.
Numerical results are under `.cache/large-control/sequence-*.json`; these are
validation records, never playback sources. All development sequence runs retain
neural state across 48 consecutive 100 ms inputs.

- FISTA / 43,992 neurons / 128×96 / 240 Hz / 40 ms observation trace:
  increasing the current-gradient limit and maintaining eight iterations at
  each feedback update improved mean raw MSE to about .01875. Standalone neural
  computation averaged about 92.8 ms; this excludes video decoding/render/body.
- Dense ADMM inverse: derived from H B only, no video-dependent precomputation.
  The 604 MB inverse passed a sparse residual check (relative error 6.6e-5).
  Mean raw MSE .01966, but computation was about 140–185 ms later in the run.
  Experimental only; not integrated into live playback.
- A 19,139-neuron, one-cell-per-pixel 160×120 selection with 94,706 upstream
  controls did not improve quality enough. It is an experimental alternative,
  not a claim that 43,992 neurons are shown in that variant.
- Last-interspike-interval brightness and a saturated pulse-response control
  approximation were worse in sequence tests. They remain benchmark-only.
- More aggressive PI feedback raised activity/cost and did not solve fidelity.
- Current experiment: penalize actual upstream firing rates, instead of only
  normalized current coefficients, to reduce unnecessary high-rate stimulation.

The Vite file watcher previously consumed roughly four CPU cores while idle;
nonpolling watching and exclusions for cached/public datasets fixed that.
New public assets require restarting Vite because that directory is excluded.

## Remaining acceptance work

Moving-frame fidelity and measurable detail gain; ≥10 new outputs/s with body;
held-out clips; local upload/codec failures and seek/cut handling; GPU equivalence
and intervention tests (cut and shuffled wiring); fixed-site identity integrity;
longer body stability; desktop/mobile visual review; production build and
updated public documentation. The old tokenizer training on the remote H100
machine is not to be interrupted for this experiment.

### Later findings

- Penalizing actual upstream spike rates helped computation: penalty 1e-5
  roughly halved spikes without worsening frame MSE; 1e-4 reduced standalone
  neural time to about 61 ms with some loss in accuracy.
- An independent 100 ms per-neuron spike count produced essentially the same
  spatial grain as the decay trace. The dominant error is control allocation,
  not merely display filtering.
- Frozen-request offline convergence: the regularized controller request had
  RMS current error 42.93 online and 41.88 after L-BFGS-B convergence (416
  iterations). Removing all penalties still left RMS 39.29 after 1,500
  iterations; that unpenalized run hit its iteration cap and is not a proven
  global feasibility bound. These tests point toward actuator/observer choice.
- A selected set of 6,901 cells with negative outgoing weights in the published
  model gave a materially better 96×72 moving reconstruction: mean raw MSE
  .009724, fixed .6-cell light-blur PSNR 22.05 dB, mean image correlation .847,
  and standalone neural compute about 56 ms. No positive recurrent drive comes
  directly from these displayed cells. This is a circuit-selection experiment;
  all physical connections and weights remain in the simulator. It is not yet
  the high-resolution live deliverable.
- Expanding that anatomy pool to 30,000 real intrinsic model-negative neurons
  is in progress, prioritizing cached and central anatomy. Intended next test:
  160×120 fixed sites selected from that pool, with remaining upstream cells
  available to stimulate. This changes which real neurons are observed, not
  their coordinates or connections.
