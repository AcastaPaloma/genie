# Full public brain — measured progress

The complete brain import is done. The visual recognition goal is not done.
See QUALITY_BAR.md for the user’s acceptance criteria. The main home page now
opens the full brain; the earlier direct projection is archived at projection.html
and the selected-site circuit at sites.html.

## Complete source coverage

FlyWire v783 official IDs and annotations match all **139,255 neurons**. Every
released skeleton vertex and parent edge belonging to these official cells is
included: **268,278,432 vertices and 268,139,177 edges**, in 270 verified chunks.
There is no additional neuron selection or branch sampling. The archive’s 18
extra nonofficial roots remain recorded in provenance rather than being counted
as official neurons. The publisher’s LOD1 / DS2 source is preserved; it is not
original-resolution EM or membrane geometry.

The 5,355,543,468-byte skeleton archive is complete and matches publisher MD5
`a4c104776f33ec539ef859064c4de3df`. The official ID array, connection table and
annotation commit `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0` are pinned and verified.
The anatomy manifest SHA-256 is
`b5b9b5dd363731fb1d218e87ffd1408a411942c5ca344e698691bb667a70e82f`.

The graph has **15,091,983 connected pairs / 54,492,922 contacts**. Its overlapping
cells reconcile exactly to the Shiu model. The 616 additional official cells have
no listed incoming or outgoing contacts and remain explicit isolated model cells.
No wiring was invented for them.

Sources: https://zenodo.org/records/10676866,
https://zenodo.org/records/10877326,
https://github.com/flyconnectome/flywire_annotations/tree/v2.1.0.

## Rendering and timing, 2026-09-15

The 1280 × 960 front-view cache integrates **every branch** into 146,791,411
nonzero neuron/pixel contributions. It takes only the current measured activity
vector, never video pixels or a fitted brightness target. Against independent
SciPy projection of a neuron-specific calibration vector, maximum error was
7.15e-7 and RMSE 7.53e-8; inactive neurons produced no light. GPU projection took
about 9 ms. A new source-independent cache is an optical calculation from anatomy,
not prerecorded neural output.

The original full indexed geometry remains uploaded for static anatomy inspection
and orbiting. Front and orbit views use different transparent optical
approximations (cable-length integration versus indexed-line fragments). Rotating
still costs substantially more than the cached frontal view.

On the Apple M5 Pro / 24 GB machine, the complete front view plus the connected
native body sustained **9.994 completed neural frames/s on a 10-FPS source** over
the recorded 120-frame window after several minutes of playback. Average capture
to GPU-completion latency was **58.08 ms**, p95 **65.10 ms**, and p95 display interval
**111.3 ms**. Mean neural compute was 39.89 ms, the cache 9.7 ms in the sampled frame,
and body including transport 43.82 ms. These are not scan-out latency measurements
or a universal guarantee for arbitrary devices/camera views.

The preceding complete-geometry redraw path was about 3.8 FPS / 259 ms latency.
After renderer/control optimization but with browser-only physics, long playback
was still limited to about 7–9 FPS by the body.

Native physics uses the **same MuJoCo 3.9.0**, body XML, exact per-step gait controls
and 0.25 ms timestep as the browser. A 40-second numerical comparison gave first
frame pose error 4.22e-14 and stable posture in both builds. Longer contact-rich
trajectories diverged numerically, while posture ranges remained comparable.
Native physics averaged 19.21 ms in that isolated test. Doubling the timestep to
0.5 ms made the fly fall over and was rejected. The application labels the active
physics backend; static-only hosting can still use WASM but was slower here.

## Image quality: failed

The current output is still predominantly a glowing anatomical shape, not a
recognizable moving Doom scene. A 1280 × 960 render does not establish 1280 × 960
scene detail. Only about 39% of the enclosing source rectangle has anatomical
support in this camera.

The matched-frame comparison is `public/data/full-brain-783/full-controller-comparison.png`.
Its middle column is actual full-graph spike activity projected through the
complete 320 × 240 control operator. The right column is a clearly labelled,
optimistic independent-brightness diagnostic, **not simulated activity and never
used by the application**. Supported-region MSEs at source time 1.9 seconds:

| Clip | Actual spikes | Independent anatomical fit |
| --- | ---: | ---: |
| map01 | 0.02612 | 0.00853 |
| map02 | 0.01388 | 0.00190 |
| map03 | 0.01038 | 0.00148 |

Across the 60 tested frames, the retained fast feedback controller averaged
0.01382 supported-region MSE. An arbor-rate/current inverse (0.01653), reverse
sensitivity heuristic (0.02259), raised trace floor (0.01663), and uniform-white
rate calibration (0.01686) were worse and remain outside the default path.
The test clips used for these comparisons are no longer untouched holdouts.

The default only adjusts tonic currents on 27,849 fixed input cells. Their entire
arbors remain visible. All neurons retain ordinary thresholds, refractory periods,
and incoming synaptic currents. Displayed activity is the owning neuron’s 40 ms
spike trace; no target brightness is written into neural or renderer state.

## Verification and remaining work

CPU/GPU propagation, inhibition, delay, reset, tonic firing and full-size indirect
spike dispatch pass. Full-graph transmission cuts produce zero noninput spikes and
zero deliveries after reset, while input cells still fire. The four older neural
regressions also pass; their selected-site quality result does not apply here.
The GPU cache and native body have independent numerical comparisons. TypeScript
and the production build pass. Reset now clears the displayed input/noninput
counters and motor meters.

Remaining: recognize scenes and motion through actual neural activity, validate
new clips and uploads after the controller improves, and improve orbit performance.
A synaptic-input readout would be a different signal from own-neuron spikes; that
change has been asked as a user preference and is not enabled by default.

Raw measurements: `.cache/full-brain/validation/`. Reproduction commands and native
service setup: README.md. No living fly, VNC reconstruction, or biological validity
is claimed.

After the final production rebuild, a fresh 120-frame front-view window measured
9.99765 completed displays/s, 58.08 ms mean capture-to-GPU-completion latency and
64.6 ms p95. The native body was active, all geometry counts matched, and the
application reported no error. Recorded in `validation/final-build-performance.json`.
This recheck does not change the failed visual-recognition status.

## Additional subthreshold-control experiment (not the default)

`GpuSynapticGateController` requests rates from the complete anatomical inverse,
then adjusts subthreshold excitability on non-motor cells. A fixed 700-cell
excitatory seed population supplies tonic spikes. All 139,243 externally biased
cells remain visible; only the seeds receive a suprathreshold bias. The remaining
12 motor outputs receive no direct drive. With transmission cut after reset,
**every non-seed cell is silent**, including the subthreshold-biased cells. This
is a stronger causal requirement than merely counting unactuated motor cells.

The initial ±50/6.8 mV-equivalent control range did not produce recognizable
images. Inspection showed some dark-target neurons still firing at 300–450 Hz
while the inhibition saturated at −50. The fitted-rate reference itself was much
better (map01 supported MSE 0.00892 versus observed 0.03118). Widening negative
control to −1000 reduced some errors but drove membrane voltages hundreds of mV
below rest, causing slow recovery. This was not promoted into the main view.

Additional experiments retain actual threshold events and all incoming synaptic
currents, but use **engineered per-cell excitability control**. These are not
physiologically validated stimulation protocols or natural fly vision:

- A rest-seeking tonic controller avoids integral windup but does not adequately
  suppress fast recurrent activity.
- Optional external shunting conductance changes the passive membrane time
  constant, with reversal at rest. Its analytical integration preserves the
  existing threshold, refractory interval and synaptic-current model. Every
  non-seed cell still needs real simulated incoming events to spike.
- The original trace trigger had a positive firing-rate bias: a fading trace is
  not the time-averaged firing rate of an individual cell. A corrected trigger
  accounts for each spike's finite exponential brightness pulse.
- An outer image-error loop now adjusts rate references from the image formed by
  **observed spikes**, rather than assuming the independent anatomical fit can
  be achieved. The references never enter the renderer.
- Target-independent greedy seed selection tests broader excitation using the
  real positive outgoing contacts. All selected seed arbors remain visible.

Across the same 60 tuning frames (not held-out validation):

| Experimental controller | Mean supported MSE | Neural compute, ms* |
| --- | ---: | ---: |
| Wide tonic inhibition | 0.011630 | 48.0 |
| Rest-seeking tonic inhibition | 0.012739 | 57.8 |
| Shunting, 60 Hz / 40 ms trace | 0.010270 | 52.3 |
| Shunting, 10 Hz / 100 ms trace | 0.013991 | 51.6 |
| Corrected trace trigger | 0.010359 | 49.7 |
| Observed-image loop, 700 seeds | 0.007993 | 28.6 |
| Observed-image loop, 3,000 coverage-selected seeds | 0.008444 | 36.2 |

*Mean excludes the three frames used for expensive diagnostic state readback.
These timings exclude anatomical display and body. The harness now records audit
time separately. They are not measurements of full-application frame rate.

**All still fail visual recognition.** The lower numerical error is not sufficient
to promote these controllers into the default experience. The latest scientific
comparison is `public/data/full-brain-783/full-spike-controller-comparison.png`;
every output column there is measured simulated activity. The source occupies its
own reference column. No independent-brightness solution is shown as playback.

For every tested gate variant, cutting transmission after reset yielded zero
non-seed spikes and zero synaptic deliveries; the constant seed cells kept firing.
The shunt tests also pass an independent isolated-membrane analytical comparison,
reversible gating through a three-neuron chain, and the existing CPU/GPU propagation
and large-dispatch tests. The observed-image correction matches an independent
identity-operator calculation and leaves measured neuronal state bit-for-bit intact.

The nearly whole-population external control in these experiments is consequential:
139,243 cells receive external excitability control, and 700 or 3,000 of them are
constant suprathreshold seeds. Only the 12 identified motor outputs are completely
unactuated. A passing transmission cut proves synaptic dependence in this model;
it does not prove natural image processing, biological feasibility, or that the
connectome alone computes the image. The main controller remains the earlier,
explicitly disclosed 27,849-cell tonic-input model.

## Event-conditioned control and rate audit

The previous goal pass was progress: it changed the full renderer/body runtime and
produced controller-failure evidence. This pass follows that evidence rather than
treating another large neuron count or low dark-frame error as completion.

`scripts/audit_spike_reachability.py` finds 135,370 neurons reachable from the
fixed 700 seeds through positive measured edges, representing 93.3% of the
320 × 240 projection's optical weight. The remaining 3,885 neurons represent
6.7%. This is an optimistic necessary condition, not a controllability result;
it ignores inhibition, timing and attainable rates. It rules out disconnected
geometry as the primary explanation for the current image error.

The new controller can prepare cells at −45.2 mV (below the −45 mV threshold)
using an external conductance. It releases that conductance only when a spike is
requested and the measured net synaptic current is positive. Thresholds,
refractoriness, 0.2 ms simulation steps and 1.8 ms transmission delays remain.
This is a very strong, engineered control input, not a physiological claim.

On the same 60 tuning frames:

| Variant | Mean supported MSE | Mean neural compute |
| --- | ---: | ---: |
| Prepared conductance, original feedback cadence | 0.010591 | 55.6 ms |
| Event-conditioned, one-tick feedback | 0.008006 | 89.5 ms |
| Event-conditioned, at most three ticks per feedback | 0.008107 | 66.6 ms |

The three-tick version's supported-region image correlations at 1.9 seconds are
0.649 / 0.687 / 0.780 for map01 / map02 / map03. Stationary-target tests show
recognizable corridor structure at a fixed 1.5× exposure, but important detail is
still absent. Roughly 45% of strong requested rates in the first two stationary
frames remain below one-fifth of the request. Every neuron's requested and measured
rate is now exported by the diagnostic harness, rather than only the worst 40.

The geometry-derived 95% supported target window is 139 × 104 at (100, 58) in the
320 × 240 operator. It includes more of the complete source image but is too small
to meet the detail goal. It remains a diagnostic, not a replacement of the full
image experience. No neurons or branches are removed for this test.

The main UI links to the explicitly labelled `?controller=events` trial. Its text
distinguishes 700 tonic seed cells, 138,543 externally gated cells and 12 unactuated
motor outputs. It states that image information is imposed by the external gates
while the connectome supplies and constrains the spikes. Every emitted light
contribution still comes from the owning neuron's actual simulated spikes.

All tested gate variants produce zero non-seed spikes and zero deliveries after
transmission is cut and state reset. GPU checks pass for one-, three-, and nine-tick
batches against the CPU model, including inhibition, delay, deterministic reset,
subthreshold conductance, observed-image gradients and large indirect dispatch.
The production build passes. The first full-application trial measured 10.0 new
displays/s, 76.5 ms mean processing latency and 83.4 ms p95 with the complete
1280 × 960 cable cache and native body active. Cutting transmission in the actual
UI and resetting produced zero non-seed spikes and zero deliveries after 218.5
simulated seconds. Normal transmission was restored afterward. The overall
recognition goal remains open.

## Feedback from the emitted image

This goal pass is progress: it improves the controller and narrows the remaining
failure with a numerical certificate, rather than accepting frame rate as visual
success.

The event trial now updates its rate references using the image formed by actual
spike traces every 20 simulated milliseconds. On the same 60 tuning frames,
supported MSE decreases from 0.008107 to 0.005545 (31.6%). Mean isolated neural
compute decreases from 66.7 to 42.3 ms. The full application separately measures
10.0 new displays/s, 60.2 ms mean latency and 67.0 ms p95, including completed
anatomical draws and the concurrently connected body. These are local measurements,
not a general hardware guarantee. `event-feedback-live-performance.json` records
the measured window. The linked `event-feedback-comparison.png` compares matched
source frames and actual spikes, with a fixed 1.5× exposure on both activity views.
Fine scene detail remains weak.

The independent anatomical audit now reports a primal/dual certificate for the
box-constrained whole-neuron fit. After 1,200 iterations, its supported MSE bounds
are 0.008529966–0.008529999 (map01), 0.001901090–0.001901256 (map02), and
0.001482598–0.001482851 (map03). These diagnostic optima preserve sharper edges
than the current spikes. More independent image-fitting iterations alone therefore
cannot close the controller's gap. This bounds only the fixed 320 × 240 optical
model, not every possible neural display. It is never used as rendered activity.
The dual formula also matches a small independently solvable identity case.

In the three stationary-target diagnostics, 97.8–98.4% of the 320 × 240 operator's
emitted light comes from non-seed cells. This is an optical attribution, not a
claim about the origin of information: the external gates still encode the image.
Those cells require incoming simulated events. All branches and seed cells remain
visible. The recognition and held-out-input acceptance criteria are still open.

### Frozen-controller sequence check

After freezing the event controller, a new 640 × 480, 35 FPS game input was
generated from previously unused maps 07, 11 and 17. It contains two scene cuts
in one 12-second file. All 120 sampled source frames and all 120 emitted frames
are distinct; neural history is preserved throughout. The full-graph numerical
harness averages 40.7 ms neural compute, excluding renderer and body. A transmission
cut after reset again produces zero non-seed spikes and zero deliveries.

Within each scene, the best tested temporal alignment is the current source frame,
rather than a source frame 100–300 ms earlier or later. Motion-difference error is
lower than a no-motion baseline. Spatial quality still fails: mean image correlations
are 0.46 / 0.54 / 0.55, and interior edge cosines are only 0.30 / 0.29 / 0.26.
The first scene's supported MSE is worse than its constant-image baseline. These
results demonstrate input response, not recognizable general-purpose playback.
`evaluate_frozen_sequence.py` checks the frozen source hashes and exports the
recorded actual-spike comparison linked from the trial. It is a diagnostic movie,
never a runtime fallback. If these scenes are used for subsequent tuning, they
must be retired as holdouts. The original three clips are now explicitly marked
as tuning data in their manifest.

The browser extension rejected programmatic local-file attachment because file-URL
access is disabled. No extension permissions were changed. That particular UI path
remains unverified in this pass; the new file is also available as a normal example
stimulus. This does not block running or reviewing the existing full-brain trial.

The final production UI was then tested with that new reel through its example
picker. It decodes the native 640 × 480 / 35 FPS input and produces 10.76 new
neural displays/s, with 60.6 ms mean latency and 68.8 ms p95 over 120 completed
frames. Every geometry chunk is loaded and the native body remains active; no
application error is present. The trial is left on normal synaptic playback.
Build and GPU regression checks pass. The user-facing main trial and recorded
comparison are retained for review. The goal remains active because fine spatial
recognition still fails.


## Signed feedback, resolution check, and new holdouts

This goal pass is **progress**. It improves measured edges and dark transitions,
retains the full release, and rejects two tempting quality shortcuts on evidence.
The visual acceptance goal remains active and is not complete.

The controller now uses signed measured connections to support externally gated
cells that need more light but lack enough incoming excitation. This changes only
external rate references. Neural membranes, currents, spikes, all connections,
and all anatomical arbors remain intact. Support is allowed only when the current
measured-image gradient calls for additional light; stale unreachable requests
must not keep generating background activity after a cut to black.

On the 60 existing tuning frames, the final precomputed implementation reduces
supported MSE from 0.005545 to 0.005244 (5.4%). At the three matched final frames,
interior edge cosines change from 0.296 / 0.279 / 0.294 to 0.373 / 0.349 / 0.293.
These are tuning measurements, not held-out recognition claims. A fixed
normalization is precomputed from every measured connection once, reducing the
new controller's isolated mean compute from 53.5 to 49.6 ms. The feedback formula
is unchanged; float reduction order can cause small differences in later spike
trains. The independent GPU checks cover both normalization regimes, signed
connections, protected input roles, absent image demand, and read-only neural
state. All pass, along with the previous CPU/GPU timing and causal checks.

The unguarded support experiment emitted hundreds of unnecessary gated spikes
per 100 ms on black input. With the image-demand guard, the final version emits
zero new gated spikes in the last measured black interval (900–1,000 ms).
The remaining mean anatomical light is 0.00150, mostly from the visible constant
seeds. A 3,000-seed variant adds about 25 times as much late black-background
light and costs more time without a meaningful overall image-error improvement;
it is rejected. Stronger support and an edge-weighted image objective also fail
to improve the tuning result and remain disabled.

A new exact 640 × 480 control operator integrates all 268,139,177 released edges
and 268,278,432 vertices. It has 60,866,742 neuron/pixel contributions and SHA-256
`e288255d679438867fa933fa006d8059af1104fe8bb9c71705e86cb2ca179a51`.
The manifest verifies the same complete anatomy hash. The worker and numerical
harness accept an explicit `controlWidth=640`, validate its size/hash, and retain
every contribution. Compact float exports preserve all pixel and neuron values
within the local collector's size limit.

The finer control operator increases isolated compute to 73.8 ms. A common
320 × 240 reprojection of actual spikes has worse edges at all three matched
frames. An independent reprojection through the full 1280 × 960 render operator
also has slightly worse edges; MSE is mixed. Thus the apparent lower native-grid
loss is not enough to promote it. The source clips are natively 320 × 240, and
bilinear enlargement adds no source detail. A further 640-pixel test fits the
entire frame into a 275 × 206 region with 95.07% anatomical coverage; it remains
blurrier. Neither change is enabled in the main view. The 1280 × 960 complete
anatomical renderer remains active, as before.

After freezing the new controller, a separate input reel was generated from maps
04, 08 and 12, seeds 91621–91623, at native 640 × 480 / 35 FPS. Its SHA-256 is
`2e6f280d0538434434f622fb859664e96b0a5b39f2569dbb0deef941ca2028e8`.
All 120 tested input and output frames are distinct. Neural history persists
across both scene cuts. The isolated mean is 49.0 ms per 100 ms neural frame.
Each scene's lowest tested temporal-error lag is zero; motion error improves over
a no-motion baseline. Mean image correlations are 0.582 / 0.676 / 0.641, and edge
cosines are 0.393 / 0.378 / 0.317. All three beat their constant-image MSE baseline,
but fine structure and object recognition remain insufficient. These maps differ
from the previous frozen reel, so those metrics are not a before/after comparison.

The new recorded diagnostic is `recorded-support-comparison.mp4`; its evaluator
checks the input hash, exact controller parameters and frozen source hashes.
Preparation refuses to overwrite prior frozen input artifacts. Earlier evidence
is retained. The UI now opens this improved event controller by default; the
previous tonic-input model remains available at `?controller=tonic`. Both actual
spike comparisons are linked clearly, and the recognition limitation remains
visible. Full-application timing for this version is recorded below after testing.

The updated default production page was then run with the new reel and complete
body. Across 120 completed frames with 120 distinct source times, it measured
10.56 new displays/s, 66.2 ms mean processing latency and 76.3 ms p95. The p95
frame interval was 107.2 ms. From source decode to completed draw, latency averaged
81.1 ms with 121.9 ms p95; physical display scan-out is not measured. All 270
chunks, 139,255 neurons and 268,139,177 branches were present, the native body
advanced beyond 98 simulated seconds, and the app reported no error. These values
are in `event-support-live-performance.json`. The production build and all four
CPU neural/control regressions also pass. The browser GPU checks passed earlier
in this same pass.

A native-picker fallback was attempted for the local-file attachment test without
changing extension permissions. No native chooser appeared on the accessible
Chrome surface, so that attachment path remains unverified. The public evaluation
file works through the normal example selector. The user-facing output still
fails the fine-detail recognition criterion; the goal remains active.


## Visual afferents, dense spike coding, and motion review

This goal pass is **progress**, not completion. It adds a transparent input-population
experiment, measures it on frozen unseen clips, and makes the complete brain easier
to inspect at a larger size. The own-spike readout, complete public anatomy and
measured connections are unchanged. Recognizable fine detail still fails.

Stationary-target diagnosis shows that simply waiting longer does not remove the
controller gap. The final 20 intervals of a six-second-per-map test have supported
MSE 0.012567 / 0.005613 / 0.004607. Raising only the requested rate scale to 180 Hz
helps the third scene but worsens the first; increasing seed drive from 50 to 200
also fails to improve consistently. Neither change is promoted on its own.
An independent periodic-rate/noise calculation suggests that finite spike density
explains only part of the gap; it is a diagnostic approximation, not a network
simulation or a proven lower bound.

The optional new controller supplies bounded external currents to all 11,391
cells annotated as `flow=afferent` and `cell_class=visual`. These IDs are validated
against the pinned metadata and excluded from the motor population. The constant
700 seeds remain. Another 127,152 cells receive the existing external conductance
control, with 12 motor outputs unactuated. The current-to-rate feedforward rule
and feedback only set input current; normal membrane integration, thresholding,
refractoriness, incoming currents and actual spikes remain active. All input
arbors remain visible. This is external image control, not natural fly vision.

At rate scale 60 Hz the visual-input variant lowers mean tuning MSE to 0.004408.
At 180 Hz it lowers it further to 0.003382, versus 0.005244 for the default. Its
isolated mean compute is 56.3 ms in that tuning run; a CPU anatomical audit was
also running, so the subsequent complete-UI timing is reported separately.
Matched final-frame MSE is 0.015647 / 0.004020 / 0.003385 and edge cosine is
0.313 / 0.364 / 0.350. The first bright corridor becomes worse; the other two
improve. This mixed visual result leaves the previous controller as the default
and exposes the new one at `?controller=visual-inputs`.

The conditional anatomical audit now supports per-neuron box constraints. It
fixes all seed values to the measured values and zeros cells without an excitatory
path from any directly driven cell, while preserving the complete rendering
operator. On the map02 stationary target, seeds alone reach 135,370 cells and
yield certified MSE bounds 0.002595611–0.002595749. Seeds plus visual inputs reach
138,452 cells and yield 0.001946043–0.001946203. Another 803 cells remain unreachable;
no new edges are added. This is deliberately optimistic: reachable-cell activity
is free, and timing, inhibition, current limits and spike noise are ignored.
The general-box dual matches an independently solved small identity problem.
Actual controller seed IDs match the structural audit exactly.

The GPU regression adds a three-cell visual-afferent → gated-cell → unactuated
motor chain. All fire with transmission; only the afferent fires after cutting
and resetting; disabling stimulation too leaves all three silent. Optical
attribution then assigns all emitted light to that direct input. Signed-support
checks cover visual-input roles as well as gated roles. All previous neural
integration, delay, inhibition, batch-size and large-dispatch GPU checks pass.

The new frozen holdout uses maps 13, 18 and 24, seeds 91701–91703, at 640 × 480 /
35 FPS. Its SHA-256 is
`1fa64c015c2c93e6a0f532d6c4f597590bedddb8910b2585145de1c6e3fd9359`.
Controller parameters and six exact source snapshots are archived before the
new input is observed. The evaluator verifies those hashes. All 120 source frames
and 120 outputs are distinct; neural history continues through both scene cuts.
Mean isolated neural compute is 54.77 ms. The best tested lag is zero in every
scene, and motion error is lower than a no-motion baseline. Mean image correlations
are 0.760 / 0.747 / 0.654; edge cosines are 0.402 / 0.424 / 0.313. These maps differ
from older holdouts, so these values cannot establish a before/after gain.

Across the new holdout, the complete 320 × 240 optical projection attributes
82.89% of light to cells requiring incoming events, 16.08% to directly stimulated
visual inputs and 1.03% to constant seeds. This is emitted-light attribution,
not information origin: all image information still comes from the external
controller. Cutting transmission and resetting gives 208,482 visual-input spikes,
13,300 seed spikes, zero other spikes and zero deliveries. Residual dependent
optical fraction is numerical roundoff (~9e-16).

The public matched-frame comparison and `recorded-visual-comparison.mp4` contain
actual simulated spikes at a fixed 1.5× exposure. Neither is a runtime fallback.
The UI labels the new input population and optical fractions explicitly. Reset
clears these counters and their visibility. **Enlarge brain** moves the existing
canvas into a full-window modal with playback controls; returning or pressing
Escape restores the paired layout. No anatomy is reselected or repainted.
Production TypeScript and Vite builds pass. Complete-UI and picker checks follow.


The production visual-input page was tested with the frozen reel through its
example selector. In the paired view, 120 completed frames with 120 distinct
source times yield 10.728 new displays/s, 74.7 ms mean latency and 83.7 ms p95.
Decoder-to-draw mean is 91.4 ms (112.5 ms p95), and the p95 frame interval is
101.3 ms. Mean neural and body times are 55.3 and 35.7 ms respectively.

The enlarged view also produces 120 distinct frames at 10.735 displays/s, with
74.2 ms mean processing latency, 82.6 ms p95, 90.5 ms mean decoder-to-draw latency
and 112.0 ms p95. Its p95 frame interval is 103.6 ms. All 270 geometry chunks and
the full 1280 × 960 cache remain active; the body advances from 40.0 to 94.3
simulated seconds while the enlarged canvas is shown. No application error is
present. These local results are saved in `event-visual-live-performance.json`.
Pause works inside the modal. Escape restores the same canvas to the paired view
and returns keyboard focus to Enlarge brain.

The existing user-owned Neuroframe tab contained the older selected-site app,
paused on a built-in example. It was updated to the new complete-brain trial.
This also made the native file picker accessible without changing extension
permissions. The public evaluation MP4 is visibly selected in that picker, but
Open remains disabled and no file reaches the page; the picker was cancelled.
Programmatic attachment was already denied by the extension's disabled file-URL
access. Local attachment remains unverified; example playback is verified. No
browser security permissions were widened.


A final intervention in the production UI disables transmission and resets all
state. After 68.6 simulated seconds, there are 81,223,208 direct visual-input
spikes and 9,234,400 seed spikes, but **zero synapse-dependent spikes, zero motor
spikes and zero deliveries**. Dependent optical fraction is only roundoff
(9.6e-16). Reset visibly clears the new direct-input counter and hides stale
optical fractions. Normal transmission and external stimulation are restored
and state reset after testing. Evidence is in `event-visual-ui-causality.json`.
The new recorded movie visibly shows actual output responding to the unseen
scenes, but fine scene recognition remains below the acceptance bar. The goal
continues active; neither speed nor causal silence is treated as visual success.


## Using weak synaptic inputs without constant seed light

The previous goal turn was **progress**: it added the visual-input controller,
verified a new frozen reel, and measured the enlarged full UI at 10.7 displays/s.
This continuation is also progress. It improves the actual-spike image and black
transitions, adds a measured weak-input mechanism and covers the browser File
path. The goal remains active because recognition is still incomplete.

The visual afferents alone reach 138,446 cells through positive measured paths;
adding all 700 constant seeds reaches only six more. Every cell and connection
remains simulated in either case. Removing seed drive reduces late dark light
from 0.001460 to 9.90e-8, but slightly worsens tuning MSE (0.003382 to 0.003424),
so removing seeds alone is insufficient. The six newly unreachable cells remain
in the full model and anatomy; they are not discarded or assigned fake inputs.

The earlier external preparation leaves a 0.2 mV gap to threshold. Under the
existing passive membrane/synaptic-current equation, one 0.275 mV-equivalent
contact produces only about 0.043 mV peak depolarization from that prepared state.
A weak but real event can therefore fail to recruit a requested cell. A new,
bounded `gateBias` parameter changes only the external conductance bias/reversal.
It cannot reach the unchanged threshold; permitted bias is at most 6.99 mV above
rest, leaving a 0.01 mV gap. Contacts, thresholds, refractory periods, membrane
integration and brightness from the owning neuron's spikes remain unchanged.

The new candidate uses bias 6.99 and conductance 327,680, with zero constant seed
currents and the same 11,391 directly stimulated visual afferents. The greater
conductance preserves approximately the old maximum incoming-current tolerance
while a gate is closed. This is extremely strong external preparation and is
explicitly disclosed; it is not a natural-vision or biophysical-validation claim.
Another 127,852 cells receive conductance control and 12 motor cells remain unactuated.

All 36 isolated analytical shunt cases pass without a spike. In a three-cell
chain with a single contact on the first connection, the original bias gives
45 / 0 / 0 spikes; the new bias gives 45 / 40 / 40 with single-tick feedback,
45 / 39 / 39 with three-tick feedback, and 45 / 38 / 38 with nine-tick feedback.
Cutting transmission and resetting gives 45 / 0 / 0 for all cases. Invalid and
suprathreshold preparation parameters are rejected before a GPU device is created.
All prior CPU/GPU integration, inhibition, reset, observed-gradient, signed-support
and large-dispatch checks pass. `weak-input-gpu-validation.json` records this result.

The complete graph's mean tuning MSE becomes 0.003205 at 54.6 ms isolated neural
compute. Matched MSE is 0.014233 / 0.003742 / 0.003279; edge cosine is 0.362 / 0.411 /
0.374, improving all three relative to the prior visual-input trial. The optical
weight of highly requested but nearly silent cells falls from 3.79% / 1.85% /
1.36% to 2.01% / 0.56% / 0.38%. A cut to black leaves mean light 1.8e-12 after one
second and no new gated spikes in the final 100 ms. Stronger upstream support
improves the first matched scene but worsens two edge scores and the 60-frame
mean, so support gain remains 0.5. All results use the same full optical operator.

Six-second stationary tests settle at MSE 0.012575 / 0.003234 / 0.002745, with
small temporal standard deviations (~3e-5–6e-5). A gap to the independent
anatomical bound remains, so waiting longer and spike noise alone do not explain
all remaining blur. The original optimistic bound is not used as rendered output.

The new frozen input uses maps 06, 20 and 29, seeds 91801–91803, at 640 × 480 and
35 FPS. SHA-256 is
`13c6b52a6d935d006d6185753a28ae1f24ede345afdf7f0b59dad4f451b9e040`.
Its six controller source snapshots and parameters are archived before evaluation.
The complete-graph harness now supports an explicitly labelled File-object input
path, and this holdout uses it. All 120 targets and 120 neural outputs are distinct;
history persists across both cuts. The best tested lag is zero in every scene.
Mean correlations are 0.764 / 0.758 / 0.812, with edge cosine 0.510 / 0.513 / 0.535
and lower motion error than a static baseline. Mean isolated compute is 55.67 ms.
These are different maps from prior holdouts, not a before/after comparison.

Mean complete-projection light attribution is 85.57% synapse-dependent and 14.43%
direct visual input, with no seed light. A transmission cut after reset gives
208,472 direct visual spikes, zero other spikes and zero deliveries. The newly
linked movie contains actual recorded spikes and is never a runtime fallback.
The main UI now uses this improved trial, with the earlier 700-seed visual trial
at `?controller=visual-seeded` and the earlier event trial at `?controller=events`.

A new browser decoder test also passes H.264 MP4 File playback (640 × 480, 12 s,
12 newly decoded frames in the sampled interval), a freshly recorded portrait
WebM File (192 × 320), correct black letterboxing, invalid-file rejection and
subsequent recovery. It uses no camera/microphone or user file. No decoder change
was needed. This verifies File → object URL → decoder → controller inputs,
separately from native picker automation, which remains unverified. Results are
in `video-file-decoder-validation.json`. Complete production UI checks follow.


The production build passes, and the bare home URL loads the new parameters:
visual inputs, zero seeds, gate bias 6.99, shunt 327,680 and support gain 0.5.
The live UI verifies all 139,255 neurons, 268,139,177 branches and 270 chunks,
with the full 1280 × 960 cache and native body. On the new evaluation reel,
120 distinct source times yield 10.765 displays/s, 73.4 ms mean processing
latency and 83.5 ms p95 in the paired view. Decoder-to-draw latency is 88.5 ms
mean and 106.0 ms p95; p95 frame interval is 102.1 ms. Mean neural/body times
are 54.4 / 34.3 ms.

The enlarged view measures 10.785 displays/s, 72.7 ms mean processing latency
and 81.8 ms p95 over another 120 distinct frames. Its decoder-to-draw mean/p95
are 88.0 / 106.4 ms and p95 interval is 100.9 ms. The body advances from 85.1
to 193.9 simulated seconds while the enlarged view remains open. Return to body
restores the same canvas and keeps playback active. No application errors or
constant seed spikes are present. `weak-input-live-performance.json` records
these local measurements. Rendered room structure is clearer, but the full
fine-detail recognition requirement remains unmet.


The final production intervention cuts transmission and resets state. After
139.4 simulated seconds, the 11,391 direct visual inputs have emitted 165,242,097
spikes, while every other cell has emitted zero: no seed spikes, no gated spikes,
no motor spikes and no synaptic deliveries. Optical attribution is exactly 100%
direct visual input in this cut condition. Normal transmission is restored and
both simulations reset afterward. Evidence is in `weak-input-ui-causality.json`.
The updated default and recorded comparison remain available for review. The
complete image still fails the fine-detail recognition criterion; the goal is
not complete. A useful next question is whether the now-more-responsive controller
can exploit the existing finer complete-arbor operator; earlier 640-pixel rejection
was measured with the older, less responsive controller.


## September 15 source checkpoint

The finer 640 × 480 control stage was left as the default before this checkpoint.
Its existing frozen evaluation (`fine-control-holdout-v1-review.json`) uses maps
05, 09 and 16 through a browser File object, with 120 distinct input/output frames,
persistent neural history, zero non-input spikes after a transmission cut, and
85.7% synapse-dependent optical contribution. At the complete 1280 × 960 common
observation resolution, the matched tuning edge cosines improve from
0.120 / 0.102 / 0.067 to 0.141 / 0.136 / 0.080. These remain weak edge scores.

The current TypeScript production build, four existing neural regression checks
and three new measurement-window checks pass. Save measurements exports raw
completed-frame timing and application state, separately identifying partial
windows and restarting measurement windows at interventions and view changes.
The download interaction has not yet completed browser QA.

A first live finer-control readout was about 8 new displays/s and 119 ms processing
latency. It was not a settled exported window, so the prior 320-control result
must not be attributed to the new default. Full production timing, enlarged-view
verification and a new recorded UI intervention remain unfinished. The attempted
four-row GPU scheduling optimization was saved only in the ignored local cache
and removed from runtime pending numerical and performance validation.

This is a source checkpoint, not completion of the recognition or performance
goals. Multi-gigabyte generated assets and raw evaluation artifacts remain local.
