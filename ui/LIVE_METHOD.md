> Historical selected-site experiment. These measurements and method details do not describe the complete-brain main view. See FULL_BRAIN_PROGRESS.md and QUALITY_BAR.md.

# Live synaptic display: implementation and limitations

## State and causal path

Local video → grayscale target → external pulse controller → upstream voltage
clamps → full delayed spiking graph → displayed cells' own spike traces → real
anatomical sites. The renderer has no access to target pixels in its activity
accumulation path. `live-neural.worker.ts` sends measured activity and descending
motor readouts; `live.ts` supplies the source only to the controller worker.

The simulator follows the published model's simplified LIF equations and signed
contact weights. Source: [Shiu et al. model](https://github.com/philshiu/Drosophila_brain_model),
revision `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`; see
[the paper](https://doi.org/10.1038/s41586-024-07763-9). This application changes
stimulation to ideal deterministic voltage clamps and chooses a different
observation trace. It is not the original experimental protocol.

Each ordinary neuron retains rest/reset −52 mV, threshold −45 mV, membrane time
constant 20 ms, synaptic time constant 5 ms, contact gain .275 mV, .2 ms time
steps, an 11-tick refractory counter, and a nine-tick propagation delay. Clamped
actuators have enforced pulses and zero refractory time; incoming synaptic
currents on them are canceled. Displayed cells and named motor outputs are
excluded from actuation. Counting transmitted weighted edges is distinct from
counting the anatomical contacts represented by those weights.

GPU batches never cross a synaptic-delay horizon or scheduled feedback boundary.
They evaluate every retained graph edge required by the spike queues. Additions
to clamped cells can be omitted because the clamp immediately cancels them;
event counts still include those deliveries. CPU/GPU chain tests and a direct
optimized/unoptimized comparison test this equivalence. No edge sampling is used.

## Anatomy and control selection

29,998 intrinsic neurons with negative outgoing weights in this model were
packaged from public FlyWire v783 binary skeletons. Two requested roots returned
404 and are recorded as failed downloads; no fabricated substitutes were made.
A −45° fixed reference view, 300 × 225 μm footprint and maximum bipartite matching
choose actual branch samples for raster coverage. For each supported cell of
the 160 × 120 raster, one candidate neuron is selected by strong, relatively
localized positive and negative incoming contacts from outside the candidate
observer pool. No DOOM data enters camera, site, or neuron selection.

The final 18,901 neurons cover 98.44% of the raster. Each has 256 deterministic
samples along genuine skeleton edges for the whole-arbor comparison. The site
view uses one of those samples, retaining its original 3D position and owner.
The metadata records source root IDs, source hashes, coordinates, sampling,
parent indices, and packaged hashes. Source:
[FlyWire anatomy](https://zenodo.org/records/10877326) and the public
[FlyWire skeleton service](https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/).

Choosing model-negative observers prevents positive recurrence originating in
that display population. It does not prove biological inhibition: the model's
neurotransmitter-to-sign assignments and single-compartment cells are simplified.
The complete graph still runs, including undisplayed cells.

There are 106,621 eligible upstream actuators, approximately 77% of the model.
Consequently this is a heavily controlled display experiment. It does not
establish that restricted sensory stimulation can make a natural brain draw
arbitrary video frames.

## Online inverse control

Let `H` map each observed neuron to its fixed raster cell, and `B` contain signed
contact strengths from the disjoint actuator population. Every 10 simulated ms,
the controller measures `H r`, estimates uncontrolled recurrent input, and updates
a desired current using feedforward inversion plus bounded PI feedback. Sparse
projected FISTA solves a constrained, regularized current problem using `H B`.
The resulting pulse rates are applied only to the actuators.

The feedforward inverse includes synaptic-current accumulation during the
refractory period. Omitting that accumulation was a real bug in the earlier
experimental controller: a requested 240 Hz gave about 357 Hz in an isolated
constant-current check, versus about 238 Hz after correction.

The deployed parameter file is `public/data/live-profile.json`. The controller
uses 32 iterations on a new target and four at subsequent feedback steps, a
current-gradient limit of 50, and an upstream firing-rate penalty of 3e-5.
Its integral is attenuated at a target change; the neurons are never reset per
video frame. Failed experiments, including the dense ADMM inverse and mixed-sign
43,992-neuron display, are research artifacts rather than live fallback paths.

## Observation and body

Brightness is the owning neuron's 40 ms exponential spike trace divided by a
300 Hz reference, clipped to [0,1]. It is a display convention, not a measured
calcium signal. A fixed 3 × 3, .6-cell Gaussian option blurs emitted light and
normalizes by anatomical support; it never references the target. Raw sites and
whole sampled arbors can be inspected. Larger arbors inherently spread one
neuron's scalar activity over many image locations.

DNa02, DNp09 and MDN outputs feed the same explicit adapter as the earlier body
experiment. The NeuroMechFly scan-derived model and official generated browser
assets run in MuJoCo in a separate worker at .25 ms. Forward, backward and turning
commands remained finite for 30 simulated seconds; trajectories differ from the
.1 ms baseline. Numerical stability is not biological validation or a claim of
trajectory equivalence. References: [NeuroMechFly](https://neuromechfly.org/) and
[MuJoCo](https://mujoco.readthedocs.io/).

## Boundaries of the claim

This demonstrates online synaptic image control in a simplified software model.
It is not a living fly, natural vision, an exact whole-brain biological simulator,
or a reconstructed brain–VNC–muscle loop. High-resolution source clips are reduced
to 160 × 120 grayscale. The reconstruction remains noisy and viewpoint-dependent.
The source video itself is never used to conceal a poor neural reconstruction.

Reference clips come from native-resolution ViZDoom rendering of Freedoom Phase
2. They contain game pixels only. Sequence arrays and comparison figures are
validation outputs; the live page never reads them as playback data. Sources:
[ViZDoom](https://vizdoom.farama.org/api/python/doom_game/) and
[Freedoom](https://freedoom.github.io/).
