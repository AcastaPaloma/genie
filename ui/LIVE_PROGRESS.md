# Live synaptic display — validated implementation

> Superseded by the user's full-brain requirement. This file records the earlier
> selected-neuron milestone, which does not meet that requirement. Follow
> [the active full-brain work](FULL_BRAIN_PROGRESS.md).

Updated 2026-09-15. The production application is available at
[Live synaptic cinema](http://127.0.0.1:4173/live.html).

The current profile uses a 160 × 120 grayscale target and 18,901 real anatomical
observation sites, with the full 138,639-neuron / 15,091,983-edge model simulated.
The displayed neurons receive no direct external control: their measured spike
histories supply the light. Local video decoding and online stimulation control
accept new clips without a clip-specific preparation or training step.

## Verified outcomes

- The production build processed 10.84 new neural outputs/s from a native 35-FPS
  local MP4 with the connected MuJoCo body and both renderers active. Mean delay
  from decoded-frame availability to neural-display submission was 74.5 ms;
  this excludes physical monitor scanout and decoder startup.
- Two held-out, native 10-FPS DOOM clips sustained their source rate with
  53–55 ms mean decoded-to-submission latency. Neural state persisted across
  the clip switch. No controller or anatomy tuning used these held-out clips.
- Separate 96-frame sequence evaluations retained recognizable moving structure
  and measurable detail above the former 40 × 30 raster. Every frame, including
  startup, was scored; the light remains grainy.
- Cutting synaptic transmission and resetting leaves displayed cells dark while
  upstream cells still spike. Shuffling postsynaptic identities worsened raw
  reconstruction error by 6.87×. Every observation site matches its real parent
  branch geometry, and actuator IDs are disjoint from display and motor cells.
- Local-file input, pause/seek independence, invalid-video recovery, raw light,
  camera and full-arbor views, export, and mobile layout passed in production.
  Regression tests and the production build passed. UI finish review: **ship**.

## Limits of this result

This is an idealized software experiment with strong external control of 106,621
upstream neurons, including voltage clamps that cancel their incoming currents.
It is not sensory-only stimulation, natural fly vision, or a demonstrated living
fly procedure. The body uses an engineered descending-neuron-to-gait adapter;
it is not a reconstructed brain–VNC–muscle loop.

160 × 120 describes the target raster, not guaranteed effective resolution. The
fixed geometry covers 98.4% of it; fine contrast and fidelity vary with content.
A fixed light-only blur reduces speckle. Full-arbor mode spreads each cell's
activity along its real branches and has lower image clarity. Codec support and
throughput depend on the browser, available GPU and competing workloads.

[README](README.md), [method](LIVE_METHOD.md), [validation](VALIDATION.md), and
[source/light comparison](public/data/live-validation/comparison.png) describe
the supported configuration and evidence. Earlier unsuccessful approaches and
superseded open issues remain in the [historical log](docs/LIVE_EXPERIMENT_LOG.md).
