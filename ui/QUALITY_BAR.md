# Full-brain playback acceptance

The user's requirement is a recognizable scene with motion a person can follow.
An impressive neuron count, a large framebuffer, low loss on dark frames, or a
passing causal intervention does not establish that requirement.

Completion requires all of these:

- At the normal main-view size, room structure, camera movement and moving
  objects remain identifiable in the brain output without consulting the source
  preview. Inspect moving sequences, including cuts and bright/dark transitions.
- Sustain at least 10 completed neural displays per second on an input with at
  least that many distinct source frames. Count source identities and completed
  GPU work, not animation callbacks or repeated images. Report processing latency,
  frame intervals and the body timing separately.
- Test previously unexamined video sequences and browser-decoded local uploads.
  Preserve neural history across frames and clip changes. No clip-specific
  brightness targets, prerecorded neural output or source-image compositing.
- Import all 139,255 official neurons and all 268,139,177 released skeleton
  edges, including input cells, and keep all measured connections. Every neuron
  is simulated in every route. A fixed-camera cache must integrate every branch,
  and rotation must show the original anatomy.
- The main view draws one published anatomical region (default: the central
  brain, 50,328 neurons / 169,589,408 vertices; the optic lobes and
  photoreceptors are withheld from **drawing** only, by published super_class
  and cell_type, never by a coordinate cut). `?region=all` must still draw the
  complete brain, and the displayed region and its neuron count must be stated
  in the interface. Withheld neurons keep spiking and keep their connections;
  a region is a display selection, not a reduced import or a reduced model.
- Verify that disconnecting transmission silences non-input cells after reset,
  and record how much of the visible reconstruction depends on those cells.
  Directly stimulated cells must remain disclosed and visible.

Numerical comparisons supplement visual review. Report both full-frame and
anatomically supported errors, a constant-image baseline, scene/edge preservation,
and temporal correspondence. Masked scores exclude unsupported image regions;
they cannot establish reconstruction of the whole rectangular source image.

Current status: **not met**. The complete anatomical import and causal simulator
work. The image remains insufficiently recognizable. Drawing only the central
brain (2026-09-15) removes the two dense optic-lobe masses that dominated the
frame and lets the camera fill it with central neuropil; this is a legibility
change to the display, and it has not yet been re-measured against the
recognition criteria above. The independent
whole-neuron brightness fit is only an optimistic anatomical feasibility test,
never a substitute for actual neural playback.
