# Current live extension

The current 160 x 120 local-video implementation, ideal-clamp assumptions and literature links are documented in [LIVE_METHOD.md](LIVE_METHOD.md). The following records the earlier whole-arbor and controller experiments; its timings and counts are historical.

# Can synaptic activity draw DOOM and move a fly?

Research and implementation notes, 14 September 2026.

**The current prototype produces coarse recognizable grayscale DOOM sequences from simulated synaptic activity on real FlyWire anatomy, while the same brain model drives a physically simulated fly. A broad feedback controller succeeds where the first 37-group controller failed. This is artificial control in a simplified model, not a demonstration that natural fly vision or a living brain can draw video.**

This is a focused review of the relevant primary papers, author implementations, and current embodiment documentation—not a claim to have read every publication about Drosophila. “Body” here means a virtual body derived from a real fly scan. No living animal or physical hardware is connected.

## What the literature supports

| Source | Useful result | Boundary for this project |
| --- | --- | --- |
| [Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9), [author code](https://github.com/philshiu/Drosophila_brain_model) | A whole-brain leaky integrate-and-fire model uses measured connectivity, synapse counts and transmitter-derived signs to predict selected sensorimotor responses. | A practical starting simulator, not a complete physiological fly. Selected circuit validation does not establish arbitrary image control or comprehensive behavior. |
| [Lappalainen et al., Nature 2024](https://doi.org/10.1038/s41586-024-07939-3), [FlyVis](https://github.com/TuragaLab/flyvis) | Connectome-constrained visual networks trained on a motion task predict neural responses across many visual cell types. | A stronger future visual front end. Task training and dynamical assumptions contribute; wiring alone is not a complete visual algorithm. |
| [Wang-Chen et al., Nature Methods 2024](https://doi.org/10.1038/s41592-024-02497-y), [NeuroMechFly documentation](https://neuromechfly.org/) | A scan-derived articulated fly supports physical contact, locomotion controllers and sensory simulation. The current project also publishes a browser MuJoCo version. | The body and low-level controller can be reused, but connecting descending activity to that controller is an additional modeling choice. |
| [Braun et al., Nature 2024](https://doi.org/10.1038/s41586-024-07523-9) | Descending neurons recruit broader descending populations involved in coordinated behavior. | DNp09 and MDN provide relevant forward/backward command signals, but a single rate-to-speed rule cannot capture the population motor hierarchy. |
| [Yang et al., Cell 2024](https://doi.org/10.1016/j.cell.2024.08.033) | Descending steering circuits influence specific limb movements and gait phases. | DNa02 is relevant to steering. Our left/right gain adapter is a coarse engineering approximation, not the measured limb-level mechanism. |
| [Sapkal et al., Nature 2024](https://doi.org/10.1038/s41586-024-07854-7) | Different contexts recruit neural pathways that halt walking. | Natural behavior includes gating and stopping circuits; high activity does not universally mean more movement. |
| [Vaxenburg et al., Nature 2025](https://doi.org/10.1038/s41586-025-09029-4), [FlyBody](https://github.com/TuragaLab/flybody) | Another whole-body physics platform supports fly locomotion with trained controllers. | An alternative body/control stack, not evidence that a connectome alone supplies muscle commands. It is not the model used here. |
| [FlyGM, February 2026 preprint](https://arxiv.org/abs/2602.17997) | A connectome-structured graph policy can be trained for embodied locomotion tasks. | A learned graph policy, not an untrained biophysical emulation. Its results do not validate our LIF model or prove the image-display goal. |

The [FlyWire wiring diagram](https://doi.org/10.1038/s41586-024-07558-y) and [whole-brain annotation](https://doi.org/10.1038/s41586-024-07686-5) supply structural identities and connections. Structural reconstruction is distinct from measuring every cell’s physiology, current state, or response to an arbitrary movie.

### What existing “embodied brain” demonstrations actually do

[Eon’s March 2026 technical account](https://eon.systems/updates/embodied-brain-emulation) integrates Shiu-style brain dynamics, visual modeling and a MuJoCo fly. Its useful disclosure is that brain-to-body mappings are hand chosen, low-level controllers provide substantial behavior, and the implemented visual activity was largely decorative in the demonstrated system. It also describes missing internal state, learning and biological dynamics. That supports the feasibility of integration, not a claim that all movement emerges directly from reconstructed synapses. Our prototype is narrower still: recorded video is an external stimulus, and body feedback is not returned to the brain.

## The implementation you can inspect

Open [the neural experiment](http://127.0.0.1:5173/embodied.html) after starting Vite. The original [projection display](http://127.0.0.1:5173/) remains a separate mode.

```text
DOOM dataset frame
  → whole-neuron image target + feedback stimulation controller
  → 8,534 separate real upstream neurons
  → full published signed connection graph + persistent LIF state
       → each rendered neuron's spike trace → its actual anatomical branches
       → selected descending-neuron rates → engineered gait adapter
                                                → MuJoCo fly body
```

**Connectivity:** the packaged Shiu v783 data contain 138,639 neuron entries and 15,091,983 directed weighted connections, representing 54,492,922 anatomical synaptic contacts. These are different counts: a weighted connection can summarize many contacts between the same two cells. Every published connection row is retained. We do not invent links between neighboring screen pixels.

**Neuron dynamics:** `src/neural.ts` implements single-compartment LIF dynamics using the author model’s resting/reset voltage −52 mV, threshold −45 mV, membrane time constant 20 ms, synaptic time constant 5 ms, refractory period 2.2 ms, synaptic delay 1.8 ms and contact weight 0.275 mV. The integration grid is 0.2 ms. External stimulation uses the reference protocol’s strong voltage kick; optical input targets have no refractory period. This is a browser port inspired by the reference protocol, not a bitwise reproduction of Brian2 scheduling or a new biological validation. See the [pinned reference implementation](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py).

Spikes schedule delayed weighted input to their downstream cells. Excitatory and inhibitory weights retain their published signs. State persists across movie frames. Passive cells can sleep below a numerical tail threshold; this changes computation, not graph connectivity. Seeded reset makes the stimulation sequence reproducible.

**Video input:** the default controller takes each 80 × 60 luminance frame, averages it to 40 × 30, and solves for one target activity per displayed neuron. A second sparse solver and feedback loop control 8,534 real upstream neurons with excitatory/inhibitory connections to the display subset. Neither displayed neurons nor identified motor outputs is directly stimulated by the video controller. Desired activities remain control targets; the simulator must actually generate the observed spikes. The old luminance and 37-group encoders remain available as comparisons. Their anatomical XY encoding is not biological retinotopy.

Optical stimulation can reach **4,000 Hz requested input rate**. The optical targets use the reference model's zero-refractory protocol; ordinary output neurons retain their 2.2 ms refractory period. This strong stimulation is an engineering assumption. It is not evidence that real neurons can follow these rates, or that this protocol can be implemented in a living fly. The controls span much of the connectome and are not restricted to sensory neurons. The experiment demonstrates simulated synaptic transmission as the causal means of controlling an image; it does not show a fly naturally interpreting the scene.

**Light:** the displayed subset contains 1,500 actual intrinsic neurons. All branch samples of a neuron share `min(1, measuredRateHz / 60)`, obtained from a 100 ms exponential spike trace. The renderer accumulates actual 3D samples into 7.5 µm cells, adds their scalar activity and divides by anatomical density. The best image is at the fixed reference camera, 15° left of the original front view; the image solver keeps that reference when the viewing camera rotates. Zoom enlarges the anatomy raster without changing its cells. No DOOM texture or desired activity is used in this render pass. Density normalization and grayscale brightness are visualization conventions, not calcium measurements or optical measurements of tissue.

The separate original projection page still samples video color independently at branch positions. It is a different experiment and does not count as synaptic image generation.

**Body:** the official NeuroMechFly browser assets provide a micro-CT-derived fly with physical joints, contacts and adhesion. We use its legs-only, position-actuated variant: 42 leg joint actuators and six adhesion actuators. The upstream gait controller combines coupled oscillators with recorded step trajectories. Our adapter takes DNa02 left/right rates, DNp09 forward activity and MDN backward activity and turns them into left/right gait drive. The rate scaling, subtraction, saturation and gains are engineered. The brain does not directly reconstruct the VNC, muscles or individual joint commands. The body is from another specimen, not the fly whose FlyWire brain was reconstructed. Asset provenance is in [public/body/provenance.json](public/body/provenance.json); the [official implementation](https://github.com/NeLy-EPFL/flygym) documents the controller and model construction.

There is no prerecorded fallback movement. The physical model may pivot, twitch, settle, fall or remain still. Gravity and contact can cause settling even when the neural drive is zero. Flight, feeding, grooming and body-to-brain sensory feedback are outside this implementation.

## Can neural activity itself make the image?

Three successive experiments address that question. All use a 40 × 30 grayscale target and the same 1,500-neuron geometry subset. Their image operator is a density-normalized, linear combination of projected branch samples. Historical experiments A and B used the original front view and an analysis-only renderer. Experiment C uses the new 15° view and the same raster/density convention as the current interactive renderer. Numerical figures crop the 300 × 225 µm image footprint; the UI also shows surrounding anatomy. None measures full-brain image capacity.

Experiment A scores only the 97.83% of pixels supported by geometry; experiment B scores all 1,200 pixels. The tables therefore also differ slightly in evaluation domain and are not a controlled head-to-head comparison.

### Experiment A: give every displayed neuron independent brightness

First solve bounded least squares, `min ||Ha − y||²` with `0 ≤ a ≤ 1`. Each column of `H` is one neuron's projected branch contribution. This is deliberately unconstrained by neural dynamics: it asks whether this geometry can represent the image at all.

| Target | MSE | PSNR |
| --- | ---: | ---: |
| Checkerboard | 0.179213 | 7.47 dB |
| DOOM frame 1 | 0.016490 | 17.83 dB |
| DOOM frame 53 | 0.001394 | 28.56 dB |
| DOOM frame 101 | 0.001337 | 28.74 dB |

Inspect [target and fitted images](public/data/whole-neuron-fit.png) and [numeric results](public/data/whole-neuron-fit.json). Coarse DOOM structure is more tractable than the checkerboard. Dark scenes can also achieve favorable pixel-error scores without retaining fine detail. This is evidence about this sampled basis and objective, not a claim that the fly has learned to draw.

### Experiment B: optimize stimulation, then actually run the spiking model

Group the 1,024 stimulation targets into an 8 × 6 anatomical grid, producing 37 occupied control groups. Warm the brain for 300 ms at 80 Hz input. From the same saved state and random sequence, measure the next 100 ms response to increasing each group to 160 Hz. These finite differences define a local linear response model.

A bounded, regularized optimizer chooses stimulation rates between 0 and 160 Hz. We then apply those rates to the **full nonlinear spiking simulator**, rather than displaying the linear prediction or assigning its output directly to neurons.

| Target | Constant-stimulation MSE | Controlled spiking MSE | Controlled PSNR |
| --- | ---: | ---: | ---: |
| Checkerboard | 0.484753 | 0.476519 | 3.22 dB |
| DOOM frame 1 | 0.452306 | 0.436411 | 3.60 dB |
| DOOM frame 53 | 0.061040 | 0.058802 | 12.31 dB |
| DOOM frame 101 | 0.061918 | 0.059900 | 12.23 dB |

The improvement is small and the images are not recognizable DOOM. Inspect [actual simulated outputs](public/data/stimulation-validation.png) and [complete calibration, controls and results](public/data/stimulation-controller.json). The browser retains this controller as “Earlier 37-group controller,” explicitly marked as producing poor image fidelity. It uses a fixed number of optimization iterations and retains the ongoing neural state; its local calibration may work even less well as that state changes.

### Why that failure is informative, but not a proof of impossibility

A long neuron can illuminate desired and undesired parts of the image simultaneously. Nonnegative brightness cannot cancel unwanted branches. On top of this geometric constraint, recurrent spiking dynamics restrict which brightness patterns our chosen inputs can produce.

Our current local controller has only 37 controls for 1,200 image values at one response horizon. Its linearized response cannot offer 1,200 independent directions. More neurons in the simulation do not remove that specific control bottleneck. Additional independently timed stimulation, a different circuit, different viewing geometry or a different brightness model could change the result. The calibration also uses one operating state and finite positive perturbations, so it is a poor global inverse for nonlinear recurrent dynamics.

### Experiment C: broad presynaptic feedback and persistent movie sequences

The new controller chooses up to three strong inputs of each sign per displayed neuron, ranked by their selectivity within the display subset. Deduplication and exclusion of displayed/motor neurons leave **8,534 distinct control cells**, with 24,427 real weighted connections into the display subset. The full simulation retains every published edge, including recurrent and off-target effects.

For each frame, bounded coordinate descent solves the anatomical inverse problem. Every 5 simulated ms, the control loop compares desired rates with measured spike traces. A LIF-derived feedforward current, proportional correction and bounded integral correction form desired input currents. A second sparse nonnegative solve chooses rates for the real upstream inputs, capped at 4,000 Hz. This is online feedback, not a trained neural network. The renderer receives only actual output spikes. The matrices used by the optimizer approximate the control response; they do not replace the full recurrent simulator.

A nine-view geometry pilot selected the −15° yaw reference. At that view, 99.5% of the 1,200 target cells contain geometry. This choice was made using record 0 examples; the controller and view were developed with this dataset. The results below are engineering validation, **not a held-out generalization claim**.

Four clips were each reset and warmed for 300 simulated ms, then played for 16 consecutive frames at 100 ms per frame. Neural history persisted within each sequence. MSE scores all 1,200 pixels, including unsupported pixels; the baseline is a flat image at each target's mean brightness.

| Clip / zero-based frames | Median PSNR | Mean MSE | MSE reduction vs. flat |
| --- | ---: | ---: | ---: |
| Record 0 / 40–55 | 24.96 dB | 0.003289 | 58.9% |
| Record 12 / 32–47 | 24.67 dB | 0.003479 | 62.6% |
| Record 35 / 0–15, bright fire | 16.96 dB | 0.018145 | 76.2% |
| Record 71 / 48–63 | 24.74 dB | 0.003518 | 61.2% |

Inspect [source versus actual spikes](public/data/synaptic-control-comparison.png), [recorded playback](public/data/synaptic-playback.gif) and [all 64 frames, protocol and measurements](public/data/synaptic-control-validation.json). Walls, floor and bright events are recognizable, but the gun, HUD and fine features remain blurred. There is no color encoding. PSNR alone is not a perceptual recognition test, especially for dark images; these results support a coarse prototype, not faithful arbitrary video reconstruction.

**Causal interventions:** resetting with synaptic transmission disabled produces exactly zero activity in every displayed neuron, despite 647,944 upstream input spikes during the 300 ms test. A separate 500 ms frame test gives MSE 0.003663 with intact wiring, 0.060002 when disconnected, and 0.030496 when incoming destinations are reassigned among displayed neurons while preserving weights (8.33× the intact error). The control populations remain disjoint. Browser testing also changes the source while the experiment is paused: the brain canvas stays byte-identical until the neural simulation resumes. These checks rule out the tested direct-texture/target-brightness bypass and demonstrate dependence on the correct measured connections.

**Boundaries:** the controller intentionally dominates the desired image through broad artificial inputs. Real connectivity is causally necessary here, but successful image control is not evidence of autonomous neural computation of DOOM, natural sensory processing, or practical living-fly stimulation. Trials with 400 Hz stimulation and ordinary refractory limits on control cells reconstructed less well. Restricted sensory stimulation, more accurate physiology, finer anatomy and a controller with independently validated generalization remain further research, not delivered capabilities. FlyVis could improve a future behavior-oriented visual input stage; it does not automatically solve inverse image control.

## Timing, verification and limits

Brain and body advance together in 2 ms simulated chunks. MuJoCo takes twenty 0.1 ms steps per chunk. The app targets **0.1× real time**, and DOOM advances at 10 frames per simulated second: approximately one new source frame per wall-clock second at that target. The screen can render much faster; the displayed render FPS does not claim faster neural content. Actual speed is measured separately. Background tabs pause work.

Automated checks confirm delayed excitatory propagation, inhibitory suppression, reproducible reset, and a causal intervention on the actual full graph: cutting synaptic transmission eliminates the tested descending-neuron response while stimulated input neurons still fire. Browser checks cover live simulation, pause/reset, the transmission intervention, the separate forward-command pulse, paused-source independence, mobile layout and reduced motion. These validate software behavior, not physiological fidelity. See [VALIDATION.md](VALIDATION.md).

The result is a usable experiment with an auditable causal chain. It is neither a living fly nor an established reproduction of its mind. The coarse simulated image goal now has working sequence evidence and causal controls. Higher fidelity, natural visual input and validation against living-fly physiology remain open.
