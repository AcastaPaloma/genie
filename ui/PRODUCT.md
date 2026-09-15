# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated by the user's request to implement one-shot. Implementation choice: TypeScript, Vite, and Three.js, with static local assets and no training-server dependency at runtime.

## Users

The project owner wants to see frames from the existing DOOM dataset formed by actual 3D fruit-fly neuronal geometry, and rotate that geometry to reveal the mechanism.

## Product Purpose

The primary live route must use the complete official FlyWire v783 brain: all 139,255 neurons and every released skeleton vertex and parent link, with their own simulated neural activity. The goal is recognizable arbitrary-video playback through measured synaptic connections, with a connected simulated body. Full-anatomy image quality and real-time performance remain experimental. Earlier selected-site, projection and small whole-arbor experiments are retained as labelled historical comparisons.

## Capabilities and Constraints

- Use real public fly neuron skeletons with source attribution and complete-release coverage audits. Do not substitute invented anatomy.
- Use real frames from this repository's DOOM data, not screenshots of the training comparison panels.
- The projection camera stays fixed while the viewing camera rotates.
- Every pixel contributing to the brain display comes from geometry; the separate source preview is explicitly labeled.
- The original projection page uses point/branch color projection, not neural simulation. The separate neural page derives one brightness per neuron from its simulated spike trace. Keep the distinction visible.
- Use the full published Shiu v783 signed connection table in the neural model. Explain the simplified LIF dynamics and artificial optical input; do not call it natural fly vision.
- The physical body is a scan-derived NeuroMechFly model in MuJoCo. Explicitly identify the engineered descending-neuron-to-gait-controller adapter and absence of a reconstructed VNC, muscles or body-to-brain feedback.
- Let the user interrupt video stimulation and synaptic transmission, reset state, and independently pulse forward-command cells to inspect causality.
- The main view includes all neurons, including externally stimulated cells. Distinguish their spikes from cells receiving only synaptic input. Identified motor-output neurons must not receive direct video stimulation. Target activities never reach the renderer.
- Keep target resolution, complete anatomy counts, anatomical support, one activity per whole arbor, spike filtering and tonic input assumptions explicit. Preserve disconnection, wiring-shuffle and paused-source checks. The earlier selected-site results do not establish full-anatomy performance.
- Retain unsuccessful earlier results as historical comparisons. Do not disguise direct brightness fitting or projection as synaptic computation.
- The live route targets at least 10 new neural outputs per wall-clock second on the measured M5 Pro backend. Source decoding, neural simulation, body physics, and screen refresh have separate clocks. Preserve neural state across frames; never substitute prerecorded neural output. The earlier baseline remains slower than real time.
- Existing model training must continue without changes. All UI code and packaged assets belong in ui/.

## Operating Context

Local desktop browser first, with responsive touch controls. The data server is accessible through SSH, but the built UI should work after downloading its packaged assets, without SSH or API credentials.

## Evidence on Hand

The project contains train/val/test ArrayRecord DOOM videos. Public FlyWire and MaleCNS resources expose real neuronal morphology. The user provided FlyWire as the anatomical visual reference.

## Product Principles

The rendered artifact leads. Make the source and method explicit. Preserve actual coordinates and neuron identity. Let exploration reveal the projection's limitations.

## Implementation Assumptions

The user has made the complete publicly released brain a hard requirement. A displayed subset, one observation site per neuron, or further branch downsampling is no longer acceptable for the primary experience. Preserve the complete released skeletons and openly distinguish the publisher's LOD/downsampling from original membrane or EM data. The 616 official neurons absent from the old model have no connections in the public table and remain explicit isolated cells. “Body” means a simulated body; no living fly is connected. High-detail recognition and ≥10 new neural frames/s are goals requiring whole-anatomy evidence, not claims inherited from the previous selected-site version.
