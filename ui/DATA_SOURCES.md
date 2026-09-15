# Data sources and attribution

## FlyWire anatomy

This experiment uses a sampled subset of FlyWire FAFB neuronal skeletons, materialization 783. Credit: **FlyWire Consortium; Dorkenwald et al. (2024); Schlegel et al. (2024)**.

- [FlyWire](https://flywire.ai/)
- [FlyWire dataset archive and license metadata](https://zenodo.org/records/10877326)
- [Neuronal wiring diagram of an adult brain](https://doi.org/10.1038/s41586-024-07558-y)
- [Whole-brain annotation and multi-connectome cell typing of Drosophila](https://doi.org/10.1038/s41586-024-07686-5)
- [Public skeleton endpoint documentation in fafbseg](https://fafbseg-py.readthedocs.io/en/latest/_modules/fafbseg/flywire/skeletonize.html)
- [Original neuron annotation table](https://github.com/flyconnectome/flywire_annotations/blob/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv)

License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), as identified by the source archive. This UI is an independent experiment, not an official FlyWire application.

Changes made: seeded selection of 1,500 annotated intrinsic neurons; 512 length-weighted interpolated edge samples per neuron; up to 100 original edges per neuron; centering; nanometer-to-micrometer conversion; Y-axis reflection; conversion to browser binary buffers; dataset-frame or false-color shading. No anatomical coordinates are warped to fit the target image. Neuron root IDs and per-skeleton SHA-256 hashes are retained in `public/data/brain.json`.

## DOOM frames

Source: [p-doom/doom-dataset on Hugging Face](https://huggingface.co/datasets/p-doom/doom-dataset), revision `4e5c7c6b6c60c7a91690d3b7e54cb7e62d6d1134`, the repository's local `data/test/data_0000.array_record`.

The dataset metadata specifies [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). This describes the dataset's stated license; DOOM names and game material remain attributable to their respective creators.

Changes made: extract the first 128 frames of records 0, 12, 35, and 71; resize 80 × 60 RGB frames to 120 × 90 with nearest-neighbor sampling; add 20-pixel black side padding in 160 × 90 tiles; package lossless WebP sprite sheets and preview covers. No learned reconstruction, generated imagery, or interpolation between time steps is used. Playback FPS is chosen by the viewer, not recovered from source metadata.

## Application dependencies

Three.js is distributed under the MIT license. Barlow and Barlow Condensed fonts are distributed under the SIL Open Font License. Dependency licenses remain in their packages; the font license is also included in `public/font-license.txt` for static distributions.

## Spiking model and synaptic connectivity

- [Shiu et al. (2024), A Drosophila computational brain model reveals sensorimotor processing](https://doi.org/10.1038/s41586-024-07763-9).
- [Author implementation and data](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960), pinned revision `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.
- Inputs: `Connectivity_783.parquet`, `Completeness_783.csv`, `model.py`; source SHA-256 hashes are in `public/data/connectome.json`. Source Git blob hashes were also checked against the pinned tree during preparation.

The connection table includes 138,639 neuron entries, 15,091,983 directed weighted connections and 54,492,922 anatomical contact counts. Preparation sorts outgoing connections into adjacency lists and stores signed contact counts without neuron/edge sampling. FlyWire data attribution and CC BY 4.0 apply. The Shiu reference code is MIT; a copy is in `public/data/shiu-model-license.txt`.

Changes: browser LIF port with an event queue and passive-cell scheduling; seeded artificial optical stimulation; local inverse-controller calibration; broad feedback stimulation of 8,534 disjoint upstream neurons; spike-trace visualization; identified descending-neuron readouts. These are our experiment, not published or validated extensions by the source authors. Input XY sampling is not real retinotopy. All 1,500 displayed root IDs map to the published model index; neuron IDs remain strings to avoid integer precision loss.

## NeuroMechFly physical body

- [Wang-Chen et al. (2024), NeuroMechFly v2](https://doi.org/10.1038/s41592-024-02497-y).
- [NeuroMechFly documentation](https://neuromechfly.org/), [FlyGym source](https://github.com/NeLy-EPFL/flygym).
- Adapted JavaScript source revision: `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`.
- Official generated browser assets revision: `0884af08981994543634563d95e9b1eb49945082`. All 44 asset paths and SHA-256 hashes are in `public/body/provenance.json`.
- MuJoCo WebAssembly runtime: the official bundled 3.9.0 build, distributed with those assets.

FlyGym and MuJoCo are Apache-2.0; license copies ship as `public/body/FlyGym-LICENSE.txt` and `public/body/MuJoCo-LICENSE.txt`. The body uses the supplied scan-derived meshes, flattened MJCF, recorded step tables, coupling parameters and physical contact model. This is the legs-only position-actuated browser variant, not a full reconstructed muscle or nervous system.

Changes: adapted scene loading and mesh rendering for the local Vite/Three.js application; removed automatic host-theme integration; seeded initial CPG phases; wired an engineered DNa02/DNp09/MDN rate adapter to left/right gait drive; synchronized brain/body simulation time. The upstream physical body is retained, including arena fixtures. The FlyWire brain and NeuroMechFly body are not from the same specimen.

The image-basis and stimulation-controller figures are generated from this project's explicit numerical experiments. They are analysis projections of geometry and simulated activity, not biological recordings. See [RESEARCH.md](RESEARCH.md) for methods and limitations.

## Synaptic image-controller assets

`presynaptic-controller.json` selects up to three strong excitatory and three strong inhibitory upstream inputs per displayed neuron, yielding 8,534 distinct control cells and 24,427 measured weighted connections into the display subset. Displayed neurons and identified motor outputs are excluded from direct stimulation. The controller stores normalized contact counts for optimization; the simulator still uses the complete original signed connection graph. Source/revision and the packaged graph SHA-256 are recorded in this asset.

`synaptic-image-operator.bin` projects the unchanged branch samples from a fixed view 15° to the left of the original front view. It sums contributions per neuron and divides by projected anatomical density on a 40 × 30 raster. `synaptic-image-operator.json` records the camera, coordinate scale, 99.5% support, geometry/operator hashes and pilot selection protocol. The interactive density-normalized renderer uses the same spatial cells and measured spike brightness.

`synaptic-control-validation.json`, `synaptic-control-comparison.png` and `synaptic-playback.gif` are this project's actual LIF sequence results. They are not source-author results, recordings of a living brain or VAE output. The GIF replays recorded simulated time and does not measure live wall-clock throughput. Rebuild scripts and evaluation limits are in README.md and RESEARCH.md.


## Live synaptic cinema extension (2026-09-15)

- FlyWire v783: 29,998 intrinsic model-negative neuron skeletons downloaded from
  the same public binary skeleton service. The final 18,901-neuron subset uses
  4,838,656 real branch samples and one fixed observation site per neuron.
  `public/data/selected-inhibitory-selective-160/brain.json` records exact roots,
  hashes, transformation and output ordering. `parent-anatomy.json` pins the
  complete preparation pool. Two missing roots returned 404; no substitutes.
- The graph remains the full pinned Shiu v783 graph. Derived controller matrices
  use measured signed contact counts, with disjoint display/motor and actuator
  IDs. The ideal voltage-clamp input protocol and model-negative selection are
  engineering modifications; see LIVE_METHOD.md.
- Native 320 x 240 Freedoom Phase 2 clips are captured through ViZDoom 1.3.0.
  `public/data/live-clips/clips.json` and `freedoom-35fps.json` record map, seed,
  frame rate, engine and hashes; `COPYING` contains BSD-3-Clause attribution.
  These are game input recordings, not prerecorded neural activity.
- MuJoCo/NeuroMechFly body assets and their original license/hash manifest are
  unchanged. The live route uses .25-ms physics steps and an engineered adapter.
- `public/data/live-validation/` contains measured comparison images and compact
  engineering evidence. The live simulation does not load these as playback.
