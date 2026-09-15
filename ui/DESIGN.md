---
name: Neuroframe
description: A film viewing table for anatomical imagery.
colors:
  surround: "#e9ecef"
  paper: "#f8f9fa"
  ink: "#242c34"
  muted: "#616d79"
  line: "#d7dce2"
  accent: "#2457df"
  accent-dark: "#1744bd"
  stage: "#101619"
  stage-ink: "#edf2f5"
  white: "#ffffff"
  focus: "#6d92f0"
  control-hover: "#e8edf5"
  control-hover-line: "#b2bfd3"
  range-track: "#cbd3df"
  segmented-surround: "#e9edf2"
  segmented-hover: "#f2f5fc"
typography:
  display:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "34px"
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: "-0.015em"
  title:
    fontFamily: "Barlow, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Barlow, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "Barlow, sans-serif"
    fontSize: "13px"
    fontWeight: 500
  brand:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "26px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0.025em"
  reading:
    fontFamily: "Barlow, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.55
rounded:
  panel: "12px"
  button: "6px"
  toolbar: "7px"
  field: "5px"
  frame: "4px"
  segment: "3px"
  toast: "8px"
spacing:
  inset-tight: "3px"
  gap-compact: "8px"
  gap-control: "12px"
  inset-mobile: "16px"
  inset-panel: "20px"
  gap-workspace: "22px"
  inset-wide: "24px"
  inset-desktop: "28px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.button}"
    padding: "0 12px"
  button-primary-hover:
    backgroundColor: "{colors.accent-dark}"
  button-subtle:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.button}"
    padding: "0 12px"
  button-subtle-hover:
    backgroundColor: "{colors.control-hover}"
  button-play:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
    rounded: "{rounded.toolbar}"
    height: "40px"
    width: "40px"
  button-reveal:
    backgroundColor: "#f0f3f4"
    textColor: "#202d34"
    rounded: "{rounded.button}"
    padding: "0 15px"
  viewport-navigation:
    backgroundColor: "#182024"
    textColor: "#b6c4cc"
    rounded: "{rounded.toolbar}"
    padding: "3px"
  clip-select:
    backgroundColor: "{colors.paper}"
    textColor: "#293944"
    rounded: "{rounded.field}"
    padding: "0 9px"
    height: "36px"
    width: "100%"
  range:
    backgroundColor: "transparent"
    height: "20px"
    width: "100%"
  dataset-tag:
    textColor: "#667583"
    rounded: "{rounded.frame}"
    padding: "3px 6px"
  inspector:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "21px 20px 0"
  segment:
    backgroundColor: "transparent"
    textColor: "#596976"
    rounded: "{rounded.segment}"
    padding: "0 9px"
  segment-active:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.accent}"
---

# Design System: Neuroframe

## Overview

**Creative North Star: "The Film Viewing Table"**

The film viewing table is a quiet instrument around a vivid specimen. A pale grey surround holds the charcoal image field, source monitor, and compact controls. Barlow Condensed gives the identity and main heading a narrow, purposeful voice; Barlow keeps the working interface legible.

The visual hierarchy comes from scale and material contrast. The anatomy occupies the largest area, while cobalt marks transport, selection, and links. The source remains visibly separate from the anatomical display. Flat panels and restrained rules organize the workspace without competing with the image.

This is a scan of the implemented system in `src/style.css`, `src/main.ts`, `src/brain.ts`, and `index.html`, aligned with the committed direction (seed `4823c486`). Frontmatter contains extracted defaults; responsive overrides and behavior appear below. Spacing and radius names are documentation aliases for observed values, not claims that the stylesheet defines a token scale.

**Key Characteristics:**

- Pale instrument surround and charcoal image field.
- Cobalt for transport, selection, and interactive emphasis.
- Condensed identity lettering with compact Barlow controls.
- Pixel-preserving source imagery and a geometry-led main view.
- Flat panels, small control radii, and explicit keyboard focus.

## Colors

A cool neutral instrument surface surrounds a near-black image field, with one cobalt interface accent.

### Primary

- **Transport Cobalt** (`accent`): playback, selected segments, timeline progress, active frame border, status dots, links, and enabled switches.
- **Pressed Cobalt** (`accent-dark`): primary-button and link hover.
- **Focus Blue** (`focus`): the visible keyboard outline, independent of selected state.

### Neutral

- **Instrument Grey** (`surround`): page background, visible between the main working regions.
- **Cool Paper** (`paper`): header, inspector, form controls, and selected segments.
- **Graphite Ink** (`ink`): principal interface text; **Slate Annotation** (`muted`) supports secondary copy.
- **Divider Grey** (`line`): panel separators, transport rule, and framed control borders.
- **Charcoal Stage** (`stage`) and **Stage Light** (`stage-ink`): dark image container and its inherited foreground.
- **White** (`white`): high-contrast transport icons and primary-button text.
- **Control Hover** and **Control Hover Line**: the shared framed/subtle control hover pair.
- **Range Track**: the unfilled track. **Segmented Surround** and **Segmented Hover** separate grouped choices and their hover state.

Component-specific stage-toolbar and reveal-button colors remain local assignments in the component tokens and snippets. Dataset pixels, generated checkerboard/solid test input, and per-neuron false colors are content, not interface palette tokens.

**The Instrument Accent Rule.** Use cobalt for actionable or selected interface states; allow source imagery and anatomy colors to carry their own content palette.

## Typography

**Display Font:** Barlow Condensed, with sans-serif fallback.

**Body Font:** Barlow, with sans-serif fallback. The app bundles Latin weights (400, 500, 600), plus Barlow Condensed (600).

The pairing gives the identity and main title a compact instrument-label character without using a separate monospace face. Controls remain sentence case; the wordmark and source badges are uppercase.

### Hierarchy

- **Display:** the main workspace heading. Its smaller-screen override is (32px) at the tablet breakpoint.
- **Title:** inspector section headings.
- **Body:** the workspace explanation; it reduces to (13px) on tablet and mobile.
- **Label:** framed and subtle action labels. Other compact controls use observed sizes from (10px) to (12px).
- **Brand:** the uppercase wordmark; the tablet/mobile size is (23px).
- **Reading:** the expandable method explanation.
- **Readouts and annotation:** angles use (19px); status, source facts, frame numbers, and hints use (9px–12px), with tabular numerals where values change. There is no general-purpose monospace font.

**The Stable Readout Rule.** Use tabular numerals for frame, time, angle, and measured-value readouts so changing values do not shift the interface.

## Layout

### Projected image surface

The current surface is a desktop-first working table. A fixed-height header (68px) precedes a centered workspace capped at (1920px). Main horizontal insets are (28px). The heading sits above a two-column grid: a flexible viewer and an inspector (304px), separated by (22px). The grid has a viewport-derived height and a minimum (646px), so shorter desktop windows scroll naturally.

The viewer stacks the image field, status, transport, filmstrip, and footnote. Its default rows are `minmax(350px, 1fr) 34px 70px 66px 28px`. The inspector scrolls within the desktop grid. Sections are separated by fine rules and approximately (20px–24px) breathing room; control gaps use smaller (3px–12px) intervals.

Responsive behavior is explicit:

- At **1600px and above**, the inspector widens to (328px), its horizontal padding becomes (24px), and film frames become (80px) tall.
- At **1100px and below**, workspace insets reduce to (20px), the inspector becomes (278px), and the column gap becomes (16px). View and performance annotations simplify, and frame-step buttons hide.
- At **800px and below**, the header becomes (62px), page insets become (16px), and the inspector moves below the viewer. The stage is (440px) tall, the inspector uses two equal internal columns, and relevant viewer annotations and step controls return.
- At **520px and below**, the stage is (380px) tall and the inspector becomes one column. The filmstrip shows six frames rather than seven. Status wraps to two lines, project context and some button text hide, and button accessible names remain available. Touch controls increase selectively: anatomy/reset rows (44px), range hit area (28px), and pattern choices (34px).

The page supports widths down to (320px). These measurements describe the implemented viewing surface, not a mandatory composition for every future screen.

### Paired specimen surface

The neural activity and body page extends the same film viewing table (seed `4823c486`), using the existing palette, typography, and panel treatment. Its implemented source is `src/embodied.css`, `src/embodied.ts`, `src/body.ts`, and `embodied.html`.

A flexible specimen area sits beside a (304px) inspector with a (22px) gap. Brain and body share one charcoal container with equal-width fields, a thin divider, and the established panel radius. Its desktop height is `clamp(460px, 59vh, 740px)`. Both fields keep their own title, viewpoint menu, and lower caption. Statistics, simulation transport, a clock explanation, and four neural-output readouts sit below the pair. The stimulus monitor and intervention controls remain in the light inspector. The stimulus canvas fills its (16:9) monitor in both dimensions, retaining the packaged source padding and pixel edges.

This surface adds three local responsive breakpoints:

- At **1200px and below**, the paired container is (490px) tall, title rows and transport can wrap, simulation time gets its own line, and the output heading stacks with its activity summary.
- At **1000px and below**, the inspector moves below the specimen/readout area into three equal columns with (25px) gaps. The paired container is (520px) tall.
- At **640px and below**, brain and body stack in two equal rows within a (710px) container; the divider becomes horizontal. The inspector becomes one column with section rules restored, neural-output readouts become two columns, the timeline takes a full row, and the method explanation becomes one column.

The shared header, page insets, focus, and typography still follow the original stylesheet's responsive rules. The lower method area uses two columns (`1.2fr 1fr`) on larger screens, with a (60px) gap that reduces to (28px) below the inspector breakpoint. Its reading copy is (14px), line height (1.65), and capped at (75ch).

### Live local-video surface

`live.html` continues the film viewing table (FORM seed `4823c486`) with the existing Barlow pairing, pale surround, charcoal specimens, and cobalt controls. Its local layout comes from `src/live.css`, layered over the shared and embodied styles; interaction and source-preview behavior come from `src/live.ts` and `src/live-source.ts`.

The live pair gives more width to the brain: `minmax(0, 1.4fr) minmax(0, 1fr)`, within the existing flexible viewer plus (304px) inspector arrangement. The shared paired-stage heights, divider, titles, and captions remain. The main container retains its (1920px) cap and uses padding (25px 28px 40px). Its source monitor uses a (4:3) aspect ratio; the preview canvas has a (320 × 240) backing size and pixel-preserving presentation. The statistics row wraps with (10px 20px) gaps, minimum height (42px), and vertical padding (8px). The method area begins after (32px).

The live container changes to padding (20px) at (1100px) and below. It inherits the embodied inspector's move below the specimens at (1000px), then stacks brain above body and changes the inspector to one column at (640px). At that narrow breakpoint, main padding is (18px 16px), header experiment links are (12px), and the inherited paired container remains (710px) tall. Long local filenames wrap inside their status line. These are route-specific overrides, not new system tokens.

## Elevation & Depth

Persistent UI is flat: the header and inspector use light tone, the stage uses darkness, and separators provide structure. No persistent panel or control has a box shadow. The only CSS shadow belongs to transient notifications: `0 6px 24px #14243126`. The anatomy's actual geometry, rotation, and angle-dependent depth cue provide the important depth in this product.

**The Flat Surface Rule.** Separate persistent surfaces with tone, spacing, and fine rules. Reserve the existing shadow treatment for the transient toast.

## Shapes

Large containers use gently rounded corners; controls use smaller, precise corners. The frontmatter records the established panel, button, toolbar, field, frame, segment, and toast radii. Borders are generally (1px); active film frames use a (2px) cobalt border. Range thumbs and tiny status indicators are circular. The anatomy switch is a compact capsule (28px × 16px), with a circular thumb.

The stage and source monitor clip their contents. Fullscreen removes the stage radius. Inline icons use a (24-unit) view box, rounded strokes, and stroke width (1.6); most render between (13px) and (20px), with larger loading artwork.

## Components

### Buttons

Controls are compact and explicit. The shared framed base has a minimum height (36px), a thin border, and medium-weight label text. Cobalt primary styling is used for recovery; the subtle header action removes the resting fill and border. Both inherit the shared hover and focus treatment. Playback is a dedicated cobalt square. The stage's pale reveal action is larger, with minimum height (42px), semibold text, and a trailing arrow; mobile reduces it to (40px).

Shared interface color transitions take (140ms). Keyboard focus is a (3px) blue outline with (3px) offset. Disabled buttons, selects, and inputs use opacity (0.45) and a waiting cursor. Mobile icon-only presentation preserves the action's accessible name.

### Inputs / Fields

The clip selector is a native select with a paper fill, thin cool border, and a compact rectangular silhouette. Display sliders use a thin track (3px) and a small paper-filled cobalt-ringed thumb. The frame seek slider fills elapsed progress in cobalt and uses a cobalt thumb. Values sit above their range controls, with labels to the left and numeric outputs to the right. The small playback-rate selector remains visually unframed. Fields use the shared visible-focus outline; no validation-error field style exists in this implementation.

### Navigation

The projected-image header combines the wordmark, project context, local status, and save action. Within that stage, a dark inset toolbar holds Front, Orbit, and Fullscreen. Toolbar buttons use a lighter dark selected fill and white selected text; hover lightens their background. Selected state remains distinct from keyboard focus. On narrow screens, Front and Orbit hide visible labels and retain title-based accessible names; Fullscreen has an explicit label.

The embodied page adds a two-page experiment navigation with “Projected image” and “Neural activity + body.” The current page uses cobalt text and a bottom rule (3px), with `aria-current="page"`; other links use muted text. Links inherit the existing focus and color-transition treatment. At (640px) and below, the first header link hides and the remaining label becomes (11px); the wordmark and explicit footer return link still lead to the projection page.

### Chips / Segmented choices

The outlined Dataset tag is a quiet metadata badge. Grouped choices sit on a pale recessed strip with small gaps; selected options use paper fill and cobalt text. Pattern and color choices expose selection through `aria-pressed`. They are rectangular controls with modest rounding, not pill chips.

### Cards / Containers

The inspector is a single paper panel containing source, display settings, and the expandable method. It has no shadow or outer stroke. Internal rules divide topics. The source monitor is a dark, clipped (16:9) window with pixel-preserving canvas rendering and a small lower-left provenance badge. The packaged source retains its original image aspect with black side padding.

### Filmstrip and transport

The strip presents numbered frame thumbnails below the range transport. Unselected thumbnail imagery has opacity (0.66); hover or the current-frame neighborhood restores full opacity, and the active neighborhood gains a cobalt border. Each frame is an accessible jump action. Frame, time, and measured-rate text stay compact and stable; performance values are live readouts rather than decorative badges.

### Anatomical stage and motion

The charcoal stage hosts an actual point-and-branch rendering, with subdued labels above and the reveal action below. Rotating the viewing camera reveals geometry while the image projection stays fixed. The current front view is identified in the toolbar; an angle readout makes camera orientation explicit. The source monitor remains visibly distinct from this geometry-rendered image.

Camera presets use an exponential ease-out over (1100ms), reducing to (1ms) for reduced motion. Orbiting is an explicit control. The loading mark rotates over (5s) while preparing assets; an error stops that rotation and displays recovery. The switch thumb transitions over (160ms). With reduced motion, CSS animations and transitions stop and playback starts paused; manually requested playback and orbit remain available.

### Feedback and disclosure

Transient status uses a dark, centered-bottom toast with the system's sole shadow. The method disclosure uses a rule and a chevron that rotates when open. Both remain secondary to the working view. The loading/error overlay occupies the stage and explains its current state.

### Paired specimens and viewpoint menus

The embodied page gives brain and body equal visual weight inside one dark panel. Its default Synaptic image controller presents a coarse grayscale anatomy raster, formed from one measured spike-trace scalar per whole neuron and normalized by branch density. The activity renderer (`src/activity-renderer.ts`) receives anatomy and measured activity, never source pixels, target activities, or controller values. The body remains rendered in its physical arena. These are scene/content treatments rather than new interface palette tokens. Each specimen keeps a title and concise caption explaining how to read it: image-control mode says “Image from synaptic spikes” and “Whole-neuron activity · density-normalized anatomy”; the other encoders retain the white-branch spike-trace caption.

Each field has a native viewpoint select with a distinct accessible name: “Brain viewpoint” or “Body viewpoint.” The brain offers Image close-up, Full brain, Angled view, and Side view. Image close-up is the default and exposes the coarse (40 × 30) grayscale result most clearly; its image-control reference is fixed at (15°) as the viewing camera orbits. Full brain restores anatomical context. The body retains Angled, Front, Side, and Top choices. Menus use a dark fill (`#263239`), light text (`#e0eaf0`), a thin border (`#53626b`), the existing button radius, minimum height (36px), and maximum width (125px). They retain keyboard operation and the shared visible-focus outline; they enable after loading. Dragging remains available independently. Brain presets retain the existing eased camera movement; body presets set their viewpoint directly.

### Simulation transport and readouts

The embodied transport reuses the cobalt framed primary button for Start/Pause experiment and a paper framed Reset button. Its frame range follows the shared input styling. Elapsed time explicitly says “s simulated”; measured throughput is labeled “× real time,” separately from “render FPS.” The clock explanation remains adjacent to transport, and no measured runtime number is promoted into a design token.

Four output groups pair a cell-type/role label with a firing-rate value in Hz and a native meter. Labels use (11px); values use (18px), weight (500), and tabular numerals. The meters are (6px) tall with the existing grey track and cobalt fill. Each meter has its own accessible name. The aggregate spike and connection-delivery summary stays beside or below the section heading rather than competing with the primary numbers.

### Stimulation and intervention controls

The inspector keeps stimulus selection, external-controller choice, and interventions in separate sections. The external-controller select and stimulation range reuse the existing field patterns. “Synaptic image controller” is the default, with “Frame luminance → stimulation” and “Earlier 37-group controller” retained for comparison. The default and earlier controller disable the manual stimulation range and show their fixed readouts, respectively (0–4,000 Hz) and (0–160 Hz); the luminance mode retains the adjustable range. Nearby mode-specific copy distinguishes upstream stimulation from displayed-neuron activity and identifies the optical control as artificial.

Synaptic transmission and video stimulation use labeled native checkboxes, with cobalt accent and square controls (17px). A full-width paper button applies the separately described forward-command pulse. Its help copy identifies the pulse as a positive control with a simulated duration. Start/Pause, Reset, and pulse behavior have different labels: scrubbing changes the stimulus while preserving history, Reset clears the shared experiment, and pulsing starts it. These distinctions stay visible in nearby copy. Loading and unavailable states inherit the established disabled treatment; reduced-motion preference starts the experiment paused.

The method area adds the existing cobalt text-link treatment for “Source vs. actual spikes,” “Recorded spike playback,” and “Sequence results and disconnection test.” These lead to comparison, recorded output, and causal evidence without introducing a new panel or palette. Recorded playback is identified separately from the slower live simulation.

### Live source selection and paired preview

The live inspector combines the existing example-clip select with a labeled native local-video file input. The file control uses Barlow (12px), line height (1.5), muted filename text, and a paper button with the existing divider color and padding (9px 10px). A status line identifies local decoding, the chosen file, or a decoding error; source dimensions and duration remain quiet metadata. Files are decoded on the user's computer. Letterboxing preserves the source aspect before grayscale stimulation input is produced.

During playback, the smaller source preview is presented with its completed neural output and labeled “Reference for this neural frame” plus source time. Loading another source or seeking while paused updates only the preview and explicitly labels it “Source preview” with “awaiting simulation” or “paused.” The brain keeps its prior state until simulation advances. Preserve that distinction when placing the reference beside the larger neural field.

### Live anatomical display controls

“Anatomical display” reuses the native inspector select for “Fixed observation sites” and “All sampled branches.” The default shows one real branch site per displayed neuron; these sites are fixed independently of the input clip and are not somas. The alternate view applies the same neuron activity across its sampled arbor. The stage caption names the current representation, and the select temporarily disables while its geometry changes. Both views retain the labeled Brain viewpoint menu and fixed reference projection while orbiting.

The checked-by-default “Soften point speckle” option reuses the cobalt native checkbox row. It switches a fixed blur of emitted light (0.6 cell) on or off; it has no continuous strength control and uses no source-image detail. Neural geometry and measured spike traces determine the visible light. Target resolution, neuron counts, anatomical coverage, and upstream actuator counts are profile content, never visual tokens. The current live profile presents (160 × 120) grayscale targets on (18,901) display neurons within the full (138,639-neuron) simulation; the nearby explanation identifies idealized upstream voltage clamps and does not frame the result as HD imagery, natural vision, or demonstrated living-fly control.

### Live transport, measurements, and causal controls

The existing cobalt Start/Pause button, paper Reset button, source-time range, and model-output readouts form the live transport. “Source time” follows the video clock; elapsed model time explicitly says “s simulated.” The statistics row separately labels “new neural frames/s,” “ms processing latency,” “× model time,” and “target pixels.” New-frame throughput counts newly simulated outputs associated with distinct decoded frames, not display refreshes. Processing latency covers capture through neural-display submission and excludes decoding and spike-trace smoothing. Placeholder dashes remain until measurements exist. The four labeled firing-rate meters retain the incumbent appearance, with the live route's content range (0–300 Hz).

Synaptic transmission and video stimulation remain separate native checkboxes. Nearby copy explains the disconnection-and-Reset check; Reset clears both simulations and pauses, while source changes preserve neural history. “Save brain view” reuses the full-width paper action in the causal-control section. The live header uses the existing current-page underline for “Live synaptic cinema,” beside “Whole-branch baseline”; the first link hides at the inherited narrow breakpoint, while the wordmark returns to projection.

Reduced motion or `live.html?paused` starts this route paused, and the shared reduced-motion CSS suppresses interface transitions and animations. Moving the page into the background pauses playback. The body follows the neural sequence one processing stage behind; the paired panels and clock explanation retain that distinction. Existing model and evidence text links use cobalt without adding a new visual treatment.

## Do's and Don'ts

### Do:

- **Do** keep the image field visually dominant and its source preview explicitly labeled.
- **Do** use the established Barlow pairing and compact, tabular numeric readouts.
- **Do** preserve visible keyboard focus and accessible names when mobile controls hide their text.
- **Do** preserve source aspect and pixel edges when presenting dataset frames.
- **Do** respect reduced-motion preferences for initial playback, interface transitions, and camera changes.

### Don't:

- **Don't** apply UI cobalt to replace the source image or false-color anatomy palette.
- **Don't** add card shadows to the persistent viewing table or inspector.
- **Don't** stretch source imagery to fill the monitor or conceal empty regions in the anatomical display.
- **Don't** replace the implemented inline stroke icons with emoji or icon-font glyphs.
