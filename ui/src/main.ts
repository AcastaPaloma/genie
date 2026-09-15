import '@fontsource/barlow/latin-400.css';
import '@fontsource/barlow/latin-500.css';
import '@fontsource/barlow/latin-600.css';
import '@fontsource/barlow-condensed/latin-600.css';
import './style.css';
import { BrainView } from './brain';
import { fetchJson, loadImage, type BrainMetadata, type Clip, type ClipManifest } from './types';

const paths: Record<string, string> = {
  play: '<path d="m8 5 11 7-11 7Z"/>',
  pause: '<path d="M8 5v14M16 5v14"/>',
  next: '<path d="m7 5 9 7-9 7ZM18 5v14"/>',
  previous: '<path d="m17 5-9 7 9 7ZM6 5v14"/>',
  rotate: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 7a7 7 0 0 1 11.6-1L20 9M4 15l2.3 3A7 7 0 0 0 18 17"/>',
  front: '<rect x="4" y="6" width="16" height="12" rx="1"/><path d="M8 3v3M16 3v3M8 18v3M16 18v3"/>',
  volume: '<path d="m12 3 9 5v8l-9 5-9-5V8Z"/><path d="m3 8 9 5 9-5M12 13v8M12 3v10"/>',
  expand: '<path d="M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  chevron: '<path d="m8 5 7 7-7 7"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
const icon = (name: string) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] ?? ''}</svg>`;
const $ = <T extends HTMLElement = HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const base = `${import.meta.env.BASE_URL}data/`;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

$('#app').innerHTML = `
  <a class="skip-link" href="#workspace">Skip to the brain viewer</a>
  <header class="app-bar">
    <a class="brand" href="${import.meta.env.BASE_URL}" aria-label="Neuroframe home"><img src="${import.meta.env.BASE_URL}favicon.svg" alt="" width="32" height="32"/><span>NEUROFRAME</span></a>
    <div class="project-name"><span>DOOM</span><span class="project-slash">/</span><span>FlyWire</span></div><a class="body-mode-link" href="${import.meta.env.BASE_URL}live.html">Live synaptic cinema</a>
    <div class="header-actions"><span class="local-status"><span class="status-dot"></span>Local experiment</span><button id="capture" aria-label="Save image" class="button subtle" data-control disabled>${icon('download')}<span>Save image</span></button></div>
  </header>
  <main id="workspace">
    <div class="workspace-heading"><div><h1>Doom on a fly brain.</h1><p>An image made of real neuronal branches. Turn it to see how.</p></div><a class="method-link" aria-label="How it works" href="#method">${icon('info')}<span>How it works</span></a></div>
    <div class="workspace-grid">
      <section class="viewer-column" aria-label="Anatomical frame viewer">
        <div id="theater" class="theater">
          <div class="viewport-heading"><div class="view-name"><span class="view-indicator"></span><span id="view-label">Front projection</span></div><div class="view-toolbar"><button id="front" class="viewport-button active" title="Return to front view (R)" data-control disabled>${icon('front')}<span>Front</span></button><button id="orbit" class="viewport-button" aria-pressed="false" title="Automatically rotate the brain" data-control disabled>${icon('rotate')}<span>Orbit</span></button><span class="toolbar-divider"></span><button id="fullscreen" class="viewport-button icon-only" aria-label="Toggle fullscreen" title="Fullscreen" data-control disabled>${icon('expand')}</button></div></div>
          <div id="brain-canvas" class="brain-canvas"></div>
          <div id="loading" class="loading-state" role="status"><div class="loading-mark">${icon('volume')}</div><strong id="loading-title">Loading the specimen</strong><span id="loading-detail">Preparing real FlyWire anatomy and DOOM frames…</span><button id="retry" class="button primary" hidden>Reload assets</button></div>
          <div class="projection-note"><span id="projection-label">Fixed reference projection</span><span id="angle-label">0°</span></div>
          <div class="reveal-wrap"><button id="reveal" class="reveal-button" data-control disabled>${icon('volume')}<span>Reveal the depth</span>${icon('arrow')}</button><span class="drag-hint">Or drag the brain to explore</span></div>
        </div>
        <div class="viewer-status"><span><span class="status-dot"></span><strong id="neuron-count">—</strong> neurons<span class="status-divider">/</span><strong id="point-count">—</strong> branch samples</span><span class="performance"><strong id="render-fps">—</strong> render FPS<span class="status-divider">/</span><strong id="content-fps">—</strong> frame updates/s</span></div>
        <div class="transport" aria-label="Playback controls">
          <button id="play" class="play-button" aria-label="Pause playback" data-control disabled>${icon('pause')}</button>
          <div class="step-buttons"><button id="previous" class="button icon-button" aria-label="Previous frame" title="Previous frame (left arrow)" data-control disabled>${icon('previous')}</button><button id="next" class="button icon-button" aria-label="Next frame" title="Next frame (right arrow)" data-control disabled>${icon('next')}</button></div>
          <div class="timeline"><div class="timeline-label"><label for="seek">Frame <strong id="frame-current">001</strong><span id="frame-total"> / 128</span></label><span><span id="time-current">00:00.0</span> <span class="muted">/ <span id="time-total">00:12.8</span></span></span></div><input id="seek" type="range" min="0" max="127" value="0" step="1" aria-label="Seek video frame" data-control disabled/></div>
          <label class="rate-label" for="rate">Playback<select id="rate" data-control disabled><option value="5">5 FPS</option><option value="10" selected>10 FPS</option><option value="15">15 FPS</option><option value="20">20 FPS</option></select></label>
        </div>
        <div id="filmstrip" class="filmstrip" aria-label="Jump to a frame"></div>
        <div class="viewer-footnote"><span>Recorded dataset frames · no live game input</span><span><kbd>Space</kbd> play / pause <kbd>R</kbd> front view</span></div>
      </section>
      <aside class="inspector" aria-label="Source and display settings">
        <section class="source-section"><div class="section-heading"><h2>The source frame</h2><span class="small-label">Dataset</span></div>
          <div class="source-monitor"><canvas id="source-frame" width="160" height="90" role="img" aria-label="DOOM dataset frame, aspect preserved with black side padding"></canvas><span id="source-badge">DOOM / TEST</span></div>
          <label for="clip-select" class="field-label">Dataset clip</label><select id="clip-select" class="clip-select" data-control disabled><option>Loading clips…</option></select>
          <div class="source-facts"><span id="clip-location">p-doom / test split</span><span>80 × 60 source</span></div>
          <div class="pattern-options" role="group" aria-label="Display input"><button data-pattern="doom" class="segmented active" aria-pressed="true" data-control disabled>DOOM</button><button data-pattern="checker" class="segmented" aria-pressed="false" data-control disabled>Checkerboard</button><button data-pattern="solid" class="segmented" aria-pressed="false" data-control disabled>Solid</button></div>
        </section>
        <section class="display-section"><div class="section-heading"><h2>The anatomical display</h2>${icon('volume')}</div>
          <div class="control-row"><label for="exposure">Brightness</label><output id="exposure-value" for="exposure">1.30×</output></div><input id="exposure" type="range" min="0.4" max="3" step="0.05" value="1.3" data-control disabled/>
          <div class="control-row"><label for="point-size">Branch point size</label><output id="point-size-value" for="point-size">2.1 px</output></div><input id="point-size" type="range" min="0.8" max="4" step="0.1" value="2.1" data-control disabled/>
          <div class="control-row"><label for="footprint">Image footprint</label><output id="footprint-value" for="footprint">100%</output></div><input id="footprint" type="range" min="0.55" max="1.2" step="0.01" value="1" data-control disabled/>
          <div class="color-control"><span>Image color</span><div class="color-options" role="group" aria-label="Image color"><button id="rgb" class="segmented active" aria-pressed="true" data-control disabled>RGB</button><button id="mono" class="segmented" aria-pressed="false" data-control disabled>Grayscale</button></div></div>
          <button id="anatomy" class="anatomy-button" aria-pressed="false" data-control disabled>${icon('volume')}<span>Show anatomy only</span><span class="switch" aria-hidden="true"></span></button>
          <p class="control-help" id="mode-description">Frame colors are sampled onto individual branch points.</p>
          <button id="reset" class="reset-button" data-control disabled>${icon('rotate')}Reset display settings</button>
        </section>
        <details id="method" class="method"><summary><span>What am I looking at?</span>${icon('chevron')}</summary><div class="method-content"><p><strong>Real anatomy. Projected color.</strong> These are sampled branches from <span id="method-count">1,500</span> actual FlyWire neurons. Each point takes its color from the frame behind it, viewed from one fixed direction.</p><p>Orbiting changes your view, not the projection. That is why the image separates in 3D.</p><p>This is <strong>projection mapping</strong>, not simulated activity or a whole-neuron brightness solver. Colors vary along a neuron. Anatomy mode uses false colors to distinguish neurons.</p><p>The 80 × 60 source is enlarged without stretching, with black side padding. The default footprint fits the densely sampled center of the brain. Empty spaces still cannot display pixels; enlarging the footprint exposes those gaps.</p><a href="https://zenodo.org/records/10877326" target="_blank" rel="noreferrer">FlyWire anatomy · Schlegel et al. ${icon('arrow')}</a><a href="https://huggingface.co/datasets/p-doom/doom-dataset" target="_blank" rel="noreferrer">DOOM dataset · p(doom) ${icon('arrow')}</a><a href="${base}brain.json" download>Download anatomy provenance ${icon('download')}</a><p class="license-note">FlyWire data: CC BY 4.0. DOOM dataset: CC0 per its metadata. 10 FPS is a chosen playback rate; the dataset’s source FPS is unspecified.</p></div></details>
      </aside>
    </div>
    <footer class="app-footer"><span>Built from the FlyWire FAFB reconstruction · materialization 783</span><span>Anatomical display experiment</span></footer>
  </main>
  <div id="toast" class="toast" role="status" hidden></div>
`;

const state = { playing: !reducedMotion, frame: 0, rate: 10, pattern: 'doom', anatomy: false, mono: false, orbit: false, ready: false };
let manifest: ClipManifest;
let clip: Clip;
let sheet: HTMLImageElement;
let brain: BrainView | null = null;
let lastFrameTime = 0;
let animationId = 0;
let loadSequence = 0;
let toastTimer = 0;
const imageCache = new Map<string, Promise<HTMLImageElement>>();
const sourceCanvas = $<HTMLCanvasElement>('#source-frame');
const sourceContext = sourceCanvas.getContext('2d', { alpha: false })!;
sourceContext.imageSmoothingEnabled = false;
const timeLabel = (frame: number) => `00:${(frame / state.rate).toFixed(1).padStart(4, '0')}`;

function notify(message: string) {
  clearTimeout(toastTimer);
  const toast = $('#toast');
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => { toast.hidden = true; }, 3800);
}

function updatePlaying() {
  $('#play').innerHTML = icon(state.playing ? 'pause' : 'play');
  $('#play').setAttribute('aria-label', state.playing ? 'Pause playback' : 'Play frames');
  $('#app').dataset.playing = String(state.playing);
  lastFrameTime = performance.now();
}

function drawSource() {
  if (!sheet || !clip) return;
  if (state.pattern === 'doom') {
    sourceContext.drawImage(sheet, (state.frame % clip.columns) * clip.width, Math.floor(state.frame / clip.columns) * clip.height, clip.width, clip.height, 0, 0, 160, 90);
  } else if (state.pattern === 'checker') {
    for (let y = 0; y < 90; y += 10) for (let x = 0; x < 160; x += 10) {
      sourceContext.fillStyle = ((x / 10 + y / 10) % 2) ? '#f1f1ed' : '#131719';
      sourceContext.fillRect(x, y, 10, 10);
    }
  } else {
    sourceContext.fillStyle = '#d9e5f5';
    sourceContext.fillRect(0, 0, 160, 90);
  }
  brain?.updateFrame();
  $('#source-badge').textContent = state.pattern === 'doom' ? 'DOOM / TEST' : state.pattern === 'checker' ? 'TEST / CHECKERBOARD' : 'TEST / SOLID';
  $('#frame-current').textContent = String(state.frame + 1).padStart(3, '0');
  $('#frame-total').textContent = ` / ${clip.frames}`;
  $('#time-current').textContent = timeLabel(state.frame);
  $('#time-total').textContent = timeLabel(clip.frames);
  const seek = $<HTMLInputElement>('#seek');
  seek.max = String(clip.frames - 1);
  seek.value = String(state.frame);
  seek.style.setProperty('--progress', `${state.frame / (clip.frames - 1) * 100}%`);
  $('#app').dataset.frame = String(state.frame);
  $('#app').dataset.pattern = state.pattern;
  document.querySelectorAll<HTMLButtonElement>('.film-frame').forEach((button) => {
    const active = Math.abs(Number(button.dataset.frame) - state.frame) < clip.frames / 14;
    button.classList.toggle('active', active);
  });
}

function buildFilmstrip() {
  const filmstrip = $('#filmstrip');
  filmstrip.replaceChildren();
  for (let i = 0; i < 7; i++) {
    const frame = Math.round(i * (clip.frames - 1) / 6);
    const button = document.createElement('button');
    button.className = 'film-frame';
    button.dataset.frame = String(frame);
    button.setAttribute('aria-label', `Jump to frame ${frame + 1}`);
    const canvas = document.createElement('canvas');
    canvas.width = 160; canvas.height = 90;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(sheet, (frame % clip.columns) * clip.width, Math.floor(frame / clip.columns) * clip.height, clip.width, clip.height, 0, 0, 160, 90);
    const label = document.createElement('span');
    label.textContent = String(frame + 1).padStart(3, '0');
    button.append(canvas, label);
    button.onclick = () => { state.frame = frame; state.playing = false; updatePlaying(); drawSource(); };
    filmstrip.append(button);
  }
}

async function selectClip(id: string) {
  const selected = manifest.clips.find((item) => item.id === id);
  if (!selected) throw new Error('That dataset clip is unavailable.');
  const sequence = ++loadSequence;
  if (!imageCache.has(id)) imageCache.set(id, loadImage(`${base}${selected.file}`).catch((error) => { imageCache.delete(id); throw error; }));
  $('#clip-location').textContent = 'Loading clip…';
  const loaded = await imageCache.get(id)!;
  if (sequence !== loadSequence) return;
  clip = selected; sheet = loaded;
  state.frame = 0;
  lastFrameTime = performance.now();
  $<HTMLSelectElement>('#clip-select').value = id;
  $('#clip-location').textContent = `Test record ${clip.recordIndex}`;
  buildFilmstrip();
  drawSource();
}

function frontView() {
  state.orbit = false;
  $('#orbit').setAttribute('aria-pressed', 'false');
  $('#orbit').classList.remove('active');
  brain?.goTo('front');
}

function updateMode() {
  brain?.setAnatomy(state.anatomy);
  brain?.setMonochrome(state.mono);
  $('#anatomy').setAttribute('aria-pressed', String(state.anatomy));
  $('#mode-description').textContent = state.anatomy ? 'False colors identify individual neurons. No activity is being simulated.' : 'Frame colors are sampled onto individual branch points.';
  $('#projection-label').textContent = state.anatomy ? 'False-color neuron anatomy' : 'Fixed reference projection';
  $('#app').dataset.mode = state.anatomy ? 'anatomy' : state.mono ? 'grayscale' : 'rgb';
  for (const [id, active] of [['rgb', !state.mono], ['mono', state.mono]] as const) {
    $(`#${id}`).classList.toggle('active', active);
    $(`#${id}`).setAttribute('aria-pressed', String(active));
  }
}

function setRange(selector: string, callback: (value: number) => void, format: (value: number) => string) {
  const input = $<HTMLInputElement>(selector);
  const update = () => { const value = Number(input.value); $(`${selector}-value`).textContent = format(value); callback(value); };
  input.addEventListener('input', update);
  return update;
}

const updateExposure = setRange('#exposure', (v) => brain?.setExposure(v), (v) => `${v.toFixed(2)}×`);
const updateSize = setRange('#point-size', (v) => brain?.setPointSize(v), (v) => `${v.toFixed(1)} px`);
const updateFootprint = setRange('#footprint', (v) => brain?.setFootprint(v), (v) => `${Math.round(v * 100)}%`);

$('#play').onclick = () => { state.playing = !state.playing; updatePlaying(); };
for (const [id, direction] of [['previous', -1], ['next', 1]] as const) {
  $(`#${id}`).onclick = () => { state.playing = false; state.frame = Math.max(0, Math.min(clip.frames - 1, state.frame + direction)); updatePlaying(); drawSource(); };
}
$('#seek').addEventListener('input', () => { state.frame = Number($<HTMLInputElement>('#seek').value); state.playing = false; updatePlaying(); drawSource(); });
$('#rate').addEventListener('change', () => { state.rate = Number($<HTMLSelectElement>('#rate').value); lastFrameTime = performance.now(); drawSource(); });
$('#clip-select').addEventListener('change', () => { const previous = clip.id; selectClip($<HTMLSelectElement>('#clip-select').value).catch((error) => { $<HTMLSelectElement>('#clip-select').value = previous; $('#clip-location').textContent = `Test record ${clip.recordIndex}`; notify(error.message); }); });
$('#front').onclick = frontView;
$('#reveal').onclick = () => {
  if (!brain) return;
  if (brain.getInfo().angle > 10) frontView();
  else { state.orbit = false; $('#orbit').setAttribute('aria-pressed', 'false'); $('#orbit').classList.remove('active'); brain.goTo('oblique'); }
};
$('#orbit').onclick = () => { state.orbit = !state.orbit; brain?.setAutoOrbit(state.orbit); $('#orbit').setAttribute('aria-pressed', String(state.orbit)); $('#orbit').classList.toggle('active', state.orbit); };
$('#rgb').onclick = () => { state.mono = false; updateMode(); };
$('#mono').onclick = () => { state.mono = true; updateMode(); };
$('#anatomy').onclick = () => { state.anatomy = !state.anatomy; updateMode(); };
document.querySelectorAll<HTMLButtonElement>('[data-pattern]').forEach((button) => {
  button.onclick = () => {
    state.pattern = button.dataset.pattern!;
    document.querySelectorAll<HTMLButtonElement>('[data-pattern]').forEach((b) => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
    drawSource();
  };
});
$('#reset').onclick = () => {
  $<HTMLInputElement>('#exposure').value = '1.3';
  $<HTMLInputElement>('#point-size').value = '2.1';
  $<HTMLInputElement>('#footprint').value = '1';
  updateExposure(); updateSize(); updateFootprint();
  state.anatomy = false; state.mono = false; updateMode(); frontView();
  notify('Display settings reset.');
};
$('#fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $('#theater').requestFullscreen();
  } catch { notify('Fullscreen is unavailable in this browser.'); }
};
$('#capture').onclick = async () => {
  try {
    const blob = await brain!.saveImage();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `neuroframe-${clip.id}-frame-${state.frame+1}.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    notify('Brain render saved as a PNG.');
  } catch { notify('Could not save the image. Please try again.'); }
};
$('#retry').onclick = () => location.reload();
$('.method-link').onclick = () => { $<HTMLDetailsElement>('#method').open = true; };
document.addEventListener('keydown', (event) => {
  if (!state.ready || event.target instanceof HTMLElement && event.target.closest('button,input,select,a,summary,textarea')) return;
  if (event.code === 'Space') { event.preventDefault(); state.playing = !state.playing; updatePlaying(); }
  if (event.key.toLowerCase() === 'r') frontView();
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); state.playing = false; state.frame = Math.max(0, Math.min(clip.frames - 1, state.frame + (event.key === 'ArrowLeft' ? -1 : 1))); updatePlaying(); drawSource(); }
});

function showError(error: unknown) {
  $('#loading').hidden = false;
  $('#loading').classList.add('error');
  $('#loading-title').textContent = 'The display could not start';
  $('#loading-detail').textContent = error instanceof Error ? error.message : 'Check that the anatomy and frame assets are available, then reload.';
  $('#retry').hidden = false;
  state.ready = false;
  $('#app').dataset.ready = 'error';
  document.querySelectorAll<HTMLButtonElement | HTMLInputElement>('[data-control]').forEach((element) => { element.disabled = true; });
}

async function start() {
  try {
    const [meta, clips] = await Promise.all([fetchJson<BrainMetadata>(`${base}brain.json`), fetchJson<ClipManifest>(`${base}clips.json`)]);
    manifest = clips;
    if (!manifest.clips.length || !meta.pointCount) throw new Error('No anatomy or dataset frames are available. Rebuild the assets and reload.');
    const select = $<HTMLSelectElement>('#clip-select'); select.replaceChildren();
    for (const item of manifest.clips) { const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; select.append(option); }
    await selectClip(manifest.clips[0].id);
    try { brain = new BrainView($('#brain-canvas'), sourceCanvas, meta); }
    catch { throw new Error('WebGL 2 is unavailable. Enable hardware acceleration or open this page in a browser with WebGL 2 support.'); }
    brain.onContextLost = () => showError(new Error('The graphics context was lost. Reload to restore the brain display.'));
    brain.onViewChange = (angle) => {
      const isFront = angle < 3;
      $('#view-label').textContent = isFront ? 'Front projection' : 'Exploring the anatomy';
      $('#angle-label').textContent = `${Math.round(angle)}°`;
      $('#front').classList.toggle('active', isFront);
      $('#reveal span').textContent = angle > 10 ? 'Return to front' : 'Reveal the depth';
      $('#app').dataset.view = isFront ? 'front' : 'orbit';
    };
    await brain.load((text) => { $('#loading-detail').textContent = text; });
    $('#neuron-count').textContent = meta.neuronCount.toLocaleString();
    $('#method-count').textContent = meta.neuronCount.toLocaleString();
    $('#point-count').textContent = meta.pointCount.toLocaleString();
    $('#loading').hidden = true;
    document.querySelectorAll<HTMLButtonElement | HTMLInputElement>('[data-control]').forEach((element) => { element.disabled = false; });
    updateMode(); updatePlaying();
    state.ready = true;
    $('#app').dataset.ready = 'true';
    $('#app').dataset.points = String(meta.pointCount);
    let previous = performance.now(), sampleStart = previous, renders = 0, updates = 0;
    lastFrameTime = previous;
    const animate = (now: number) => {
      animationId = requestAnimationFrame(animate);
      if (document.hidden || !state.ready) { previous = now; lastFrameTime = now; sampleStart = now; renders = 0; updates = 0; return; }
      const elapsed = now - lastFrameTime;
      if (state.playing && elapsed >= 1000 / state.rate) {
        const steps = Math.floor(elapsed / (1000 / state.rate));
        state.frame = (state.frame + steps) % clip.frames;
        lastFrameTime += steps * 1000 / state.rate;
        drawSource(); if (state.pattern === 'doom') updates++;
      }
      brain!.render(now, Math.min((now-previous)/1000, 0.1)); previous = now; renders++;
      if (now-sampleStart >= 1000) {
        const seconds = (now-sampleStart)/1000;
        $('#render-fps').textContent = String(Math.round(renders/seconds));
        $('#content-fps').textContent = state.playing ? (updates/seconds).toFixed(1) : '0';
        renders = 0; updates = 0; sampleStart = now;
      }
    };
    animationId = requestAnimationFrame(animate);
  } catch (error) { console.error(error); showError(error); }
}

window.addEventListener('pagehide', (event) => { if(event.persisted)return;cancelAnimationFrame(animationId);brain?.dispose(); });
void start();
