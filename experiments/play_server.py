"""Interactive world model: hold the models on the GPU and generate one frame per request, WASD-controlled.

    python3 experiments/play_server.py [--port 8008] [--steps 3] [--keep 4] [--dtype auto] [--keys W=1,A=6,...]

Then open http://localhost:8008 (from a laptop: ssh -L 8008:localhost:8008 piano first).
Keys: W/A/S/D (arrows too) turn the camera through the latent actions; 0-7 send raw codes; hold to keep
moving; R resets from a random val clip; Q toggles the default vs 25 MaskGIT passes; [ ] change passes by one.
The prompt is the first 4 frames of a validation clip.
HTTP: GET /step?code=K&steps=S -> image/jpeg of the new frame with headers X-Frame, X-Ms (server time), X-Context
(frames in the dynamics window), X-Rebuilt (1 when the cache restarted), X-Pan (px), X-Code; GET /reset[?clip=N]
-> {clip, prompt}; GET /frame -> current jpeg; GET /keys (alias /config) -> {keys, steps, K}.

Speed: experiments/fast_infer.py runs the trained modules with a key/value cache for the context, a one-frame
decode, fused projections and MaskGIT sampling in `--steps` passes (3 by default: with this dynamics checkpoint
2, 3, 6 and 8 passes never collapsed on val clips while 4 did 17% of the time). Positions in the dynamics window
are absolute, so when the 16-frame window fills the cache restarts from the last `--keep` frames.

Latent actions are a camera-pan quantizer (see the LAM checkpoint's per-code pans), so W/S cannot mean forward/
backward: A = turn left (scene moves right), D = turn right, W and S are the two no-turn codes, and W+A / W+D /
S+A / S+D are the milder turn codes. Override with --keys.
"""
import argparse, io, json, random, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_ROOT))
import numpy as np, torch
from PIL import Image
import dynamics_model as dm
from experiments import fast_infer as fi
from experiments.vq_variants import StrideClipDataset

DEFAULT_KEYS = "W=1,S=4,A=6,D=3,WA=7,WD=5,SA=0,SD=2"   # measured on lam_spatial_a100_pan: 6/0/7 pan the scene right, 3/5/2 left

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Genie DOOM</title>
<style>body{background:#111;color:#ddd;font:14px monospace;margin:20px} img{width:640px;height:360px;image-rendering:pixelated;display:block;border:2px solid #444}
#hud{margin-top:8px;white-space:pre-wrap} kbd{background:#333;padding:2px 6px;border-radius:3px} .dim{color:#888}</style></head><body>
<img id="f" src="/frame?t=0"><div id="hud">loading...</div>
<div style="margin-top:8px"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> / arrows: A,D turn; W,S the two no-turn codes; combine for mild turns &nbsp; <kbd>0</kbd>-<kbd>7</kbd> raw codes &nbsp; hold to keep moving &nbsp; <kbd>R</kbd> reset &nbsp; <kbd>Q</kbd> %STEPS%/25 passes &nbsp; <kbd>[</kbd> <kbd>]</kbd> passes -/+</div>
<div class="dim" style="margin-top:4px">key map: %KEYMAP_TEXT%</div>
<script>
const KEYMAP=%KEYMAP%, DEFAULT_STEPS=%STEPS%, ARROWS={ArrowUp:"W",ArrowDown:"S",ArrowLeft:"A",ArrowRight:"D"};
const img=document.getElementById('f'), hud=document.getElementById('hud');
const held=new Set(); let running=false, steps=DEFAULT_STEPS, frameUrl=null, fps=null;
function combo(){for(const k of held)if(/^[0-7]$/.test(k))return{label:'code '+k,code:+k};
 const f=held.has('W')?'W':held.has('S')?'S':'', t=held.has('A')?'A':held.has('D')?'D':'', key=f+t;
 return key?{label:key,code:KEYMAP[key]}:null;}
async function step(c){const t0=performance.now();
 const r=await fetch(`/step?code=${c.code}&steps=${steps}`); if(!r.ok){hud.textContent='server error: '+await r.text();return;}
 const url=URL.createObjectURL(await r.blob()); img.src=url; if(frameUrl)URL.revokeObjectURL(frameUrl); frameUrl=url;
 const dt=performance.now()-t0; fps=fps===null?1000/dt:0.8*fps+0.2*1000/dt; const h=n=>r.headers.get(n);
 hud.textContent=`${c.label} -> code ${c.code} | frame ${h('X-Frame')} | ${fps.toFixed(1)} fps (${dt.toFixed(0)} ms round trip, ${h('X-Ms')} ms on the server) | ${steps} passes | context ${h('X-Context')} frames${h('X-Rebuilt')==='1'?' (cache restarted)':''} | pan ${h('X-Pan')} px`;}
async function run(){if(running)return;running=true;try{let c;while((c=combo())!==null)await step(c);}finally{running=false;}}
async function reset(){held.clear();const r=await fetch('/reset');const j=await r.json();img.src='/frame?t='+Date.now();fps=null;
 hud.textContent=`reset: val clip ${j.clip}, ${j.prompt} prompt frames. hold W/A/S/D.`;}
document.addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.length===1?e.key.toUpperCase():(ARROWS[e.key]||'');
 if('WASD'.includes(k)&&k||/^[0-7]$/.test(k)){held.add(k);run();e.preventDefault();}
 else if(k==='R')reset();else if(k==='Q'){steps=steps===25?DEFAULT_STEPS:25;hud.textContent='MaskGIT passes: '+steps;}
 else if(k==='['||k===']'){steps=Math.max(1,Math.min(50,steps+(k===']'?1:-1)));hud.textContent='MaskGIT passes: '+steps;}});
document.addEventListener('keyup',e=>{const k=e.key.length===1?e.key.toUpperCase():(ARROWS[e.key]||'');held.delete(k);});
window.addEventListener('blur',()=>held.clear());
reset();
</script></body></html>"""


def parse_keys(text, k):
    keys = {}
    for part in text.split(","):
        name, code = part.split("="); code = int(code)
        if not 0 <= code < k: raise ValueError(f"key {name}: code {code} outside 0..{k - 1}")
        keys[name.strip().upper()] = code
    missing = {"W", "S", "A", "D", "WA", "WD", "SA", "SD"} - set(keys)
    if missing: raise ValueError(f"--keys must map {sorted(missing)}")
    return keys


def resolve_dtype(name, device, *, decoder):
    if name != "auto": return getattr(torch, name)
    if device.type == "cuda": return torch.bfloat16
    if device.type == "mps": return torch.float16     # ~20% faster than fp32 here; same collapse rate and PSNR on val clips
    return torch.float32


class World:
    """Holds the models and one player's cached generation state."""

    def __init__(self, args):
        self.device = dm._resolve_device(args.device)
        self.dtype, self.dec_dtype = resolve_dtype(args.dtype, self.device, decoder=False), resolve_dtype(args.decoder_dtype, self.device, decoder=True)
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.model = dm.DynamicsModel(**ck["model_config"]); self.model.load_state_dict(ck["model_state_dict"])
        self.model = self.model.to(self.device, self.dtype).eval()
        self.tok = dm.load_tokenizer(args.tokenizer, self.device).to(self.dec_dtype); self.lam = dm.load_lam(args.lam, self.device)
        self.dataset = StrideClipDataset(_ROOT / "data/val", T=16, stride=ck.get("stride", 4), start_step=16, max_windows=args.clips)
        self.window, self.prompt, self.keep = self.model.temporal_pos.shape[0], args.prompt, args.keep
        if not 1 <= self.keep < self.window or not 1 <= self.prompt < self.window: raise ValueError("--keep and --prompt must be in 1..window-1")
        self.temperature, self.confidence, self.steps = args.temperature, args.confidence, args.steps
        self.keys = parse_keys(args.keys, self.lam.K)
        self.lock = threading.Lock(); self.jpeg = b""; self.frame = 0; self.last = None
        print(f"loaded on {self.device}: dynamics epoch {ck.get('epoch')} step {ck.get('global_step')} ({self.dtype}), tokenizer ({self.dec_dtype}), "
              f"K={self.lam.K}, {len(self.dataset)} prompt clips, window {self.window}, keep {self.keep} on restart", flush=True)
        self.reset(0)
        if not args.no_check:
            errors = fi.check(self.model, self.tok, self.ids, self.actions)
            print("fast paths match the model forward: " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()), flush=True)
        for _ in range(2): self.step(self.keys["W"], self.steps)          # warm up kernels so the first key press is fast
        r = self.step(self.keys["A"], self.steps)
        print(f"warm step: {r['ms']} ms at {self.steps} passes -> ~{1000 / max(1, r['ms']):.1f} fps", flush=True)
        self.reset()

    @torch.no_grad()
    def reset(self, clip=None):
        with self.lock:
            self.clip = random.randrange(len(self.dataset)) if clip is None else clip
            video = self.dataset[self.clip][0][None, :self.prompt].to(self.device)
            _, ids, _ = self.tok.encode(video.to(self.dec_dtype))
            actions, _, _ = self.lam.encode(video)                            # transitions inside the prompt
            self.ids, self.actions = ids, actions.to(self.dtype)
            self.cache = fi.Cache(); fi.dynamics(self.model, self.ids, self.actions, None, self.cache, commit=True)
            self.pending = None; self.frame = 0; self.last = None
            self._render(video[0, -1])
            return dict(clip=self.clip, prompt=self.prompt)

    @torch.no_grad()
    def step(self, code, steps):
        with self.lock:
            t0 = time.time(); rebuilt = False
            action = dm.action_vectors(self.lam, torch.tensor([code], device=self.device)).to(self.dtype)
            if self.cache.frames + (self.pending is not None) + 1 > self.window:   # absolute temporal positions: restart the cache
                self.cache, self.pending = fi.Cache(), None                          # (self.ids already holds the pending frame)
                fi.dynamics(self.model, self.ids[:, -self.keep:], self.actions[:, -(self.keep - 1):] if self.keep > 1 else self.actions[:, :0], None, self.cache, commit=True)
                rebuilt = True
            new = fi.sample(self.model, self.cache, action, steps=steps, temperature=self.temperature, confidence_threshold=self.confidence, pending=self.pending)
            self.pending = (new, action)                                            # recorded during the next step's first pass
            self.ids = torch.cat([self.ids, new[:, None]], 1)[:, -self.window:]
            self.actions = torch.cat([self.actions, action[:, None]], 1)[:, -(self.window - 1):]
            rgb = self._render(fi.decode(self.tok, new[:, None])[0, 0])            # .cpu() inside waits for the GPU
            self.frame += 1
            return dict(frame=self.frame, ms=int((time.time() - t0) * 1000), context=self.cache.frames + 1, rebuilt=rebuilt, pan=self._pan(rgb), code=code)

    def _render(self, frame):
        rgb = frame[:90].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=92); self.jpeg = buf.getvalue()
        return rgb

    def _pan(self, rgb, max_shift=14):
        """Horizontal shift of the scene since the last frame, in pixels (+ = content moved right); CPU, no GPU syncs."""
        g = rgb[8:72].astype(np.float32).mean(-1); previous, self.last = self.last, g
        if previous is None: return 0
        errors = [(((previous[:, s:] - g[:, :160 - s]) if s >= 0 else (previous[:, :160 + s] - g[:, -s:])) ** 2).mean() for s in range(-max_shift, max_shift + 1)]
        return int(np.argmin(errors)) - max_shift


def make_handler(world):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keep-alive: no TCP handshake per frame
        def log_message(self, *a): pass
        def _send(self, body, ctype, headers=()):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in headers: self.send_header(k, str(v))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path == "/":
                page = (PAGE.replace("%KEYMAP%", json.dumps(world.keys)).replace("%STEPS%", str(world.steps))
                        .replace("%KEYMAP_TEXT%", "  ".join(f"{k}={v}" for k, v in world.keys.items())))
                return self._send(page.encode(), "text/html")
            if u.path == "/frame": return self._send(world.jpeg, "image/jpeg")
            if u.path in ("/keys", "/config"):
                return self._send(json.dumps(dict(keys=world.keys, steps=world.steps, K=world.lam.K, codes=world.lam.K)).encode(), "application/json")
            if u.path == "/reset": return self._send(json.dumps(world.reset(int(q["clip"][0]) if "clip" in q else None)).encode(), "application/json")
            if u.path == "/step":
                code = max(0, min(world.lam.K - 1, int(q.get("code", ["0"])[0]))); steps = max(1, min(50, int(q.get("steps", [str(world.steps)])[0])))
                r = world.step(code, steps)
                return self._send(world.jpeg, "image/jpeg", [("X-" + k.replace("_", "-").title(), int(v)) for k, v in r.items()])
            self.send_response(404); self.end_headers()
    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_ROOT / "data/checkpoints/dynamics_a100_epoch6_weights.pt"))
    ap.add_argument("--tokenizer", default=str(_ROOT / "data/checkpoints/vae_latest_best_weights.pt"))
    ap.add_argument("--lam", default=str(_ROOT / "data/checkpoints/lam_spatial_a100_pan.pt"))
    ap.add_argument("--port", type=int, default=8008); ap.add_argument("--device", default=None)
    ap.add_argument("--prompt", type=int, default=4, help="prompt frames taken from a val clip")
    ap.add_argument("--clips", type=int, default=64, help="how many val clips to draw prompts from")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3, help="MaskGIT passes per frame (Q in the page toggles to 25)")
    ap.add_argument("--keep", type=int, default=4, help="frames the cache restarts from when the 16-frame window fills")
    ap.add_argument("--confidence", type=float, default=None, help="early exit: fix tokens sampled with >= this probability (e.g. 0.9)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="dynamics model dtype (auto: bf16 on CUDA, fp16 on MPS, else fp32)")
    ap.add_argument("--decoder-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="tokenizer dtype (auto: bf16 on CUDA, fp16 on MPS, else fp32)")
    ap.add_argument("--keys", default=DEFAULT_KEYS, help="latent code per key combo, e.g. W=1,S=4,A=6,D=3,WA=7,WD=5,SA=0,SD=2")
    ap.add_argument("--no-check", action="store_true", help="skip the start-up comparison of the fast paths against the model forward")
    args = ap.parse_args()
    world = World(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(world))
    print(f"serving on http://localhost:{args.port}  ({args.steps} MaskGIT passes, temperature {args.temperature}, keys {args.keys})", flush=True)
    server.serve_forever()
