"""Previously unused game maps for the frozen full-brain event controller.

This generates only input game video. It contains no neural output. The two
scene cuts are part of one decoded input, so neural state must persist across them.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import vizdoom as vzd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='unseen-maps-640')
parser.add_argument('--purpose', choices=['evaluation','tuning'], default='evaluation')
parser.add_argument('--maps', nargs=3, default=['map07', 'map11', 'map17'])
parser.add_argument('--seed', type=int, default=91573)
parser.add_argument('--support-gain', type=float, default=0)
parser.add_argument('--rate-scale', type=float, default=60)
parser.add_argument('--visual-inputs', action='store_true')
parser.add_argument('--gate-bias', type=float)
parser.add_argument('--shunt', type=float, default=16384)
parser.add_argument('--seed-count', type=int)
parser.add_argument('--control-width', type=int, choices=[320, 640], default=320)
args = parser.parse_args()
if not args.name.replace('-', '').isalnum():
    raise ValueError('Use a simple artifact name')
out = root / "public/data/live-evaluation"
out.mkdir(exist_ok=True)
destination = out / f"{args.name}.mp4"
if destination.exists() or destination.with_suffix('.json').exists():
    raise ValueError('Preserve frozen evidence: choose a new artifact name')
wad = Path(vzd.__file__).parent / "freedoom2.wad"
controller = dict(controller="gate", gateMode=4, shunt=args.shunt, rateScale=args.rate_scale,
                  traceMs=40, traceTrigger=1, feedbackTicks=1,
                  simulationBatchTicks=3, imageFeedback=4)
if args.support_gain:
    controller['supportGain'] = args.support_gain
if args.visual_inputs:
    controller['useVisualInputs'] = 1
if args.gate_bias is not None:
    if not 0 <= args.gate_bias <= 6.99:
        raise ValueError('Gate preparation must stay below threshold')
    controller['gateBias'] = args.gate_bias
if args.seed_count is not None:
    if not 0 <= args.seed_count <= 139255:
        raise ValueError('Invalid seed population size')
    controller['seedCount'] = args.seed_count
sources = {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
           for name in ["gpu-synaptic-gate-controller.ts", "gpu-circuit.ts", "gpu-solver.ts", "gpu-rate-support.ts", "full-neural.worker.ts", "live-source.ts"]}
snapshot = out / f'{args.name}-controller'
snapshot.mkdir(exist_ok=False)
for name in sources:
    (snapshot / name).write_bytes((root / 'src' / name).read_bytes())
encoder = subprocess.Popen([
    imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
    "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x480",
    "-framerate", "35", "-i", "-", "-an", "-c:v", "libx264", "-crf", "16",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)
], stdin=subprocess.PIPE)
segments = []
for index, map_id in enumerate(args.maps):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(wad))
    game.set_doom_map(map_id)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_render_hud(True)
    game.set_seed(args.seed + index)
    game.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.init()
    for frame in range(140):
        if game.is_episode_finished():
            game.new_episode()
        encoder.stdin.write(game.get_state().screen_buffer.tobytes())
        game.make_action([int(frame % 100 < 65), int(frame % 100 >= 65), int(frame % 28 < 4)], 1)
    game.close()
    segments.append(dict(map=map_id, startSeconds=index * 4, frames=140, seed=args.seed + index))
encoder.stdin.close()
if encoder.wait():
    raise RuntimeError("Video encoding failed")
record = dict(file=destination.name, purpose=args.purpose, game="Freedoom Phase 2", engine=vzd.__version__,
              nativeResolution=[640, 480], nativeFps=35, frames=420,
              segments=segments, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
              frozenController=controller, controlWidth=args.control_width, controllerSourceSha256=sources, controllerSnapshotDirectory=snapshot.name,
              interpretation="Evaluation input generated after controller parameters were frozen. If used for later tuning, it ceases to be held-out evidence." if args.purpose=="evaluation" else "Controller/resolution tuning input. Never held-out validation evidence.")
destination.with_suffix('.json').write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
