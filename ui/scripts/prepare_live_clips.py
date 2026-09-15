"""Capture native-resolution Freedoom clips for the live decoder and holdout tests.

Only game frames are saved; neural output is always recomputed in the browser.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import vizdoom as vzd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/data/live-clips'
OUT.mkdir(parents=True, exist_ok=True)
wad = Path(vzd.__file__).parent / 'freedoom2.wad'
records = []
for map_id, seed, name, holdout in [('map01', 321, 'Industrial corridors', False), ('map02', 117, 'Loading bay', True), ('map03', 991, 'Open chambers', True)]:
    game = vzd.DoomGame()
    game.set_doom_game_path(str(wad))
    game.set_doom_map(map_id)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_render_hud(True)
    game.set_seed(seed)
    game.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.init()
    destination = OUT / f'{map_id}.mp4'
    encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '320x240', '-framerate', '10', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    for frame in range(96):
        if game.is_episode_finished():
            game.new_episode()
        pixels = game.get_state().screen_buffer
        encoder.stdin.write(pixels.tobytes())
        if frame in [0, 8, 24, 48, 72]:
            Image.fromarray(pixels).save(OUT / f'{map_id}-{frame}.png')
        game.make_action([int(frame % 48 < 24), int(frame % 48 >= 24), int(frame % 4 == 0)], 3 + frame % 2)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')
    game.close()
    record = dict(id=map_id, name=name, file=f'live-clips/{map_id}.mp4', width=320, height=240, fps=10, frames=96, seed=seed, holdout=holdout, sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    records.append(record)
    print(record, flush=True)
(OUT / 'clips.json').write_text(json.dumps(dict(game='Freedoom Phase 2', engine=f'ViZDoom {vzd.__version__}', wadSha256=hashlib.sha256(wad.read_bytes()).hexdigest(), license='BSD-3-Clause; credits and notice in COPYING', source='https://freedoom.github.io/', clips=records), indent=2) + '\n')
