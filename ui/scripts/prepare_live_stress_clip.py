"""A native 35-fps game input for local-file scheduling tests, no neural output."""
import hashlib,json,subprocess
from pathlib import Path
import imageio_ffmpeg
import vizdoom as vzd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.cache/test-clips';OUT.mkdir(exist_ok=True)
wad=Path(vzd.__file__).parent/'freedoom2.wad';game=vzd.DoomGame();game.set_doom_game_path(str(wad));game.set_doom_map('map01');game.set_screen_resolution(vzd.ScreenResolution.RES_320X240);game.set_screen_format(vzd.ScreenFormat.RGB24);game.set_window_visible(False);game.set_sound_enabled(False);game.set_render_hud(True);game.set_seed(7729);game.set_available_buttons([vzd.Button.MOVE_FORWARD,vzd.Button.TURN_RIGHT,vzd.Button.ATTACK]);game.init()
destination=OUT/'freedoom-35fps.mp4';encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','320x240','-framerate','35','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
for frame in range(350):
 if game.is_episode_finished():game.new_episode()
 encoder.stdin.write(game.get_state().screen_buffer.tobytes());game.make_action([int(frame%168<84),int(frame%168>=84),int(frame%14==0)],1)
encoder.stdin.close();assert encoder.wait()==0;game.close()
record=dict(file=destination.name,game='Freedoom Phase 2',map='map01',seed=7729,fps=35,frames=350,nativeResolution=[320,240],engine=vzd.__version__,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),purpose='Local-file throughput stress input. Unique native game frames, no interpolation or recorded neural output.')
(OUT/'freedoom-35fps.json').write_text(json.dumps(record,indent=2));print(record)
