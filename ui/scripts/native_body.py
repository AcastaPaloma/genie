"""Local native MuJoCo service for the unchanged NeuroMechFly body.

The browser sends the exact per-timestep controls produced by its existing gait
controller. This process executes every physics step at the same timestep.
It receives no video, neuron geometry, or filesystem paths from clients.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import threading
import time
import uuid

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "public/body/assets"
MODEL = ASSETS / "model/fly.xml"
BASE = mujoco.MjModel.from_xml_path(str(MODEL))
BASE.opt.timestep = .00025
INFO = dict(backend="native MuJoCo", version=mujoco.__version__, timestep=.00025,
            modelSha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            metadataSha256=hashlib.sha256((ASSETS / "model_meta.json").read_bytes()).hexdigest(),
            nq=BASE.nq, ngeom=BASE.ngeom, nu=BASE.nu)
SESSIONS = {}
LOCK = threading.Lock()
ALLOWED = {f"http://{host}:{port}" for host in ("127.0.0.1", "localhost") for port in (4173, 5173)}


class Session:
    def __init__(self):
        self.data = mujoco.MjData(BASE)
        self.lock = threading.Lock()
        self.used = time.monotonic()
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(BASE, self.data, 0)
        mujoco.mj_forward(BASE, self.data)

    def state(self, elapsed):
        return b"".join(np.asarray(a, dtype="<f8").tobytes() for a in
                        ([self.data.time, elapsed, BASE.nq, BASE.ngeom], self.data.qpos,
                         self.data.geom_xpos, self.data.geom_xmat))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, value, status=200):
        binary = isinstance(value, bytes)
        data = value if binary else json.dumps(value).encode()
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in ALLOWED:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        if self.headers.get("Origin") not in ALLOWED:
            self.respond({"error": "Unknown local application origin"}, 403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.respond(INFO if self.path == "/info" else {"error": "Not found"}, 200 if self.path == "/info" else 404)

    def do_POST(self):
        try:
            if self.headers.get("Origin") not in ALLOWED:
                self.respond({"error": "Unknown local application origin"}, 403)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= BASE.nu * 400 * 8:
                raise ValueError("Invalid control packet size")
            raw = self.rfile.read(size)
            if self.path == "/session":
                config = json.loads(raw)
                if config.get("timestep") != BASE.opt.timestep:
                    raise ValueError("Native physics must retain the validated timestep")
                with LOCK:
                    expired = [key for key, session in SESSIONS.items() if time.monotonic()-session.used > 1800]
                    for key in expired:
                        del SESSIONS[key]
                    if len(SESSIONS) >= 8:
                        raise ValueError("Too many active body sessions")
                    key = uuid.uuid4().hex
                    SESSIONS[key] = Session()
                self.respond(dict(session=key, **INFO))
                return
            _, group, key, action = self.path.split("/")
            if group != "session" or key not in SESSIONS:
                raise ValueError("Unknown body session")
            session = SESSIONS[key]
            with session.lock:
                session.used = time.monotonic()
                start = time.perf_counter()
                if action == "reset":
                    session.reset()
                elif action == "step":
                    controls = np.frombuffer(raw, dtype="<f8")
                    if controls.size != BASE.nu * 400 or not np.isfinite(controls).all():
                        raise ValueError("Expected all 400 physics controls for exactly 100 ms")
                    before = session.data.time
                    for control in controls.reshape(400, BASE.nu):
                        session.data.ctrl[:] = control
                        mujoco.mj_step(BASE, session.data)
                    if abs(session.data.time-before-.1) > 1e-6 or not np.isfinite(session.data.qpos).all():
                        raise ValueError("The native body did not advance correctly")
                else:
                    raise ValueError("Unknown body operation")
                self.respond(session.state((time.perf_counter()-start)*1000))
        except (ValueError, KeyError, TypeError) as error:
            self.respond({"error": str(error)}, 400)


if __name__ == "__main__":
    print(json.dumps(dict(address="http://127.0.0.1:8769", **INFO)), flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8769), Handler).serve_forever()
