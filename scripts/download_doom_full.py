"""Download an entire split of p-doom/doom-dataset (all shards, not just the pinned subset).
Usage: python3 scripts/download_doom_full.py [train|val|test] [output_dir]   (stdlib only)
Verifies size and, when Hugging Face reports it, the sha256 of every file. Re-runnable: skips verified files."""
import hashlib, json, shutil, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS = "p-doom/doom-dataset"
split = sys.argv[1] if len(sys.argv) > 1 else "train"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / f"{split}_full"
out.mkdir(parents=True, exist_ok=True)

def listing(split):
    files, url = [], f"https://huggingface.co/api/datasets/{DS}/tree/main/{split}?limit=1000"
    while url:
        resp = urllib.request.urlopen(urllib.request.Request(url), timeout=60)
        files += json.load(resp); url = None
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link: url = link.split("<")[1].split(">")[0]
    return [f for f in files if f["path"].endswith(".array_record")]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""): h.update(chunk)
    return h.hexdigest()

def fetch(entry):
    name = Path(entry["path"]).name; dest = out / name; want = entry.get("lfs", {}).get("oid")
    if dest.exists() and dest.stat().st_size == entry["size"] and (want is None or sha256(dest) == want):
        return f"ok      {name}"
    tmp = dest.with_suffix(".partial")
    url = f"https://huggingface.co/datasets/{DS}/resolve/main/{entry['path']}?download=true"
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh: shutil.copyfileobj(r, fh, length=4 << 20)
    if tmp.stat().st_size != entry["size"] or (want and sha256(tmp) != want):
        tmp.unlink(); return f"FAILED  {name} (size/sha mismatch)"
    tmp.replace(dest); return f"fetched {name}"

files = listing(split)
print(f"{split}: {len(files)} shards, {sum(f['size'] for f in files) / 1e9:.1f} GB -> {out}", flush=True)
with ThreadPoolExecutor(4) as pool:
    for i, msg in enumerate(pool.map(fetch, files), 1):
        print(f"[{i}/{len(files)}] {msg}", flush=True)
print("DONE", flush=True)
