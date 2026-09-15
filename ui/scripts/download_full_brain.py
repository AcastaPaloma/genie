"""Download pinned, complete FlyWire v783 public data with checksum verification.

No neuron selection. Partial transfers resume only after a valid HTTP range
response; completed sources must match the publisher's recorded MD5 and size.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from prepare_assets import CACHE, CONTEXT

DEST = CACHE / "full-brain"
SOURCES = [
    (10676866, "proofread_root_ids_783.npy", 1114168, "e0e6c19732fd8c7a4e39a2d170105421"),
    (10877326, "sk_lod1_783_healed_ds2.parquet", 5355543468, "a4c104776f33ec539ef859064c4de3df"),
    (10676866, "proofread_connections_783.feather", 852022274, "f48f972d262323a102aed49af1396b8a"),
]
ANNOTATION_REVISION = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"


def ranged_transfer(client, url, target, size, workers):
    """Bounded parallel HTTP ranges, retaining byte-for-byte source contents."""
    chunk_size = 32 * 1024 * 1024
    partial = target.with_suffix(target.suffix + ".ranges.partial")
    state_path = partial.with_suffix(partial.suffix + ".json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"url": url, "size": size, "chunkSize": chunk_size, "completed": []}
    assert state["url"] == url and state["size"] == size and state["chunkSize"] == chunk_size
    done = set(state["completed"])
    fd = os.open(partial, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    # Reuse a prefix from the earlier ordinary transfer, without assuming that
    # a sparsely allocated file means that any missing ranges were downloaded.
    old = target.with_suffix(target.suffix + ".partial")
    if not done and old.exists():
        with old.open("rb") as f:
            for i in range(old.stat().st_size // chunk_size):
                os.pwrite(fd, f.read(chunk_size), i * chunk_size)
                done.add(i)
    start_time = time.monotonic()

    def piece(i):
        start = i * chunk_size
        end = min(size, start + chunk_size) - 1
        for attempt in range(5):
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("content-range") != expected:
                        raise RuntimeError(f"Invalid HTTP range response: {response.status_code} {response.headers.get('content-range')}")
                    cursor = start
                    for data in response.iter_bytes(1024 * 1024):
                        if cursor + len(data) > end + 1:
                            raise RuntimeError("Range exceeded the requested size")
                        written = os.pwrite(fd, data, cursor)
                        if written != len(data):
                            raise OSError("Incomplete local write")
                        cursor += written
                    if cursor != end + 1:
                        raise RuntimeError("Truncated HTTP range")
                return i
            except (httpx.HTTPError, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    total = (size + chunk_size - 1) // chunk_size
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(piece, i) for i in range(total) if i not in done]
            for future in concurrent.futures.as_completed(futures):
                done.add(future.result())
                state["completed"] = sorted(done)
                temporary_state = state_path.with_suffix('.writing')
                temporary_state.write_text(json.dumps(state))
                temporary_state.replace(state_path)
                print(f"{target.name}: {len(done)}/{total} verified HTTP ranges; {time.monotonic() - start_time:.0f}s", flush=True)
    finally:
        os.close(fd)
    return partial


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(client, record, name, size, checksum, workers=8):
    target = DEST / name
    url = f"https://zenodo.org/api/records/{record}/files/{name}/content"
    if target.exists():
        if target.stat().st_size != size or digest(target) != checksum:
            raise ValueError(f"Existing source does not match published checksum: {name}")
        print(f"Verified cached {name}", flush=True)
    elif size > 100 * 1024 * 1024 and workers > 1:
        partial = ranged_transfer(client, url, target, size, workers)
        if partial.stat().st_size != size or digest(partial) != checksum:
            raise ValueError(f"Published checksum mismatch: {name}")
        partial.replace(target)
        print(f"Downloaded and verified {name}: {size / 1e9:.3f} GB", flush=True)
    else:
        partial = target.with_suffix(target.suffix + ".partial")
        start = time.monotonic()
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get("content-range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Server did not honor the requested resume range")
                    last = time.monotonic()
                    with partial.open("ab" if offset else "wb") as f:
                        for chunk in response.iter_bytes(2 * 1024 * 1024):
                            f.write(chunk)
                            offset += len(chunk)
                            if offset > size:
                                raise ValueError(f"Source exceeds the pinned size: {name}")
                            if time.monotonic() - last >= 10:
                                print(f"{name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB; {time.monotonic() - start:.0f}s", flush=True)
                                last = time.monotonic()
                if offset != size:
                    raise RuntimeError(f"Truncated source: {offset}/{size}")
                break
            except (httpx.HTTPError, RuntimeError) as error:
                if attempt == 4:
                    raise
                print(f"Retrying {name}: {error}", flush=True)
                time.sleep(min(2 ** attempt, 8))
        if partial.stat().st_size != size or digest(partial) != checksum:
            raise ValueError(f"Published checksum mismatch: {name}")
        partial.replace(target)
        print(f"Downloaded and verified {name}: {size / 1e9:.3f} GB", flush=True)
    return dict(file=name, url=url, bytes=size, md5=checksum, sha256=digest(target, "sha256"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-synapses", action="store_true", help="Also download all published synapse locations, including unproofread partners")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--file", help="Download just this pinned file")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    sources = SOURCES.copy()
    if args.include_synapses:
        sources.append((10676866, "flywire_synapses_783.feather", 9492998242, "f8f1b97c9d4b0ea9b4c8b287f6b99091"))
    if args.file:
        sources = [source for source in sources if source[1] == args.file]
        if not sources:
            parser.error("Unknown pinned source")
    manifest = dict(materialization=783, annotationRevision=ANNOTATION_REVISION, files=[])
    with httpx.Client(verify=CONTEXT, timeout=httpx.Timeout(90, connect=30), follow_redirects=True) as client:
        annotation = DEST / "annotations-v2.1.0.tsv"
        annotation_url = f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_REVISION}/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
        if not annotation.exists():
            response = client.get(annotation_url)
            response.raise_for_status()
            annotation.write_bytes(response.content)
        manifest["annotations"] = dict(file=annotation.name, url=annotation_url, sha256=digest(annotation, "sha256"))
        for source in sources:
            manifest["files"].append(download(client, *source, workers=args.workers))
            (DEST / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
