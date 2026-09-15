"""Package real FlyWire skeleton samples and original DOOM dataset frames.

Run from the repository root with .venv/bin/python ui/scripts/prepare_assets.py.
Skeletons use the public source documented by fafbseg.get_skeletons. No CAVE
credentials, electron-microscopy volume, or remote training process is needed.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import ssl
import struct
import sys
import time
import urllib.request

import certifi
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "ui"
OUT = APP / "public/data"
CACHE = APP / ".cache"
BASE = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783"
ANNOTATIONS = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, context=CONTEXT, timeout=25) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def skeleton(row):
    root_id = row["root_id"]
    data = fetch(f"{BASE}/{root_id}", CACHE / "skeletons" / root_id)
    vertices, edges = struct.unpack("<II", data[:8])
    expected = 8 + vertices * 12 + edges * 8
    if vertices < 2 or edges < 1 or len(data) < expected:
        raise ValueError(f"Invalid skeleton {root_id}")
    xyz = np.frombuffer(data, dtype="<f4", count=vertices*3, offset=8).reshape(-1, 3)
    links = np.frombuffer(data, dtype="<u4", count=edges*2, offset=8+vertices*12).reshape(-1, 2)
    if not np.isfinite(xyz).all() or links.max() >= vertices:
        raise ValueError(f"Invalid coordinates/edges {root_id}")
    return row, xyz, links, hashlib.sha256(data).hexdigest()


def anatomy(count):
    raw = fetch(ANNOTATIONS, CACHE / "flywire_annotations.tsv")
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    # Sample annotated brain neurons reproducibly; exclude extrinsic peripheral
    # fibres so the browser scene is the brain rather than the whole nerve cord.
    rows = [r for r in rows if r["root_id"] and r["flow"] == "intrinsic"]
    rows.sort(key=lambda r: int(r["root_id"]))
    rng = np.random.default_rng(783)
    chosen = [rows[i] for i in rng.choice(len(rows), size=count, replace=False)]
    points, lines, neurons, failures = [], [], [], []
    vertex_count = edge_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(skeleton, row) for row in chosen]
        for index, future in enumerate(futures):
            try:
                row, xyz, links, checksum = future.result()
            except Exception as error:
                failures.append({"root_id": chosen[index]["root_id"], "error": str(error)})
                continue
            local_rng = np.random.default_rng(int(row["root_id"]) % 2**32)
            a, b = xyz[links[:, 0]], xyz[links[:, 1]]
            lengths = np.linalg.norm(b-a, axis=1).astype(np.float64)
            valid = lengths > 0
            a, b, lengths = a[valid], b[valid], lengths[valid]
            if not len(lengths):
                continue
            # Every display sample lies on a real skeletal edge. Retain identity.
            take = local_rng.choice(len(a), size=512, p=lengths / lengths.sum())
            t = local_rng.random((len(take), 1))
            sampled = a[take] * (1-t) + b[take] * t
            neuron_index = len(neurons)
            points.append(np.column_stack((sampled, np.full(len(sampled), neuron_index))))
            line_take = local_rng.choice(len(a), size=min(100, len(a)), replace=False)
            segment_xyz = np.stack((a[line_take], b[line_take]), axis=1).reshape(-1, 3)
            lines.append(np.column_stack((segment_xyz, np.full(len(segment_xyz), neuron_index))))
            vertex_count += len(xyz)
            edge_count += len(links)
            neurons.append({"rootId": row["root_id"], "cellType": row["cell_type"],
                            "superClass": row["super_class"], "side": row["side"],
                            "sha256": checksum, "vertices": len(xyz), "edges": len(links)})
            if (index+1) % 100 == 0:
                print(f"Downloaded {index+1}/{count} FlyWire skeletons", flush=True)
    if len(neurons) < count * 0.8:
        raise RuntimeError(f"Only {len(neurons)}/{count} skeletons available: {failures[:3]}")
    points = np.concatenate(points).astype("<f4")
    lines = np.concatenate(lines).astype("<f4")
    center = (np.percentile(points[:, :3], .1, axis=0) + np.percentile(points[:, :3], 99.9, axis=0)) / 2
    # A rigid change of coordinate convention and nm -> um scaling, no warping.
    for array in (points, lines):
        array[:, :3] = (array[:, :3] - center) / 1000
        array[:, 1] *= -1
    points.tofile(OUT / "brain-points.bin")
    lines.tofile(OUT / "brain-lines.bin")
    bounds = [np.percentile(points[:, :3], .1, axis=0).tolist(), np.percentile(points[:, :3], 99.9, axis=0).tolist()]
    metadata = dict(dataset="FlyWire FAFB", materialization=783, sourceUrl=BASE,
                    publicationUrl="https://doi.org/10.1038/s41586-024-07686-5",
                    archiveUrl="https://zenodo.org/records/10877326",
                    license="CC-BY-4.0",
                    annotationsUrl=ANNOTATIONS, annotationsSha256=hashlib.sha256(raw).hexdigest(),
                    selection="Seed 783; random sample of annotated intrinsic neurons",
                    requestedNeurons=count, neuronCount=len(neurons), pointCount=len(points),
                    lineVertexCount=len(lines), sourceVertexCount=vertex_count, sourceEdgeCount=edge_count,
                    coordinateUnits="micrometers", sourceCenterNm=center.tolist(), bounds=bounds,
                    transform="Subtract sourceCenterNm; divide by 1000; negate y. No nonrigid deformation.",
                    pointFormat="Little-endian float32 [x, y, z, neuronIndex]",
                    assetSha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                 for name in ("brain-points.bin", "brain-lines.bin")},
                    rendering="512 points sampled along true edges per neuron; up to 100 original edges shown as lines",
                    neurons=neurons, failedDownloads=failures,
                    attribution="FlyWire Consortium; Dorkenwald et al. (2024); Schlegel et al. (2024).")
    (OUT / "brain.json").write_text(json.dumps(metadata, separators=(",", ":")))
    print(f"Packaged {len(neurons)} neurons, {len(points):,} branch points. Bounds: {bounds}", flush=True)


def clips():
    sys.path.insert(0, str(ROOT))
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from dataloader import _VideoRecordUnpickler
    source = ROOT / "data/test/data_0000.array_record"
    reader = ArrayRecordDataSource([str(source)])
    entries = []
    selections = [(0, "Crossfire"), (12, "Stone corridors"), (35, "Close encounter"), (71, "Under fire")]
    for record_index, name in selections:
        record = _VideoRecordUnpickler(io.BytesIO(reader[record_index])).load()
        raw = record["raw_video"]
        video = np.frombuffer(raw, dtype=np.uint8).reshape(record["sequence_length"], 60, 80, 3)
        total = min(len(video), 128)
        sheet = Image.new("RGB", (160 * 16, 90 * ((total+15)//16)))
        for i in range(total):
            # Preserve the source's 4:3 aspect inside the 16:9 texture. The
            # padding is black, not fabricated imagery or stretched anatomy.
            frame = Image.new("RGB", (160, 90))
            frame.paste(Image.fromarray(video[i]).resize((120, 90), Image.Resampling.NEAREST), (20, 0))
            sheet.paste(frame, ((i%16)*160, (i//16)*90))
        filename = f"doom-{record_index}.webp"
        sheet.save(OUT / filename, lossless=True)
        sheet.crop((0, 0, 160, 90)).resize((320, 180), Image.Resampling.NEAREST).save(OUT / f"doom-{record_index}-cover.webp", lossless=True)
        entries.append(dict(id=f"doom-{record_index}", name=name, file=filename,
                            cover=f"doom-{record_index}-cover.webp", frames=total, columns=16,
                            width=160, height=90, originalWidth=80, originalHeight=60,
                            contentRect=[20, 0, 120, 90], transform="Nearest-neighbor resize to 120x90; 20px black pillarbox on each side; lossless WebP",
                            recordIndex=record_index, startFrame=0, split="test",
                            sourceFile=source.name, rawVideoSha256=hashlib.sha256(raw).hexdigest()))
    metadata = dict(dataset="p-doom/doom-dataset", sourceUrl="https://huggingface.co/datasets/p-doom/doom-dataset",
                    revision="4e5c7c6b6c60c7a91690d3b7e54cb7e62d6d1134", displayFps=10,
                    sourceFps=None, note="10 FPS is display playback; source FPS is unspecified. Original frames, no VAE inference.",
                    license="CC0-1.0 (dataset metadata)", clips=entries)
    (OUT / "clips.json").write_text(json.dumps(metadata, indent=2))
    print(f"Exported {sum(c['frames'] for c in entries)} original DOOM frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=1500)
    parser.add_argument("--only", choices=("brain", "clips"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    if args.only != "brain":
        clips()
    if args.only != "clips":
        anatomy(args.neurons)
