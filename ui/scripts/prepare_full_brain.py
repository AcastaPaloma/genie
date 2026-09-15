"""Package every official FlyWire v783 neuron and every released skeletal edge.

No neuron selection, random samples, downsampling or synthetic gap filling.
The publisher's complete SWC archive is streamed; each neuron is kept intact in
an indexed GPU chunk. Original radii are retained alongside vertex positions.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_assets import APP, CACHE
from download_full_brain import SOURCES, digest

SOURCE = CACHE / "full-brain"
OUT = APP / "public/data/full-brain-783"


def identity():
    official = np.load(SOURCE / "proofread_root_ids_783.npy").astype(np.uint64)
    assert len(official) == 139255 and len(np.unique(official)) == 139255
    with (SOURCE / "annotations-v2.1.0.tsv").open() as f:
        annotations = {int(r["root_id"]): r for r in csv.DictReader(f, delimiter="\t")}
    assert set(annotations) == set(official.tolist())
    with (CACHE / "research/Drosophila_brain_model-Completeness_783.csv").open() as f:
        old = [int(r[""]) for r in csv.DictReader(f)]
    assert set(old) <= set(annotations)
    # Preserve old model indices, then append every missing official neuron.
    ids = old + sorted(set(annotations) - set(old))
    return ids, {root: i for i, root in enumerate(ids)}, annotations


def verify_source(name):
    _, _, size, md5 = next(s for s in SOURCES if s[1] == name)
    path = SOURCE / name
    assert path.stat().st_size == size and digest(path) == md5, f"Unverified complete source: {name}"
    return path


class VerifiedRangeReader(io.RawIOBase):
    """Wait for completed HTTP ranges while an immutable archive downloads.

    Output cannot be marked complete until the entire publisher checksum passes.
    This overlaps parsing with acquisition; it does not accept holes as data.
    """
    def __init__(self, path, state_path):
        self.raw = path.open("rb")
        self.state_path = state_path
        self.last_message = 0.

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.raw.tell()
    def seek(self, offset, whence=0): return self.raw.seek(offset, whence)

    def read(self, size=-1):
        start = self.raw.tell()
        if size < 0: size = 5355543468 - start
        if size == 0: return b""
        while True:
            try:
                state = json.loads(self.state_path.read_text())
                chunk = state["chunkSize"]
                needed = set(range(start // chunk, (start + size - 1) // chunk + 1))
                if needed <= set(state["completed"]): break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() - self.last_message > 30:
                print(f"Importer waiting for verified source bytes {start:,}–{start+size:,}", flush=True)
                self.last_message = time.monotonic()
            time.sleep(1)
        return self.raw.read(size)

    def readinto(self, destination):
        data = self.read(len(destination)); destination[:len(data)] = data
        return len(data)

    def close(self):
        self.raw.close(); super().close()


def prepare_geometry(max_vertices=1000000, stream_download=False):
    path = SOURCE / "sk_lod1_783_healed_ds2.parquet"
    if stream_download and not path.exists():
        partial = path.with_suffix(path.suffix + ".ranges.partial")
        reader = VerifiedRangeReader(partial, partial.with_suffix(partial.suffix + ".json"))
        meta = pq.ParquetFile(reader, metadata=pq.read_metadata(SOURCE / "parquet-metadata.parquet"))
    else:
        path = verify_source(path.name)
        meta = pq.ParquetFile(path)
    ids, mapping, annotations = identity()
    OUT.mkdir(parents=True, exist_ok=True)
    center = np.asarray(json.loads((APP / "public/data/brain.json").read_text())["sourceCenterNm"], dtype=np.float64)
    print(f"Archive: {meta.metadata.num_rows:,} vertices in {meta.metadata.num_row_groups} row groups", flush=True)
    columns = ["node_id", "parent_id", "radius", "x", "y", "z", "neuron"]
    assert set(columns) <= set(meta.schema.names)
    chunks, neurons, extras = [], {}, []
    vertices, links, isolated = [], [], []
    count = edge_count = isolated_count = chunk_neurons = 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    start = time.monotonic()

    def flush():
        nonlocal count, edge_count, isolated_count, chunk_neurons
        if not count:
            return
        points = np.concatenate(vertices).astype("<f4", copy=False)
        edges = np.concatenate(links).astype("<u4", copy=False) if links else np.empty((0, 2), "<u4")
        singles = np.concatenate(isolated).astype("<u4", copy=False) if isolated else np.empty(0, "<u4")
        name = f"skeleton-{len(chunks):04d}.bin"
        target = OUT / name
        with target.with_suffix(".partial").open("wb") as f:
            f.write(struct.pack("<IIII", 785, len(points), len(edges), len(singles)))
            points.tofile(f); edges.tofile(f); singles.tofile(f)
        target.with_suffix(".partial").replace(target)
        chunks.append(dict(file=name, sha256=digest(target, "sha256"), vertices=len(points), edges=len(edges),
                           isolatedVertices=len(singles), neuronCount=chunk_neurons,
                           bounds=[points[:, :3].min(axis=0).tolist(), points[:, :3].max(axis=0).tolist()]))
        vertices.clear(); links.clear(); isolated.clear()
        count = edge_count = isolated_count = chunk_neurons = 0
        if len(chunks) % 10 == 0:
            print(f"{len(neurons):,}/139,255 neurons; {len(chunks)} complete-geometry chunks; {time.monotonic() - start:.1f}s", flush=True)

    def pack(data):
        nonlocal count, edge_count, isolated_count, chunk_neurons, lo, hi
        root = int(data["neuron"][0])
        assert np.all(data["neuron"] == root)
        if root not in mapping:
            extras.append(dict(rootId=str(root), vertices=len(data["neuron"])))
            return
        if root in neurons:
            raise ValueError(f"Noncontiguous or duplicated neuron in archive: {root}")
        n = len(data["node_id"])
        if count and count + n > max_vertices:
            flush()
        xyz = np.column_stack([data[k] for k in ("x", "y", "z")]).astype(np.float64)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite anatomy: {root}")
        node_ids = data["node_id"].astype(np.int64)
        parent_ids = data["parent_id"].astype(np.int64)
        order = np.argsort(node_ids)
        sorted_ids = node_ids[order]
        assert len(np.unique(node_ids)) == n
        children = np.flatnonzero(parent_ids >= 0)
        where = np.searchsorted(sorted_ids, parent_ids[children])
        if (where >= n).any() or not np.array_equal(sorted_ids[where], parent_ids[children]):
            raise ValueError(f"Broken released parent link: {root}")
        parents = order[where]
        local_edges = np.column_stack((children, parents)).astype(np.uint32)
        used = np.zeros(n, dtype=bool); used[children] = True; used[parents] = True
        alone = np.flatnonzero(~used).astype(np.uint32)
        coords = (xyz - center) / 1000
        coords[:, 1] *= -1
        radius = np.asarray(data["radius"], dtype=np.float64) / 1000
        if not np.isfinite(radius).all() or (radius < 0).any():
            raise ValueError(f"Invalid released radius: {root}")
        packed = np.column_stack((coords, np.full(n, mapping[root]), radius)).astype("<f4")
        lo = np.minimum(lo, packed[:, :3].min(axis=0)); hi = np.maximum(hi, packed[:, :3].max(axis=0))
        vertices.append(packed); links.append(local_edges + count); isolated.append(alone + count)
        neurons[root] = dict(rootId=str(root), modelIndex=mapping[root], vertices=n, edges=len(local_edges),
                             roots=int((parent_ids < 0).sum()), isolatedVertices=len(alone),
                             cellType=annotations[root]["cell_type"], flow=annotations[root]["flow"],
                             superClass=annotations[root]["super_class"], side=annotations[root]["side"])
        count += n; edge_count += len(local_edges); isolated_count += len(alone); chunk_neurons += 1

    pending = None
    for batch in meta.iter_batches(batch_size=1000000, columns=columns):
        data = {key: batch.column(key).to_numpy(zero_copy_only=False) for key in columns}
        boundaries = np.r_[0, np.flatnonzero(np.diff(data["neuron"]) != 0) + 1, batch.num_rows]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            row = {key: value[a:b] for key, value in data.items()}
            if pending is not None:
                if pending["neuron"][0] == row["neuron"][0]:
                    row = {key: np.concatenate((pending[key], row[key])) for key in columns}
                else:
                    pack(pending)
                pending = None
            if b == batch.num_rows:
                pending = {key: value.copy() for key, value in row.items()}
            else:
                pack(row)
    if pending is not None:
        pack(pending)
    flush()
    while not path.exists():
        print("Geometry parsed; waiting for the downloader's complete-archive checksum", flush=True)
        time.sleep(10)
    verify_source(path.name)
    missing = sorted(set(ids) - set(neurons))
    record = dict(version=1, materialization=783, expectedNeurons=len(ids), neuronCount=len(neurons),
                  complete=not missing, missingRootIds=[str(x) for x in missing],
                  vertexCount=sum(n["vertices"] for n in neurons.values()),
                  edgeCount=sum(n["edges"] for n in neurons.values()),
                  isolatedVertexCount=sum(n["isolatedVertices"] for n in neurons.values()),
                  sourceVertexCount=meta.metadata.num_rows, sourceExtraNeurons=extras,
                  bounds=[lo.tolist(), hi.tolist()], sourceCenterNm=center.tolist(),
                  coordinates="source nm translated by sourceCenterNm, divided by 1000, shared Y reflection; no neuron-dependent transform",
                  rendering="Every released parent edge as an indexed line; isolated nodes as points. Radius retained in vertex data. No neuron selection or additional downsampling.",
                  binary="u32 [785, vertices, edges, isolated]; f32 [x_um,y_um,z_um,modelIndex,radius_um] per vertex; u32 [child,parent] per edge; u32 isolated vertex indices",
                  source=dict(url="https://zenodo.org/records/10877326", file=path.name, md5=digest(path), sha256=digest(path, "sha256")),
                  officialIdsSha256=digest(SOURCE / "proofread_root_ids_783.npy", "sha256"),
                  annotationsSha256=digest(SOURCE / "annotations-v2.1.0.tsv", "sha256"),
                  chunks=chunks, neurons=[neurons[root] for root in ids if root in neurons])
    assert record["vertexCount"] + sum(n["vertices"] for n in extras) == record["sourceVertexCount"]
    temporary = OUT / "brain.json.partial"
    temporary.write_text(json.dumps(record, separators=(",", ":")))
    temporary.replace(OUT / "brain.json")
    (SOURCE / "geometry-audit.json").write_text(json.dumps({k: v for k, v in record.items() if k not in ("neurons", "chunks")}, indent=2))
    if missing:
        raise RuntimeError(f"Full-brain import incomplete: {len(missing)} missing official neurons. See geometry-audit.json. No complete profile was produced.")
    print(json.dumps({k: record[k] for k in ("complete", "neuronCount", "vertexCount", "edgeCount", "isolatedVertexCount")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-vertices", type=int, default=1000000)
    parser.add_argument("--stream-download", action="store_true")
    args = parser.parse_args()
    prepare_geometry(args.chunk_vertices, args.stream_download)
