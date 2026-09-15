"""Integrate every real skeleton edge into a whole-neuron image operator.

Each 3D branch is split at actual projected pixel boundaries, retaining its
complete cable length. This is a transparent, linear, cable-length observation
model, not a biological optical model. No frame enters basis construction.
"""
import argparse
import json
import time

import numpy as np
from scipy.sparse import coo_matrix, save_npz

from prepare_full_brain import OUT, SOURCE
from download_full_brain import digest


def crossings(start, delta, extent):
    end = start + delta
    lo = np.floor(np.minimum(start, end)).astype(np.int64) + 1
    hi = np.ceil(np.maximum(start, end)).astype(np.int64) - 1
    lo = np.maximum(lo, 1); hi = np.minimum(hi, extent - 1)
    count = np.maximum(0, hi - lo + 1)
    edge = np.repeat(np.arange(len(start)), count)
    offset = np.repeat(np.r_[0, np.cumsum(count[:-1])], count)
    boundary = lo[edge] + np.arange(len(edge)) - offset
    fraction = (boundary - start[edge]) / delta[edge]
    return edge, fraction


def main(width):
    meta = json.loads((OUT / "brain.json").read_text())
    assert meta["complete"] and meta["neuronCount"] == 139255
    height = width * 3 // 4
    lower, upper = np.array(meta["bounds"])
    center = (lower[:2] + upper[:2]) / 2
    span = max(upper[0] - lower[0], (upper[1] - lower[1]) * width / height) * 1.02
    footprint = np.array([span, span * height / width])
    all_rows, all_columns, all_weights = [], [], []
    source_edges = 0; source_vertices = 0; integrated_length = 0.
    started = time.monotonic()
    for chunk in meta["chunks"]:
        raw = np.memmap(OUT / chunk["file"], mode="r", dtype=np.uint8)
        version, n, e, alone = np.frombuffer(raw, "<u4", count=4)
        assert version == 785
        points = np.frombuffer(raw, "<f4", count=int(n)*5, offset=16).reshape(-1, 5)
        edges = np.frombuffer(raw, "<u4", count=int(e)*2, offset=16+int(n)*20).reshape(-1, 2)
        assert np.array_equal(points[edges[:, 0], 3], points[edges[:, 1], 3])
        a = points[edges[:, 0], :3].astype(np.float64)
        b = points[edges[:, 1], :3].astype(np.float64)
        length = np.linalg.norm(b - a, axis=1)
        xy0 = ((a[:, :2] - center) / footprint + .5) * [width, height]
        xy1 = ((b[:, :2] - center) / footprint + .5) * [width, height]
        assert ((xy0 >= 0) & (xy0 < [width, height])).all()
        assert ((xy1 >= 0) & (xy1 < [width, height])).all()
        delta = xy1 - xy0
        edge_index = np.arange(int(e))
        crossing_x, time_x = crossings(xy0[:, 0], delta[:, 0], width)
        crossing_y, time_y = crossings(xy0[:, 1], delta[:, 1], height)
        event_edge = np.concatenate((edge_index, edge_index, crossing_x, crossing_y))
        event_time = np.concatenate((np.zeros(int(e)), np.ones(int(e)), time_x, time_y))
        order = np.lexsort((event_time, event_edge))
        event_edge = event_edge[order]; event_time = event_time[order]
        valid = (event_edge[1:] == event_edge[:-1]) & (event_time[1:] > event_time[:-1])
        selected = event_edge[:-1][valid]
        fractions = np.diff(event_time)[valid]
        midpoint = (event_time[1:][valid] + event_time[:-1][valid]) / 2
        pixel = np.floor(xy0[selected] + delta[selected] * midpoint[:, None]).astype(int)
        row = (height - 1 - pixel[:, 1]) * width + pixel[:, 0]
        col = points[edges[selected, 0], 3].astype(np.int32)
        weight = length[selected] * fractions
        if alone:
            nodes = np.frombuffer(raw, "<u4", count=int(alone), offset=16+int(n)*20+int(e)*8)
            single = points[nodes]
            single_xy = np.floor(((single[:, :2] - center) / footprint + .5) * [width, height]).astype(int)
            row = np.r_[row, (height - 1 - single_xy[:, 1]) * width + single_xy[:, 0]]
            col = np.r_[col, single[:, 3].astype(np.int32)]
            weight = np.r_[weight, np.full(int(alone), span / width)]
        block = coo_matrix((weight, (row, col)), shape=(width * height, meta["neuronCount"])).tocoo()
        block.sum_duplicates()
        all_rows.append(block.row.astype(np.uint32)); all_columns.append(block.col.astype(np.uint32)); all_weights.append(block.data.astype(np.float32))
        source_edges += int(e); source_vertices += int(n); integrated_length += float(length.sum())
        if len(all_rows) % 20 == 0:
            print(f"Integrated {source_edges:,} complete branch edges; {time.monotonic()-started:.1f}s", flush=True)
    basis = coo_matrix((np.concatenate(all_weights), (np.concatenate(all_rows), np.concatenate(all_columns))),
                       shape=(width * height, meta["neuronCount"])).tocsr()
    basis.eliminate_zeros()
    density = np.asarray(basis.sum(axis=1)).ravel()
    basis.data /= np.repeat(np.maximum(density, 1e-12), np.diff(basis.indptr))
    assert source_edges == meta["edgeCount"] and source_vertices == meta["vertexCount"]
    save_npz(SOURCE / f"whole-arbor-{width}.npz", basis)
    h = basis.tocoo()
    target = OUT / f"whole-arbor-{width}.bin"
    with target.open("wb") as f:
        np.array([784, width*height, meta["neuronCount"], h.nnz], "<u4").tofile(f)
        h.row.astype("<u4").tofile(f); h.col.astype("<u4").tofile(f); h.data.astype("<f4").tofile(f)
    record = dict(resolution=[width, height], normal=[0, 0, 1], center=center.tolist(), footprintMicrometers=footprint.tolist(),
                  neurons=meta["neuronCount"], vertices=source_vertices, edges=source_edges,
                  integratedCableMicrometers=integrated_length, pixelSupport=float((density > 0).mean()), entries=h.nnz,
                  operatorSha256=digest(target, "sha256"), sourceAnatomySha256=digest(OUT / "brain.json", "sha256"),
                  method="Complete cable-length integration at pixel boundaries, one brightness per whole neuron, normalized by anatomical cable density; no source-image input.",
                  caveat="Linear transparent observation model. It does not claim biological optical fidelity or exact equivalence to the interactive indexed-line rasterizer.")
    (OUT / f"whole-arbor-{width}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--width", type=int, default=320)
    main(parser.parse_args().width)
