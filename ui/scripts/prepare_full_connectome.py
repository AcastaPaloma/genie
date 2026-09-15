"""Reconcile the canonical 139,255-cell release with the published LIF graph.

All official neuron IDs and all released proofread connections are retained.
Existing published signs are preserved. Any source missing a published sign is
reported explicitly; no unreported guessed connection or neuron is introduced.
"""
import gzip
import hashlib
import json
import struct

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as parquet
from scipy.sparse import coo_matrix

from prepare_assets import APP, CACHE
from prepare_full_brain import SOURCE, OUT, identity, verify_source
from download_full_brain import digest


def main():
    path = verify_source("proofread_connections_783.feather")
    ids, mapping, annotations = identity()
    n = len(ids)
    roots = np.asarray(ids, dtype=np.uint64)
    order = np.argsort(roots); sorted_roots = roots[order]
    raw = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    pre_root = raw["pre_pt_root_id"].to_numpy().astype(np.uint64)
    post_root = raw["post_pt_root_id"].to_numpy().astype(np.uint64)
    pre_position = np.searchsorted(sorted_roots, pre_root)
    post_position = np.searchsorted(sorted_roots, post_root)
    assert (pre_position < n).all() and (post_position < n).all()
    assert np.array_equal(sorted_roots[pre_position], pre_root)
    assert np.array_equal(sorted_roots[post_position], post_root)
    pre = order[pre_position]; post = order[post_position]
    counts = raw["syn_count"].to_numpy().astype(np.int64)
    assert (counts > 0).all()
    matrix = coo_matrix((counts, (pre, post)), shape=(n, n), dtype=np.int64).tocsr()
    matrix.sum_duplicates(); matrix.sort_indices()
    old = parquet.read_table(CACHE / "research/Drosophila_brain_model-Connectivity_783.parquet",
                            columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory x Connectivity"])
    old_pre = old["Presynaptic_Index"].to_numpy()
    old_post = old["Postsynaptic_Index"].to_numpy()
    old_count = old["Connectivity"].to_numpy()
    old_weight = old["Excitatory x Connectivity"].to_numpy()
    signs = np.zeros(n, dtype=np.int8)
    signs[old_pre] = np.sign(old_weight).astype(np.int8)
    assert np.array_equal(signs[old_pre], np.sign(old_weight))
    published_counts = coo_matrix((old_count, (old_pre, old_post)), shape=(n, n)).tocsr()
    delta = matrix - published_counts
    changed = delta.tocoo()
    audit = dict(officialNeurons=n, publishedModelNeurons=138639,
                 addedNeurons=n-138639, sourceRows=len(counts), edgeCount=matrix.nnz,
                 anatomicalSynapseCount=int(matrix.sum()), selfConnections=int(np.count_nonzero(matrix.diagonal())),
                 publishedContactCount=int(published_counts.sum()), changedPairs=delta.nnz,
                 addedContacts=int(delta.data[delta.data > 0].sum()), removedContacts=int(-delta.data[delta.data < 0].sum()),
                 changedPairExamples=[dict(pre=ids[int(a)], post=ids[int(b)], difference=int(c))
                                      for a, b, c in zip(changed.row[:20], changed.col[:20], changed.data[:20])])
    missing_sign = np.flatnonzero((signs == 0) & (np.diff(matrix.indptr) > 0))
    audit["neuronsWithoutPublishedOutgoingSign"] = [str(ids[int(i)]) for i in missing_sign]
    audit["addedNeuronIncomingContacts"] = int(matrix[:, 138639:].sum())
    audit["addedNeuronOutgoingContacts"] = int(matrix[138639:, :].sum())
    (SOURCE / "connectivity-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if len(missing_sign):
        raise RuntimeError("The complete release contains presynaptic neurons without a published LIF sign. Audit and model them explicitly before packaging.")
    assert matrix.data.max() < 32768
    repeated_pre = np.repeat(np.arange(n), np.diff(matrix.indptr))
    signed = (matrix.data * signs[repeated_pre]).astype("<i2")
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "connectome.bin.gz"
    with output.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0, compresslevel=6) as packed:
            packed.write(struct.pack("<III", 783, n, matrix.nnz))
            packed.write(matrix.indptr.astype("<u4").tobytes())
            packed.write(matrix.indices.astype("<u4").tobytes())
            packed.write(signed.tobytes())
    with gzip.open(output, 'rb') as stream:
        decoded_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    previous = json.loads((APP / "public/data/connectome.json").read_text())
    meta = {**previous, "neuronCount": n, "edgeCount": matrix.nnz,
            "anatomicalSynapseCount": int(matrix.sum()), "sha256": digest(output, "sha256"), "uncompressedSha256": decoded_hash,
            "source": "https://zenodo.org/records/10676866", "wholeRelease": True,
            "sourceConnectivitySha256": digest(path, "sha256"),
            "neuronIds": [str(root) for root in ids], "reconciliation": audit,
            "renderMap": list(range(n)), "missingRenderedNeurons": 0,
            "inputEncoding": "Online experimental stimulation; not a reconstructed retina or validated visual response."}
    sensory = [i for i, root in enumerate(ids) if annotations[root]["flow"] == "afferent" and annotations[root]["cell_class"] == "visual"]
    meta["visualSensoryIds"] = sensory
    (OUT / "connectome.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Packaged {n:,} neurons, {matrix.nnz:,} connections; {len(sensory):,} anatomical visual afferents", flush=True)


if __name__ == "__main__":
    main()
