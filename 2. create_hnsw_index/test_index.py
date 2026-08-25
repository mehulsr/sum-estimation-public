"""Smoke-test indexing end to end, with no Docker and no cloud cluster.

Builds a tiny embedding run on disk (numpy only - step 1 is not needed), indexes
it in embedded mode, and checks the points, payloads, ids and search results that
step 3 depends on. Add `--docker` to repeat the run against a local Docker Qdrant,
which is the only way to exercise a real HNSW index.

    python test_index.py
    python test_index.py --docker
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from qdrant_client import models

from config import settings
from qdrant_connection import KNOWN_COLLECTION_CONFIGS, connect

HERE = os.path.dirname(os.path.abspath(__file__))

# One collection per input layout, so both reader paths get covered.
CASES = (
    ("amazon-reviews_distilbert", "sharded"),
    ("open-images_clip_vit_l14_336", "single"),
)
ROWS = 60
CHUNK_ROWS = 25


def write_run(directory: str, prefix: str, rows: int, dim: int, normalised: bool, layout: str):
    """Write a fake step-1 run: sharded with a manifest, or one matrix."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((rows, dim)).astype(np.float32)
    if normalised:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    asset_ids = [f"asset-{i:04d}" for i in range(rows)]
    variant = "normalised" if normalised else "raw"
    suffix = "_embeddings_normalised.npy" if normalised else "_embeddings.npy"

    if layout == "single":
        np.save(os.path.join(directory, f"{prefix}{suffix}"), vectors)
        with open(os.path.join(directory, f"{prefix}_ids.txt"), "w") as f:
            f.write("\n".join(asset_ids) + "\n")
        return vectors, asset_ids

    chunks = []
    for index, start in enumerate(range(0, rows, CHUNK_ROWS)):
        end = min(start + CHUNK_ROWS, rows)
        stem = f"{prefix}_{index:05d}"
        np.save(os.path.join(directory, f"{stem}{suffix}"), vectors[start:end])
        with open(os.path.join(directory, f"{stem}_ids.txt"), "w") as f:
            f.write("\n".join(asset_ids[start:end]) + "\n")
        chunks.append({
            "index": index, "stem": stem, "rows": end - start, "source_rows": end,
            "files": {variant: f"{stem}{suffix}", "ids": f"{stem}_ids.txt"},
        })
    manifest = {
        "dim": dim, "chunk_rows": CHUNK_ROWS, "write_raw": not normalised,
        "write_normalised": normalised, "rows": rows, "source_rows_consumed": rows,
        "chunks": chunks,
    }
    with open(os.path.join(directory, f"{prefix}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return vectors, asset_ids


def run_insert(prefix: str, collection: str, embeddings_dir: str, mode: str, index_dir: str):
    command = [
        sys.executable, "qdrant_insert.py",
        "--collection", collection,
        "--embeddings-prefix", prefix,
        "--embeddings-dir", embeddings_dir,
        "--qdrant", mode,
        "--batch-size", "16",
    ]
    environment = dict(os.environ, HNSW_INDEX_DIR=index_dir)
    if mode == "embedded":
        # Persist so this process can reopen the same index after the subprocess exits.
        environment["QDRANT_EMBEDDED_PATH"] = os.path.join(embeddings_dir, "qdrant_store")
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise AssertionError(
            f"qdrant_insert.py exited {completed.returncode}\n"
            f"{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}"
        )
    return environment.get("QDRANT_EMBEDDED_PATH")


def check(collection_key: str, layout: str, root: str, mode: str) -> str:
    vcfg = KNOWN_COLLECTION_CONFIGS[collection_key]
    collection_name = settings.COLLECTION_NAME[collection_key]
    workdir = os.path.join(root, f"{collection_key}_{layout}_{mode}")
    prefix = f"smoke_{collection_key}"
    index_dir = os.path.join(workdir, "hnsw_index")

    vectors, asset_ids = write_run(
        workdir, prefix, ROWS, vcfg["size"], vcfg["normalised"], layout
    )
    embedded_path = run_insert(prefix, collection_key, workdir, mode, index_dir)

    target = connect(mode, embedded_path)
    client = target.client
    assert client.collection_exists(collection_name), f"{collection_name} was not created"
    count = client.count(collection_name=collection_name, exact=True).count
    assert count == ROWS, f"expected {ROWS} points, found {count}"

    # Payloads: every level field present, plus the step-1 asset id for that row.
    records, _ = client.scroll(collection_name=collection_name, limit=ROWS, with_payload=True)
    assert len(records) == ROWS, len(records)
    for record in records:
        for j in range(settings.NUM_LEVELS):
            value = record.payload.get(f"level_{j}")
            assert isinstance(value, int) and 1 <= value <= settings.MAX_RARITY_IN_VECTOR_DB, \
                f"bad level_{j}={value!r} on point {record.id}"
        assert record.payload.get("asset_id") == asset_ids[record.id], \
            f"point {record.id} carries {record.payload.get('asset_id')!r}"

    # Search: a stored vector must retrieve its own point first.
    probe = ROWS // 2
    hits = client.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(name=vcfg["vector_name"], vector=vectors[probe].tolist()),
        limit=1,
    )
    assert hits and hits[0].id == probe, f"nearest to row {probe} was {hits[0].id if hits else None}"

    # Level lookups are exact-value, one query per level, as step 3 issues them.
    # Every point must be reachable through exactly one of them, and each query
    # must return only points sitting at that level.
    by_level = {}
    for record in records:
        by_level.setdefault(record.payload["level_0"], []).append(record.id)

    matched = 0
    for level, expected_ids in sorted(by_level.items()):
        hits = client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(name=vcfg["vector_name"], vector=vectors[probe].tolist()),
            query_filter=models.Filter(must=[models.FieldCondition(
                key="level_0", match=models.MatchValue(value=level))]),
            limit=ROWS,
            with_payload=True,
        )
        assert sorted(h.id for h in hits) == sorted(expected_ids), (
            f"level_0 == {level} returned {sorted(h.id for h in hits)}, "
            f"expected {sorted(expected_ids)}"
        )
        assert all(h.payload["level_0"] == level for h in hits), \
            f"level_0 == {level} returned a point at another level"
        matched += len(hits)
    assert matched == ROWS, f"exact-value level queries covered {matched}/{ROWS} points"

    # A level nobody was assigned must come back empty, not fall back to a range.
    absent = max(by_level) + 1
    empty = client.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(name=vcfg["vector_name"], vector=vectors[probe].tolist()),
        query_filter=models.Filter(must=[models.FieldCondition(
            key="level_0", match=models.MatchValue(value=absent))]),
        limit=ROWS,
    )
    assert not empty, f"level_0 == {absent} matched {len(empty)} points but nothing was assigned it"

    # Metadata goes to this collection's own file, and nowhere else.
    max_levels_path = os.path.join(index_dir, collection_name, "max_levels.json")
    assert os.path.exists(max_levels_path), f"no max_levels.json at {max_levels_path}"
    written = [
        os.path.relpath(os.path.join(base, name), index_dir)
        for base, _, names in os.walk(index_dir) for name in names
    ]
    assert written == [os.path.join(collection_name, "max_levels.json")], (
        f"expected one file under {index_dir}, found {written}"
    )

    with open(max_levels_path) as f:
        maxima = json.load(f)
    assert list(maxima) == [collection_name], f"file holds {list(maxima)}"
    assert sorted(maxima[collection_name]) == sorted(
        f"level_{j}" for j in range(settings.NUM_LEVELS)), maxima[collection_name]
    # The recorded maximum bounds the values a caller iterates over, so no point
    # may sit above it.
    assert maxima[collection_name]["level_0"] == max(by_level), (
        f"max_levels says level_0 tops out at {maxima[collection_name]['level_0']} "
        f"but a point sits at {max(by_level)}"
    )

    client.delete_collection(collection_name)
    return f"{ROWS} points, {layout} input, {len(by_level)} distinct level_0 values"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true",
                        help="Also index against a local Docker Qdrant (real HNSW).")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    args = parser.parse_args()

    modes = ["embedded"] + (["docker"] if args.docker else [])
    if args.docker and not docker_available():
        raise SystemExit("--docker needs a running Docker daemon; start Docker and retry.")

    root = tempfile.mkdtemp(prefix="hnsw-smoke-")
    print(f"Scratch: {root}\n")
    results = []
    for mode in modes:
        for collection_key, layout in CASES:
            label = f"{mode}: {collection_key} ({layout})"
            print(f"[Test] {label} ...", flush=True)
            try:
                results.append((label, "PASS", check(collection_key, layout, root, mode)))
            except AssertionError as error:
                results.append((label, "FAIL", str(error)))

    print("\n" + "=" * 78)
    for label, status, detail in results:
        print(f"{status:4}  {label:52} {detail if status == 'PASS' else ''}")
        if status == "FAIL":
            print(detail)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"\nScratch kept at {root}")
    raise SystemExit(any(status == "FAIL" for _, status, _ in results))


if __name__ == "__main__":
    main()
