"""Create an HNSW collection and index a step-1 embedding run into it.

Reads whichever layout step 1 produced (sharded, or a single matrix), assigns the
rarity levels the sampling algorithm needs, uploads points shard by shard, and
records the per-level maxima step 3 reads.

Each point carries:
    level_0 .. level_{NUM_LEVELS-1}   independent capped-geometric rarity draws
    asset_id                          the step-1 asset id for this row, when present

Step 3 queries a level field by exact value - `level_i == v`, one query per level
value - so those payload indexes are built for lookups rather than ranges, and
`<HNSW_INDEX_DIR>/<collection>/max_levels.json` records the largest value each
field actually takes so the caller knows how many values there are to iterate.

Point ids are the row's position in the whole run, so they stay stable and unique
across shards.

Examples
--------
Local test with no Docker and no cloud (exact search, HNSW parameters ignored):
    python qdrant_insert.py --collection amazon-reviews_distilbert \\
        --embeddings-prefix synthetic_synthetic --qdrant embedded

Real local HNSW index in a Docker Qdrant:
    python qdrant_insert.py --collection amazon-reviews_distilbert \\
        --embeddings-prefix synthetic_synthetic --qdrant docker

Cloud cluster (QDRANT_HOST set in .env):
    python qdrant_insert.py --collection open-images_clip_vit_l14_336 \\
        --embeddings-prefix open-images_clip_vit_l14_336
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from qdrant_client import models
from tqdm import tqdm

import embedding_inputs
from config import settings
from qdrant_connection import KNOWN_COLLECTION_CONFIGS, Target, add_qdrant_args, connect

_rng = np.random.default_rng()


def draw_levels(num_points: int, num_levels: int) -> np.ndarray:
    """Draw `num_levels` independent capped geometric(0.5) rarity values per point.

    Vectorised: one call covers a whole shard, rather than one draw per
    (point, level) as the original pipeline did.
    """
    return np.minimum(
        _rng.geometric(0.5, size=(num_points, num_levels)),
        settings.MAX_RARITY_IN_VECTOR_DB,
    ).astype(np.int64)


def resolve_collection(name: str) -> tuple[str, dict]:
    """Resolve a collection key to (qdrant_name, vector_config)."""
    if name not in KNOWN_COLLECTION_CONFIGS:
        raise SystemExit(
            f"Unknown collection '{name}'. Known keys: {', '.join(KNOWN_COLLECTION_CONFIGS)}."
        )
    return settings.COLLECTION_NAME[name], KNOWN_COLLECTION_CONFIGS[name]


def create_collection(target: Target, collection_name: str, vcfg: dict, num_levels: int) -> None:
    """Create the collection with indexing deferred until the load finishes."""
    client = target.client
    if client.collection_exists(collection_name):
        print(f"[Collection] Deleting existing '{collection_name}'")
        client.delete_collection(collection_name)

    create_kwargs = dict(
        collection_name=collection_name,
        vectors_config={
            vcfg["vector_name"]: models.VectorParams(
                size=vcfg["size"], distance=models.Distance(vcfg["distance"]), on_disk=True
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=settings.HNSW_M,
            ef_construct=settings.HNSW_EF_CONSTRUCT,
            full_scan_threshold=settings.HNSW_FULL_SCAN_THRESHOLD,
            on_disk=True,
            payload_m=settings.HNSW_M,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,
            memmap_threshold=5000,
            # 0 means "do not index yet": building the graph while points are still
            # arriving would have it rebuilt repeatedly as segments grow.
            indexing_threshold=0,
        ),
        on_disk_payload=True,
    )
    # Sharding, replication and quantization only make sense on a real cluster.
    if target.is_cloud:
        create_kwargs["shard_number"] = 3
        create_kwargs["replication_factor"] = 2
        create_kwargs["write_consistency_factor"] = 1
        create_kwargs["quantization_config"] = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=False
            )
        )

    client.create_collection(**create_kwargs)
    print(f"[Collection] Created '{collection_name}' (vector='{vcfg['vector_name']}', "
          f"size={vcfg['size']}, distance={vcfg['distance']}, m={settings.HNSW_M})")

    if target.is_server:
        for j in range(num_levels):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f"level_{j}",
                field_schema=_level_index_schema(),
            )
        print(f"[Collection] Indexed level_0 .. level_{num_levels - 1} for equality lookups")
    else:
        print("[Collection] Skipping payload indexes (embedded mode ignores them)")


def _level_index_schema():
    """Payload index for the level fields: exact-value lookups only.

    Levels are only ever queried as `level_i == v` (see `make_level_requests` in
    step 3), never as a range, so the index is built without range support. That
    is smaller and quicker to build; older clients that cannot express the
    distinction fall back to a plain integer index, which also serves equality.
    """
    try:
        return models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER, lookup=True, range=False
        )
    except (AttributeError, TypeError):  # pragma: no cover - older qdrant-client
        return models.PayloadSchemaType.INTEGER


def build_payloads(levels: np.ndarray, asset_ids: list[str] | None):
    """Yield one payload dict per row, lazily so nothing is held for the whole shard."""
    level_keys = [f"level_{j}" for j in range(levels.shape[1])]
    # One conversion for the whole shard beats int() per level per point.
    for row, values in enumerate(levels.tolist()):
        payload = dict(zip(level_keys, values))
        if asset_ids is not None:
            payload["asset_id"] = asset_ids[row]
        yield payload


def upload(
    target: Target,
    collection_name: str,
    vector_name: str,
    source: embedding_inputs.EmbeddingInput,
    num_levels: int,
    batch_size: int,
    parallel: int,
) -> dict:
    """Upload every shard; return the maximum value each level takes.

    Maxima are accumulated per shard rather than from one big matrix, so memory
    stays flat regardless of how many points the run holds.
    """
    maxima = np.zeros(num_levels, dtype=np.int64)
    with tqdm(total=source.total_rows, unit="pt", desc=f"Indexing -> {collection_name}") as bar:
        for offset, vectors, asset_ids in source.iter_shards():
            rows = vectors.shape[0]
            levels = draw_levels(rows, num_levels)
            np.maximum(maxima, levels.max(axis=0), out=maxima)
            # upload_collection batches internally and takes the numpy array as it
            # is, which avoids building a PointStruct per point.
            target.client.upload_collection(
                collection_name=collection_name,
                vectors={vector_name: np.ascontiguousarray(vectors)},
                payload=build_payloads(levels, asset_ids),
                ids=range(offset, offset + rows),
                batch_size=batch_size,
                parallel=parallel,
                wait=True,
            )
            bar.update(rows)
    return {f"level_{j}": int(maxima[j]) for j in range(num_levels)}


def enable_indexing(target: Target, collection_name: str) -> None:
    """Restore the indexing threshold so Qdrant builds the HNSW graph once, now."""
    if not target.is_server:
        return
    target.client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=settings.INDEXING_THRESHOLD
        ),
    )
    print(f"[Index] indexing_threshold restored to {settings.INDEXING_THRESHOLD}; "
          f"Qdrant is building the HNSW graph in the background.")


def write_max_levels(collection_name: str, max_per_level: dict) -> str:
    """Record this collection's per-level maxima, in its own file.

    One file per collection, at `<HNSW_INDEX_DIR>/<collection>/max_levels.json`,
    so indexing one collection never rewrites another's metadata and runs can
    proceed in parallel. Written via a temporary file and replaced atomically, so
    a crash mid-write cannot leave a half-written file behind.

    Level queries are exact-value, so these maxima are the iteration bound: values
    run from 1 to the recorded maximum for each level field.
    """
    directory = os.path.join(settings.HNSW_INDEX_DIR, collection_name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "max_levels.json")

    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump({collection_name: max_per_level}, f, indent=2, sort_keys=True)
    os.replace(temporary, path)

    print(f"[Levels] Wrote max levels -> {path}")
    print(f"         {max_per_level}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", required=True,
                        help=f"Collection key. One of: {', '.join(KNOWN_COLLECTION_CONFIGS)}")
    parser.add_argument("--embeddings-prefix", required=True,
                        help="Step-1 output prefix, e.g. 'synthetic_synthetic'. Reads the "
                             "sharded layout when <prefix>_manifest.json exists, else the "
                             "single <prefix>_embeddings[_normalised].npy.")
    parser.add_argument("--embeddings-dir", default=settings.EMBEDDINGS_DIR,
                        help=f"Where step 1 wrote its files (default: {settings.EMBEDDINGS_DIR}).")
    parser.add_argument("--num-levels", type=int, default=settings.NUM_LEVELS,
                        help=f"Independent rarity levels per point (default: {settings.NUM_LEVELS}).")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Points per upsert request (default: 1000).")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent upload workers (default: 1). Raise for a remote "
                             "cluster where one connection is the bottleneck.")
    add_qdrant_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    collection_name, vcfg = resolve_collection(args.collection)

    source = embedding_inputs.resolve(args.embeddings_dir, args.embeddings_prefix, vcfg["normalised"])
    if source.dim != vcfg["size"]:
        raise SystemExit(
            f"Embedding dim {source.dim} != collection vector size {vcfg['size']} "
            f"for '{args.collection}'."
        )
    print(f"[Input] {args.embeddings_prefix}: {source.describe()}")

    target = connect(args.qdrant, args.embedded_path)
    if args.parallel > 1 and not target.is_server:
        print("[Upload] Embedded mode is single-process; ignoring --parallel.")
        args.parallel = 1

    create_collection(target, collection_name, vcfg, args.num_levels)
    max_per_level = upload(
        target, collection_name, vcfg["vector_name"], source,
        args.num_levels, args.batch_size, args.parallel,
    )
    enable_indexing(target, collection_name)
    write_max_levels(collection_name, max_per_level)

    indexed = target.client.count(collection_name=collection_name, exact=True).count
    if indexed != source.total_rows:
        raise SystemExit(f"Indexed {indexed} points but the input had {source.total_rows}.")
    print(f"[Done] {indexed} points in '{collection_name}' via {target.mode} ({target.location}).")


if __name__ == "__main__":
    main()
