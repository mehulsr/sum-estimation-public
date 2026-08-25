# Step 2 — HNSW index creation

Loads a step-1 embedding run into a Qdrant collection with an HNSW index, attaches
the rarity-level payloads the sampler needs, and records the per-level maxima that
step 3 reads.

| Script | Purpose |
|---|---|
| `qdrant_insert.py` | Create the collection and index an embedding run into it |
| `test_index.py` | Smoke-test indexing end to end, no Docker required |

`qdrant_connection.py` resolves the connection, `embedding_inputs.py` reads step-1
output, and `config.py` holds the settings (all overridable in `.env`).

## Where the index runs

`--qdrant` picks one of three targets, and every script accepts it:

| Mode | What it is | Real HNSW? |
|---|---|---|
| `embedded` | The client's own local mode — no server, no Docker | **No** — exact search, HNSW parameters ignored |
| `docker` | A local Qdrant server, started automatically | Yes |
| `cloud` | The cluster in `QDRANT_HOST`, sharded and quantized | Yes |

`auto` (the default) means `cloud` when `QDRANT_HOST` is set, otherwise `docker`.

Use **`embedded`** to check the pipeline works — it starts instantly and needs
nothing installed beyond `requirements.txt`. Because it answers queries by exact
search, it validates ids, payloads, filters and recall *plumbing*, but it cannot
tell you anything about approximate-search quality.

Use **`docker`** for a real local index. The container is started for you
(`LOCAL_QDRANT_*` in `.env` controls image, port and volume); Docker must be
running.

## Indexing a run

```bash
cd "2. create_hnsw_index"

# quick local check, no Docker
python qdrant_insert.py --collection amazon-reviews_distilbert \
    --embeddings-prefix synthetic_amazon-reviews_distilbert --qdrant embedded

# real local HNSW index
python qdrant_insert.py --collection amazon-reviews_distilbert \
    --embeddings-prefix amazon-reviews_distilbert --qdrant docker
```

`--collection` is one of the three keys in `KNOWN_COLLECTION_CONFIGS`
(`open-images_resnet-50`, `open-images_clip_vit_l14_336`,
`amazon-reviews_distilbert`), which fixes the vector name, dimensionality,
distance and whether normalised vectors are expected. `--embeddings-prefix` is the
step-1 prefix; the sharded layout is read when `<prefix>_manifest.json` exists,
otherwise the single `<prefix>_embeddings[_normalised].npy`. Other flags:
`--embeddings-dir`, `--num-levels`, `--batch-size`, `--parallel`.

A dimensionality mismatch, a missing prefix, or a run whose vectors are raw when
the collection wants normalised is refused up front rather than half-indexed.

### What each point gets

```
id                              row position in the whole run — stable across shards
level_0 .. level_{N-1}          independent capped-geometric rarity draws
asset_id                        the step-1 asset id for that row, when the ids file is present
```

Level fields are queried **by exact value** — `level_i == v`, one query per value,
never a range — so their payload indexes are created for lookups only, which is
smaller and faster to build than one that also supports ranges.

Because the queries are exact-value, the largest value each field takes is what
tells a caller how many queries to issue. Each collection records its own maxima,
under `hnsw_index/` at the repo root:

```
hnsw_index/
  amazon-reviews_distilbert/max_levels.json
  open-images_resnet-50/max_levels.json
```

```json
{"amazon-reviews_distilbert": {"level_0": 24, "level_1": 24}}
```

One file per collection means indexing one never rewrites another's metadata, so
collections can be indexed in parallel; the file is still keyed by collection name
so several can be merged into one dict without ambiguity. Re-indexing replaces
that collection's file, written atomically. `HNSW_INDEX_DIR` moves the whole tree
(the tests point it at a scratch directory).

The Qdrant collection is named after the key itself, so `--collection
amazon-reviews_distilbert` indexes into a collection of that name.

## Efficiency notes

Indexing 10M points is dominated by upload round trips and index building, so:

- **Vectors stream shard by shard**, memory-mapped, so peak memory is one shard
  regardless of run size. Rarity levels are drawn per shard and the maxima
  accumulated, rather than materialising an `(N, num_levels)` matrix.
- **Uploads go through `upload_collection`**, which batches internally and takes
  the numpy array as-is instead of building a `PointStruct` per point.
  `--batch-size` sets points per request; `--parallel` adds workers, which pays off
  on a remote cluster where a single connection is the bottleneck.
- **HNSW building is deferred.** The collection is created with
  `indexing_threshold=0` so no graph is built while points are arriving, then the
  threshold is restored (`INDEXING_THRESHOLD`, default 10000) once the load is
  done, so the graph is built once. On a server this continues in the background
  after the script exits — the collection reports `green` when it has finished.
- **Graph parameters** are `HNSW_M` / `HNSW_EF_CONSTRUCT` /
  `HNSW_FULL_SCAN_THRESHOLD` in `.env` (defaults 32 / 32 / 10000), so a sweep does
  not need a code change.

## Testing

```bash
python test_index.py            # embedded mode: both input layouts
python test_index.py --docker   # also against a real local HNSW index
```

It writes a tiny run to a scratch directory (numpy only — step 1 is not needed),
indexes it, and checks the collection exists, the point count matches, every point
carries valid level fields and its step-1 `asset_id`, and a stored vector retrieves
itself. It then issues one exact-value query per observed `level_0` value and
checks they partition the points — every point reached by exactly one query, none
returned at the wrong level, an unassigned value coming back empty — and that the
collection's `max_levels.json` is the only file written and matches the highest
value actually stored.

## Extending to other HNSW indices

Everything Qdrant-specific in this directory sits behind two seams: how a
collection is created and loaded (`qdrant_insert.py`), and how it is connected to
(`qdrant_connection.py`). Adding a different local index — `hnswlib`, FAISS
`IndexHNSWFlat`, pgvector, LanceDB — means writing an equivalent of the first and
registering it in the second, with two things to keep in mind:

1. **Filtered search is the hard requirement, not vector search.** The sampler
   asks for nearest neighbours *among the points whose `level_i` equals a given
   value*. Qdrant does this natively with an indexed payload field. A backend
   without filtered search emulates it well precisely because the predicate is
   equality: the level values partition the points, so building one index per
   (level field, value) is exact rather than an approximation, and the indexes are
   small — level populations fall geometrically, so most of the data sits in the
   first few. The alternative, over-fetching and filtering client-side, changes
   the recall characteristics the experiments measure.
2. **The consumer is step 3, not this step.** `3. run experiments/qdrant_helpers.py`
   talks to Qdrant directly via `search_batch`. A second backend is only useful
   once that query path is behind the same interface — roughly `topk(query, k,
   level)`, `count(level)`, `sample(level, n)` and `fetch_vectors(ids)`, with
   `level` an exact value in `1 .. max_level`. Extract that first; otherwise a new
   index can be built but not measured.

The parts that are already backend-agnostic and worth reusing: `embedding_inputs.py`
(reads step-1 output whatever the layout), the rarity-level draw in
`qdrant_insert.py`, and the per-collection `max_levels.json` contract.
