"""Read whatever step 1 wrote: a sharded run, or a single-file one.

Step 1 shards its output by default (`<prefix>_00000_embeddings.npy` plus a
`<prefix>_manifest.json`) and writes one unnumbered matrix when run with
`--chunk-rows 0`. Both layouts are read here behind one interface, so the insert
script never branches on which it got.

Vectors are memory-mapped and handed over shard by shard, so indexing a run far
larger than RAM only ever holds one shard's worth of pages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Shard:
    """One matrix of vectors plus the asset ids that name its rows."""

    vectors_path: str
    ids_path: str | None
    rows: int


@dataclass(frozen=True)
class EmbeddingInput:
    """A whole step-1 run, in insertion order."""

    prefix: str
    variant: str
    shards: tuple[Shard, ...]
    dim: int
    sharded: bool

    @property
    def total_rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    def describe(self) -> str:
        layout = f"{len(self.shards)} shards" if self.sharded else "single file"
        return f"{self.total_rows} x {self.dim} {self.variant} vectors ({layout})"

    def iter_shards(self) -> Iterator[tuple[int, np.ndarray, list[str] | None]]:
        """Yield `(offset, vectors, asset_ids)` per shard.

        `offset` is the row's position in the whole run, which the caller uses as
        the point id so ids stay unique and stable across shards.
        """
        offset = 0
        for shard in self.shards:
            vectors = np.load(shard.vectors_path, mmap_mode="r")
            asset_ids = None
            if shard.ids_path and os.path.exists(shard.ids_path):
                with open(shard.ids_path, encoding="utf-8") as f:
                    asset_ids = f.read().splitlines()
                if len(asset_ids) != vectors.shape[0]:
                    raise SystemExit(
                        f"{shard.ids_path} has {len(asset_ids)} ids but "
                        f"{os.path.basename(shard.vectors_path)} has {vectors.shape[0]} "
                        f"rows; the two must line up row for row."
                    )
            yield offset, vectors, asset_ids
            offset += vectors.shape[0]


def _variant_key(normalised: bool) -> str:
    return "normalised" if normalised else "raw"


def resolve(embeddings_dir: str, prefix: str, normalised: bool) -> EmbeddingInput:
    """Locate a step-1 run, preferring the sharded layout."""
    variant = _variant_key(normalised)
    manifest_path = os.path.join(embeddings_dir, f"{prefix}_manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        chunks = manifest.get("chunks") or []
        if not chunks:
            raise SystemExit(f"{manifest_path} lists no shards; re-run step 1.")

        shards = []
        for chunk in chunks:
            files = chunk["files"]
            if variant not in files:
                raise SystemExit(
                    f"{manifest_path} has no '{variant}' vectors (it holds: "
                    f"{', '.join(k for k in files if k != 'ids')}). This collection needs "
                    f"{variant} vectors - re-run step 1 with --write {variant}."
                )
            shards.append(Shard(
                vectors_path=os.path.join(embeddings_dir, files[variant]),
                ids_path=os.path.join(embeddings_dir, files["ids"]) if "ids" in files else None,
                rows=chunk["rows"],
            ))
        missing = [s.vectors_path for s in shards if not os.path.exists(s.vectors_path)]
        if missing:
            raise SystemExit(
                f"{manifest_path} references files that are gone:\n  "
                + "\n  ".join(missing)
            )
        dim = int(np.load(shards[0].vectors_path, mmap_mode="r").shape[1])
        return EmbeddingInput(prefix, variant, tuple(shards), dim, sharded=True)

    suffix = "_embeddings_normalised.npy" if normalised else "_embeddings.npy"
    single = os.path.join(embeddings_dir, f"{prefix}{suffix}")
    if not os.path.exists(single):
        raise SystemExit(
            f"No step-1 output for prefix '{prefix}' in {embeddings_dir}.\n"
            f"Looked for {os.path.basename(manifest_path)} (sharded) and "
            f"{os.path.basename(single)} (single file).\n"
            f"Run step 1 first, or check EMBEDDINGS_DIR / --embeddings-prefix."
        )
    matrix = np.load(single, mmap_mode="r")
    ids_path = os.path.join(embeddings_dir, f"{prefix}_ids.txt")
    shard = Shard(single, ids_path if os.path.exists(ids_path) else None, int(matrix.shape[0]))
    return EmbeddingInput(prefix, variant, (shard,), int(matrix.shape[1]), sharded=False)
