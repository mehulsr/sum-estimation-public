"""Qdrant connections for the three ways this step gets run.

    cloud     QDRANT_HOST is set - a real cluster, sharded and quantized.
    docker    no host set - a local server is started via Docker. Builds a real
              HNSW index, which is what step 3's recall numbers depend on.
    embedded  no server at all: the client's own local mode, backed by a
              directory or pure memory. Starts instantly and needs no Docker, so
              it is what the tests use - but note it answers queries by exact
              search and ignores the HNSW parameters, so it validates the
              pipeline, not the index.

`connect()` picks between them and reports which one it used; every script takes
`--qdrant` to force a specific mode.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass

from qdrant_client import QdrantClient

from config import settings

# Vector configuration for each known collection. `vector_name` matches
# `collections_dict` in `3. run experiments/qdrant_helpers.py`, and `normalised`
# selects which step-1 embedding variant to load (raw vs L2-normalised).
KNOWN_COLLECTION_CONFIGS = {
    "open-images_resnet-50": {"vector_name": "abs1", "size": 2048, "distance": "Euclid", "normalised": False},
    "open-images_clip_vit_l14_336": {"vector_name": "unit", "size": 768, "distance": "Dot", "normalised": True},
    "amazon-reviews_distilbert": {"vector_name": "abs", "size": 768, "distance": "Euclid", "normalised": False},
}

MODES = ("auto", "cloud", "docker", "embedded")


@dataclass
class Target:
    """A connected Qdrant plus what kind of deployment it is."""

    client: QdrantClient
    mode: str
    location: str

    @property
    def is_server(self) -> bool:
        """True when a real Qdrant process is answering, i.e. HNSW is real."""
        return self.mode in ("cloud", "docker")

    @property
    def is_cloud(self) -> bool:
        """True only for a distributed cluster, where sharding/quantization apply."""
        return self.mode == "cloud"


def _is_qdrant_ready(url: str) -> bool:
    for endpoint in ("/readyz", "/healthz"):
        try:
            with urllib.request.urlopen(f"{url}{endpoint}", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def ensure_local_qdrant(timeout_sec: int = 60) -> str:
    """Ensure a local Qdrant server is running via Docker; return its URL."""
    url = f"http://localhost:{settings.LOCAL_QDRANT_PORT}"
    if _is_qdrant_ready(url):
        print(f"[Qdrant] Reusing local server at {url}")
        return url

    if shutil.which("docker") is None:
        raise SystemExit(
            "Docker is needed to start a local Qdrant server. Either start Docker, "
            "set QDRANT_HOST for a remote cluster, or use --qdrant embedded (no "
            "server, exact search instead of HNSW)."
        )

    print(f"[Qdrant] Starting local server '{settings.LOCAL_QDRANT_CONTAINER}' "
          f"({settings.LOCAL_QDRANT_IMAGE}) on port {settings.LOCAL_QDRANT_PORT}")
    subprocess.run(["docker", "rm", "-f", settings.LOCAL_QDRANT_CONTAINER], capture_output=True)
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", settings.LOCAL_QDRANT_CONTAINER,
            "-p", f"{settings.LOCAL_QDRANT_PORT}:6333",
            "-v", f"{settings.LOCAL_QDRANT_VOLUME}:/qdrant/storage",
            settings.LOCAL_QDRANT_IMAGE,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Could not start the local Qdrant container. Is the Docker daemon "
            f"running?\n{result.stderr.strip()}\n"
            "Alternatively use --qdrant embedded, which needs no Docker."
        )

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _is_qdrant_ready(url):
            print(f"[Qdrant] Local server ready at {url}")
            return url
        time.sleep(2)
    raise SystemExit(f"Local Qdrant did not become ready within {timeout_sec}s.")


def connect(mode: str = "auto", embedded_path: str | None = None) -> Target:
    """Connect to Qdrant in the requested mode (see the module docstring)."""
    if mode not in MODES:
        raise SystemExit(f"Unknown --qdrant mode '{mode}'. Choose from: {', '.join(MODES)}.")
    if mode == "auto":
        mode = "cloud" if settings.QDRANT_HOST else "docker"

    if mode == "cloud":
        if not settings.QDRANT_HOST:
            raise SystemExit("--qdrant cloud needs QDRANT_HOST set in .env.")
        print(f"[Qdrant] Using cloud cluster at {settings.QDRANT_HOST}")
        client = QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY or None,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )
        return Target(client, "cloud", settings.QDRANT_HOST)

    if mode == "docker":
        url = ensure_local_qdrant()
        return Target(QdrantClient(url=url, timeout=settings.QDRANT_TIMEOUT), "docker", url)

    location = embedded_path or settings.QDRANT_EMBEDDED_PATH
    if location in ("", ":memory:"):
        print("[Qdrant] Embedded in-memory (exact search, HNSW parameters ignored)")
        return Target(QdrantClient(location=":memory:"), "embedded", ":memory:")
    print(f"[Qdrant] Embedded at {location} (exact search, HNSW parameters ignored)")
    return Target(QdrantClient(path=location), "embedded", location)


def add_qdrant_args(parser) -> None:
    """Register the connection flags shared by the scripts here."""
    parser.add_argument(
        "--qdrant", choices=MODES, default="auto",
        help="Where to index: 'auto' (default: cloud when QDRANT_HOST is set, else "
             "a Docker server), 'docker' for a real local HNSW index, or 'embedded' "
             "for a serverless client with no Docker (exact search, no HNSW).",
    )
    parser.add_argument(
        "--embedded-path", default=None,
        help="Directory for --qdrant embedded (default: QDRANT_EMBEDDED_PATH, or "
             "in-memory when that is ':memory:').",
    )


def get_qdrant_client(mode: str = "auto", embedded_path: str | None = None) -> QdrantClient:
    """Convenience wrapper for callers that do not care about the deployment kind."""
    return connect(mode, embedded_path).client
