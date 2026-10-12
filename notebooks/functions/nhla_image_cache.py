"""Decode-once, fixed-scale JPEG cache for NHLA classification training.

Scanner exports hold ~1M full-resolution face JPGs (1 mm/px along the board,
0.29 mm/px across it; ~1800-5000 x 370-1300 px). Re-reading and decoding them over
the Windows/WSL file share every epoch costs far more than the GPU work. This module
does that once: every face is resized by the same ``scale`` (so absolute board and
defect sizes are preserved, which matters for NHLA cutting rules), shrunk further only
if it would not fit the ``max_height`` x ``max_width`` canvas, and stored re-encoded
as a JPEG in sharded ``.bin`` files on the Linux filesystem.

Layout of a cache directory::

    shards/00000.bin       concatenated JPEG bytes for indices [0, SHARD_SIZE)
    shards/00000.idx.npy   (SHARD_SIZE, 4) int64 rows: offset, length, height, width
    index.npy              (N, 5) int64 rows: shard, offset, length, height, width
                           (length == 0 marks an image that failed to decode)

Building is resumable: each shard is written to a temp name and renamed when complete.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

SHARD_SIZE = 1024
JPEG_QUALITY = 95


def default_cache_root() -> Path:
    return Path(os.environ.get("NHLA_CACHE_DIR", Path.home() / ".cache" / "nhla_images"))


def scaled_size(width: int, height: int, scale: float, max_height: int, max_width: int) -> Tuple[int, int]:
    """Size after applying *scale*, shrunk (aspect preserved) only if it exceeds the canvas."""
    factor = min(scale, max_width / width, max_height / height)
    return max(1, round(width * factor)), max(1, round(height * factor))


def load_scaled(path: str, scale: float, max_height: int, max_width: int) -> Image.Image:
    with Image.open(path) as image:
        size = scaled_size(*image.size, scale, max_height, max_width)
        image.draft("RGB", size)  # JPEG DCT downscale: decode at 1/2..1/8 resolution when possible
        return image.convert("RGB").resize(size, Image.BILINEAR)


def _encode_shard(task) -> Tuple[int, int]:
    shard_id, paths, shard_dir, scale, max_height, max_width = task
    rows = np.zeros((len(paths), 4), dtype=np.int64)
    failed = 0
    tmp_bin = shard_dir / f"{shard_id:05d}.bin.tmp"
    with tmp_bin.open("wb") as fh:
        for row, path in enumerate(paths):
            try:
                image = load_scaled(path, scale, max_height, max_width)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
                payload = buffer.getvalue()
            except Exception:  # corrupt or missing image -> length 0
                failed += 1
                continue
            rows[row] = (fh.tell(), len(payload), image.height, image.width)
            fh.write(payload)
    os.replace(tmp_bin, shard_dir / f"{shard_id:05d}.bin")
    with (shard_dir / f"{shard_id:05d}.idx.tmp.npy").open("wb") as fh:
        np.save(fh, rows)
    os.replace(shard_dir / f"{shard_id:05d}.idx.tmp.npy", shard_dir / f"{shard_id:05d}.idx.npy")
    return shard_id, failed


def ensure_image_cache(
    paths: Sequence[str],
    scale: float,
    max_height: int,
    max_width: int,
    cache_root: str | Path | None = None,
    name_hint: str = "nhla",
    num_workers: int | None = None,
) -> Tuple[Path, Dict[str, int]]:
    """Build (or resume / reuse) the cache for *paths*.

    Returns ``(cache_dir, path_to_index)``. Images that failed to decode are left out
    of ``path_to_index`` so callers can drop those rows.
    """
    unique_paths = sorted(set(map(str, paths)))
    digest = hashlib.sha1("\n".join(unique_paths).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(cache_root or default_cache_root()) / f"{name_hint}_s{scale:g}_{max_height}x{max_width}_{digest}"
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.npy"

    if not index_path.exists():
        shard_count = (len(unique_paths) + SHARD_SIZE - 1) // SHARD_SIZE
        todo = [k for k in range(shard_count) if not (shard_dir / f"{k:05d}.idx.npy").exists()]
        if todo:
            num_workers = num_workers or max(1, (os.cpu_count() or 2) - 2)
            print(f"Building image cache: {len(todo)}/{shard_count} shards ({len(unique_paths):,} images) -> {cache_dir} ({num_workers} workers)")
            tasks = [
                (k, unique_paths[k * SHARD_SIZE:(k + 1) * SHARD_SIZE], shard_dir, scale, max_height, max_width)
                for k in todo
            ]
            with mp.get_context("fork").Pool(num_workers) as pool:
                for _ in tqdm(pool.imap_unordered(_encode_shard, tasks), total=len(tasks), unit="shard", desc="cache"):
                    pass

        index = np.zeros((len(unique_paths), 5), dtype=np.int64)
        for k in range(shard_count):
            rows = np.load(shard_dir / f"{k:05d}.idx.npy")
            index[k * SHARD_SIZE:k * SHARD_SIZE + len(rows), 0] = k
            index[k * SHARD_SIZE:k * SHARD_SIZE + len(rows), 1:] = rows
        np.save(index_path, index)
        (cache_dir / "meta.json").write_text(json.dumps({
            "count": len(unique_paths),
            "scale": scale,
            "max_height": max_height,
            "max_width": max_width,
            "jpeg_quality": JPEG_QUALITY,
            "failed": int((index[:, 2] == 0).sum()),
        }, indent=2))

    index = np.load(index_path)
    failed = int((index[:, 2] == 0).sum())
    if failed:
        print(f"Image cache: {failed} image(s) failed to decode and will be skipped")
    path_to_index = {path: idx for idx, path in enumerate(unique_paths) if index[idx, 2] > 0}
    return cache_dir, path_to_index
