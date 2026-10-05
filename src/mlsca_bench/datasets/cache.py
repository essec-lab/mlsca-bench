# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""See what is in the dataset cache, how much space it takes, and remove datasets."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .download import cache_root
from .registry import _ALIAS_STORE, DatasetRegistryError, _REGISTRY, format_size, normalize_dataset_name


@dataclass(frozen=True)
class CachedDataset:
    """One dataset folder in the cache."""

    name: str
    path: Path
    size_bytes: int
    known: bool            # False for folders that match no registered dataset

    @property
    def size(self) -> str:
        return format_size(self.size_bytes)


def _folder_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).lstat().st_size
            except OSError:
                pass
    return total


def cached_datasets(destination: str | os.PathLike[str] | None = None) -> tuple[CachedDataset, ...]:
    """List the datasets downloaded to the cache (largest first)."""

    root = cache_root(destination)
    if not root.is_dir():
        return ()
    entries = [
        CachedDataset(p.name, p, _folder_size(p), p.name in _REGISTRY)
        for p in root.iterdir()
        if p.is_dir() and not p.is_symlink()
    ]
    return tuple(sorted(entries, key=lambda e: (-e.size_bytes, e.name)))


def cache_usage(destination: str | os.PathLike[str] | None = None) -> int:
    """Total bytes used by the dataset cache."""

    return sum(entry.size_bytes for entry in cached_datasets(destination))


def remove_cached_dataset(name: str, destination: str | os.PathLike[str] | None = None) -> int:
    """Delete one dataset's downloaded files from the cache and return the bytes freed.

    Only a folder directly inside the cache can be removed; your own datasets
    registered with a ``path`` are never touched.
    """

    root = cache_root(destination).resolve()
    if not name or any(sep in name for sep in ("/", "\\")) or name in (".", ".."):
        raise DatasetRegistryError(f"{name!r} is not a dataset name.")
    canonical = _ALIAS_STORE.get(normalize_dataset_name(name), name)
    target = (root / canonical).resolve()
    if target.parent != root or canonical in ("", ".", ".."):
        raise DatasetRegistryError(f"Refusing to remove {target}: it is not a dataset folder inside {root}.")
    if not target.is_dir():
        raise FileNotFoundError(f"{canonical!r} is not in the cache ({root}).")
    freed = _folder_size(target)
    try:
        shutil.rmtree(target)
    except PermissionError as error:
        raise PermissionError(
            f"Could not remove {target}: {error.filename or error} is in use. Close the datasets "
            "that use it (on Windows an open file cannot be deleted), or restart Python, and try again."
        ) from error
    return freed


__all__ = ["CachedDataset", "cache_usage", "cached_datasets", "remove_cached_dataset"]
