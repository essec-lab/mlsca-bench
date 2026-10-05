# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for two-class ``.npy`` datasets with filename-implied labels.

Used by the re-encryption distinguishing datasets: each implementation/split
holds two trace files (e.g. ``fixed.npy`` and ``random.npy``) whose label is the
file, not a stored array. The two files are memory-mapped and concatenated, and
a label vector (0 for the first class, 1 for the second) is synthesized.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._concat import ConcatArray
from .npy import _ArrayFieldsDataset, _FIELDS


class TwoClassNpyDataset(_ArrayFieldsDataset):
    """Concatenate two per-class ``.npy`` trace files and synthesize labels.

    ``class0_file`` gets label 0 and ``class1_file`` gets label 1. Files are
    located at ``[base_dir]/[split]/<class file>`` relative to the source; pass
    ``base_dir`` to select an implementation and ``split`` to select train/test.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        class0_file: str,
        class1_file: str,
        base_dir: str | None = None,
        split: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        files = self._all_files(source)

        def resolve(class_file: str) -> Path:
            parts = tuple(
                part for part in (base_dir, split, class_file) if part
            )
            matches = [p for p in files if p.parts[-len(parts):] == parts]
            if not matches:
                raise FileNotFoundError(
                    f"Required file was not found: {'/'.join(parts)}"
                )
            if len(matches) > 1:
                raise RuntimeError(f"Multiple files match {'/'.join(parts)}.")
            return matches[0]

        map0 = np.load(resolve(class0_file), mmap_mode="r", allow_pickle=False)
        map1 = np.load(resolve(class1_file), mmap_mode="r", allow_pickle=False)
        self._sources = [map0, map1]

        labels = np.concatenate(
            [np.zeros(len(map0), dtype=np.int8), np.ones(len(map1), dtype=np.int8)]
        )
        self._arrays = {field: None for field in _FIELDS}
        self._arrays["traces"] = ConcatArray([map0, map1])
        self._arrays["labels"] = labels

        dataset_metadata = {
            "format": "npy",
            "variant": "two-class",
            "classes": MappingProxyType({class0_file: 0, class1_file: 1}),
        }
        dataset_metadata.update(metadata or {})
        self._metadata = MappingProxyType(dataset_metadata)
        self._finalize()

    @staticmethod
    def _all_files(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> tuple[Path, ...]:
        if isinstance(source, (str, os.PathLike)):
            roots: tuple[Path, ...] = (Path(source).expanduser(),)
        else:
            roots = tuple(Path(path).expanduser() for path in source)
        files: list[Path] = []
        for root in roots:
            if root.is_dir():
                files.extend(p for p in root.rglob("*.npy") if p.is_file())
            elif root.is_file() and root.suffix.casefold() == ".npy":
                files.append(root)
        return tuple(files)

    def _release(self) -> None:
        for mmap in self._sources:
            handle = getattr(mmap, "_mmap", None)
            if handle is not None:
                handle.close()


__all__ = ["TwoClassNpyDataset"]
