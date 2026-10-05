# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for SIMPLE-DATASET manifest containers (e.g. the SMAesH dataset).

A manifest dataset is a directory with a ``manifest.json`` that lists ordered
chunks; each chunk names one ``.npy`` file per field (``traces``,
``umsk_plaintext``, ``umsk_key``, ...). The per-field chunk files are memory-
mapped and concatenated, so the (very large) dataset stays lazy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset
from ._concat import ConcatArray
from .npy import _FIELDS
from ..errors import describe_paths

_ABOUT = "SIMPLE-DATASET-MANIFEST"


def _find_manifest(
    source: str | os.PathLike[str] | Iterable[Path], subdir: str | None = None
) -> Path:
    """Locate the ``manifest.json`` to load.

    ``subdir`` names a split partition (e.g. ``"vk0"``); when several manifests
    are present (one per extracted split archive), only the one whose path
    contains that fragment is used. If exactly one manifest exists, it is
    returned regardless (so pointing the path straight at one split dir works).
    """

    if isinstance(source, (str, os.PathLike)):
        roots: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        roots = tuple(Path(path).expanduser() for path in source)
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "manifest.json":
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob("manifest.json"))
    if not found:
        raise FileNotFoundError(f"No manifest.json was found in {describe_paths(roots)}.")

    if subdir is not None and len(found) > 1:
        narrowed = [m for m in found if subdir in str(m)]
        if len(narrowed) == 1:
            return narrowed[0]
        if len(narrowed) > 1:
            raise RuntimeError(
                f"Multiple manifest.json files match split {subdir!r}; point the "
                "path at the parent of the extracted split directories."
            )
        raise FileNotFoundError(
            f"No manifest.json for split {subdir!r} was found under the path."
        )

    if len(found) > 1:
        raise RuntimeError(
            "Multiple manifest.json files were found; point the path at one "
            "dataset directory (or pass a split)."
        )
    return found[0]


class ManifestNpyDataset(SideChannelDataset):
    """Assemble a manifest dataset's fields from its per-chunk ``.npy`` files.

    ``field_map`` maps each public field to the manifest field name that holds
    it, e.g. ``{"traces": "traces", "keys": "umsk_key"}``.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        field_map: Mapping[str, str],
        split: str | None = None,
        subdir: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if "traces" not in field_map:
            raise ValueError("A manifest field_map must map the traces field.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        self._maps: list[np.memmap] = []

        manifest_path = _find_manifest(source, subdir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("about") != _ABOUT:
            raise ValueError(f"{manifest_path} is not a {_ABOUT}.")
        base = manifest_path.parent
        chunks = manifest.get("chunks", {})
        if not chunks:
            raise ValueError("Manifest declares no chunks.")

        self._fields: dict[str, DatasetArray | None] = {field: None for field in _FIELDS}
        try:
            chunk_list = list(chunks.values())
            for public_name, manifest_field in field_map.items():
                present = [manifest_field in c["files"] for c in chunk_list]
                if not any(present):
                    # A field can be legitimately absent from a split (e.g. the
                    # SMAesH fixed-key attack set has no msk_key); expose None.
                    # ``traces`` is mandatory.
                    if public_name == "traces":
                        raise KeyError(
                            f"Required traces field {manifest_field!r} is missing."
                        )
                    self._fields[public_name] = None
                    continue
                if not all(present):
                    raise KeyError(
                        f"Field {manifest_field!r} is present in some chunks but "
                        "not others."
                    )
                parts = []
                for chunk in chunk_list:
                    mmap = np.load(
                        base / chunk["files"][manifest_field]["path"],
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    self._maps.append(mmap)
                    parts.append(mmap)
                self._fields[public_name] = ConcatArray(parts)

            dataset_metadata = {
                "format": "npy-manifest",
                "id": manifest.get("id"),
                "path": str(manifest_path),
                "chunks": len(chunks),
            }
            dataset_metadata.update(metadata or {})
            self._metadata = MappingProxyType(dataset_metadata)
            validate_dataset(self)
        except Exception:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self._name!r} is closed.")

    def _field(self, field_name: str) -> DatasetArray | None:
        self._ensure_open()
        return self._fields[field_name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> DatasetArray:
        value = self._field("traces")
        assert value is not None
        return value

    @property
    def plaintexts(self) -> DatasetArray | None:
        return self._field("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        return self._field("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        return self._field("keys")

    @property
    def masks(self) -> DatasetArray | None:
        return self._field("masks")

    @property
    def labels(self) -> DatasetArray | None:
        return self._field("labels")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        for mmap in self._maps:
            handle = getattr(mmap, "_mmap", None)
            if handle is not None:
                handle.close()
        self._closed = True

    def __len__(self) -> int:
        value = self._fields["traces"]
        assert value is not None
        return len(value)

    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        self._ensure_open()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range.")

        def read(field_name: str) -> np.ndarray[Any, Any] | None:
            value = self._fields[field_name]
            return None if value is None else np.asarray(value[index])

        trace = read("traces")
        assert trace is not None
        return TraceSample(
            trace=trace,
            plaintext=read("plaintexts"),
            ciphertext=read("ciphertexts"),
            key=read("keys"),
            mask=read("masks"),
            label=read("labels"),
        )


__all__ = ["ManifestNpyDataset"]
