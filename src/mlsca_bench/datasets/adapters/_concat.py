# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lazily present a dataset split that is spread across several chunk files."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import (
    DatasetArray,
    SideChannelDataset,
    TraceSample,
    validate_dataset,
)

_FIELDS = ("traces", "plaintexts", "ciphertexts", "keys", "masks", "labels")


class ConcatArray:
    """A read-only, axis-0 concatenation view over several array-likes.

    The parts are never copied; element and contiguous-slice reads touch only
    the parts they overlap, so a concatenated field stays as lazy as its backing
    HDF5 datasets or memory maps.
    """

    def __init__(self, parts: Sequence[DatasetArray]) -> None:
        if not parts:
            raise ValueError("ConcatArray requires at least one part.")
        self._parts = tuple(parts)
        self._lengths = [len(part) for part in self._parts]
        self._offsets = list(itertools.accumulate(self._lengths))
        self._total = self._offsets[-1]
        first = self._parts[0]
        self._trailing = tuple(first.shape[1:])
        self._dtype = np.dtype(first.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._total, *self._trailing)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._total

    def _part_of(self, index: int) -> tuple[int, int]:
        position = bisect.bisect_right(self._offsets, index)
        start = self._offsets[position - 1] if position else 0
        return position, index - start

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._total)
            if step == 1:
                pieces = []
                begin = 0
                for part, length in zip(self._parts, self._lengths):
                    end = begin + length
                    lower, upper = max(start, begin), min(stop, end)
                    if lower < upper:
                        pieces.append(np.asarray(part[lower - begin : upper - begin]))
                    begin = end
                if pieces:
                    return np.concatenate(pieces)
                return np.empty((0, *self._trailing), dtype=self._dtype)
            return np.stack([self[item] for item in range(start, stop, step)])
        if not isinstance(index, (int, np.integer)):
            raise TypeError("ConcatArray indexes must be integers or slices.")
        if index < 0:
            index += self._total
        if index < 0 or index >= self._total:
            raise IndexError("ConcatArray index out of range.")
        part_index, local = self._part_of(index)
        return np.asarray(self._parts[part_index][local])


class ConcatDataset(SideChannelDataset):
    """Stitch several same-schema chunk datasets into one logical split.

    Each chunk is loaded by an ordinary single-file adapter; this wrapper
    concatenates their aligned fields and routes sample reads to the owning
    chunk, so no traces are copied into memory.
    """

    def __init__(
        self,
        datasets: Sequence[SideChannelDataset],
        *,
        name: str,
        split: str | None,
    ) -> None:
        if not datasets:
            raise ValueError("ConcatDataset requires at least one chunk dataset.")
        self._datasets = tuple(datasets)
        self._name = name
        self._split = split
        self._closed = False

        self._fields: dict[str, ConcatArray | None] = {}
        for field_name in _FIELDS:
            values = [getattr(dataset, field_name) for dataset in self._datasets]
            present = [value is not None for value in values]
            if not any(present):
                self._fields[field_name] = None
            elif not all(present):
                raise ValueError(
                    f"Field {field_name!r} is present in some chunks but not others."
                )
            else:
                self._fields[field_name] = ConcatArray(values)

        # Forward optional adapter-specific fields (Chameleon: pinpoints, and
        # DFS-only frequencies) when every chunk exposes them, so they survive
        # chunking. getattr keeps non-Chameleon chunked datasets unaffected.
        self._extra: dict[str, ConcatArray | None] = {}
        for extra_name in ("pinpoints", "frequencies"):
            values = [getattr(dataset, extra_name, None) for dataset in self._datasets]
            self._extra[extra_name] = (
                ConcatArray(values) if all(v is not None for v in values) else None
            )

        self._lengths = [len(dataset) for dataset in self._datasets]
        self._offsets = list(itertools.accumulate(self._lengths))
        base_metadata = dict(self._datasets[0].metadata)
        base_metadata["chunks"] = len(self._datasets)
        self._metadata = MappingProxyType(base_metadata)
        validate_dataset(self)

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
    def pinpoints(self) -> DatasetArray | None:
        """Chameleon per-trace AES start/end offsets, if the chunks provide them."""
        self._ensure_open()
        return self._extra["pinpoints"]

    @property
    def frequencies(self) -> DatasetArray | None:
        """Chameleon DFS per-trace clock schedule, if the chunks provide it."""
        self._ensure_open()
        return self._extra["frequencies"]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        for dataset in self._datasets:
            dataset.close()
        self._closed = True

    def __len__(self) -> int:
        return self._offsets[-1]

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
        position = bisect.bisect_right(self._offsets, index)
        start = self._offsets[position - 1] if position else 0
        return self._datasets[position][index - start]


__all__ = ["ConcatArray", "ConcatDataset"]
