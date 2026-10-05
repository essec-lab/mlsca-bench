# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for datasets stored as flat, headerless binary (``.dat``/``.bin``).

Each field lives in its own file as a contiguous block of a fixed dtype. A
field spec gives the file name, the element dtype, the number of elements per
trace (``item_length``), and an optional leading ``offset`` of header bytes to
skip. Files are memory-mapped, so large trace files are never copied wholesale.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .npy import _ArrayFieldsDataset, _FIELDS


class RawBinaryDataset(_ArrayFieldsDataset):
    """Load aligned fields from flat binary files using memory maps.

    ``layout`` maps each field to a spec mapping with keys ``filename`` (a
    template that may contain ``{split}``), ``dtype``, ``item_length`` (the
    number of elements per trace/record), and an optional ``offset`` giving the
    number of leading header bytes to skip.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        layout: Mapping[str, Mapping[str, Any]] | None = None,
        record: Mapping[str, Any] | None = None,
        split: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if (layout is None) == (record is None):
            raise ValueError(
                "Provide exactly one of 'layout' (one file per field) or "
                "'record' (a single interleaved-record file)."
            )
        if layout is not None and "traces" not in layout:
            raise ValueError("A raw-binary layout must define a traces file.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        self._record_mmap: np.memmap | None = None
        by_name = self._index_files(source)

        self._arrays = {}
        try:
            if record is not None:
                self._map_record(record, by_name)
            else:
                for field_name in _FIELDS:
                    spec = layout.get(field_name)
                    if spec is None:
                        self._arrays[field_name] = None
                        continue
                    self._arrays[field_name] = self._map_field(
                        field_name, spec, by_name
                    )

            dataset_metadata = {"format": "binary"}
            dataset_metadata.update(metadata or {})
            self._metadata = MappingProxyType(dataset_metadata)
            self._finalize()
        except Exception:
            self.close()
            raise

    def _map_record(
        self, record: Mapping[str, Any], by_name: Mapping[str, list[Path]]
    ) -> None:
        """Memory-map one file of fixed interleaved records and split its fields."""

        filename = str(record["filename"]).format(split=self._split or "")
        offset = int(record.get("offset", 0))
        fields = record["fields"]
        if "traces" not in fields:
            raise ValueError("A raw-binary record must map the 'traces' field.")

        entries = []
        for entry in record["dtype"]:
            field_name, format_string = entry[0], entry[1]
            if len(entry) >= 3:
                shape = entry[2]
                entries.append(
                    (field_name, format_string, tuple(shape))
                    if isinstance(shape, (list, tuple))
                    else (field_name, format_string, (int(shape),))
                )
            else:
                entries.append((field_name, format_string))
        record_dtype = np.dtype(entries)

        matches = by_name.get(filename, [])
        if not matches:
            raise FileNotFoundError(f"Required binary file was not found: {filename}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple binary files named {filename} were found.")

        self._record_mmap = np.memmap(
            matches[0], dtype=record_dtype, mode="r", offset=offset
        )
        for field_name in _FIELDS:
            source_field = fields.get(field_name)
            if source_field is None:
                self._arrays[field_name] = None
                continue
            column = self._record_mmap[source_field]
            if column.ndim == 1:
                column = column.reshape(len(column), 1)
            self._arrays[field_name] = column

    def _map_field(
        self,
        field_name: str,
        spec: Mapping[str, Any],
        by_name: Mapping[str, list[Path]],
    ) -> np.memmap:
        filename = str(spec["filename"]).format(split=self._split or "")
        item_length = int(spec["item_length"])
        if item_length < 1:
            raise ValueError(f"{field_name} item_length must be a positive integer.")
        dtype = np.dtype(spec["dtype"])
        offset = int(spec.get("offset", 0))

        matches = by_name.get(filename, [])
        if not matches:
            raise FileNotFoundError(f"Required binary file was not found: {filename}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple binary files named {filename} were found.")

        path = matches[0]
        payload = path.stat().st_size - offset
        stride = item_length * dtype.itemsize
        if payload < 0 or payload % stride != 0:
            raise ValueError(
                f"{filename} size is not a whole number of {field_name} records "
                f"of {item_length} {dtype.name} values."
            )
        count = payload // stride
        mapped = np.memmap(
            path, dtype=dtype, mode="r", offset=offset, shape=(count, item_length)
        )
        return mapped

    @staticmethod
    def _index_files(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> dict[str, list[Path]]:
        if isinstance(source, (str, os.PathLike)):
            supplied: tuple[Path, ...] = (Path(source).expanduser(),)
        else:
            supplied = tuple(Path(path).expanduser() for path in source)
        by_name: dict[str, list[Path]] = {}
        for path in supplied:
            if path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file():
                        by_name.setdefault(item.name, []).append(item)
            elif path.is_file():
                by_name.setdefault(path.name, []).append(path)
        return by_name

    def _release(self) -> None:
        for value in self._arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        # Record-mode field views keep the memory map alive; drop the reference
        # and let it close when the views are collected (avoids use-after-free).
        self._record_mmap = None


__all__ = ["RawBinaryDataset"]
