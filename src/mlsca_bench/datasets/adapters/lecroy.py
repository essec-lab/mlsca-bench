# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for datasets stored as one LeCroy ``.trc`` waveform per trace.

Used by the DPA Contest v4 releases: a space-separated index file names, for
each trace, its cryptographic material (hex) and the ``.trc``/``.trc.bz2`` file
holding its samples. Each waveform is decoded from its WAVEDESC header (LeCroy
template 2.3); traces are read lazily, one file per accessed index.
"""

from __future__ import annotations

import warnings

import bz2
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset

# Byte offsets within the WAVEDESC block (LeCroy template 2.3).
_OFF_COMM_TYPE = 32
_OFF_COMM_ORDER = 34
_OFF_WAVE_DESCRIPTOR = 36
_OFF_USER_TEXT = 40
_OFF_TRIGTIME_ARRAY = 48
_OFF_WAVE_ARRAY_1 = 60


def _read_lecroy(data: bytes) -> np.ndarray[Any, Any]:
    """Decode the first data array of a LeCroy ``.trc`` waveform."""

    position = data[:64].find(b"WAVEDESC")
    if position < 0:
        raise ValueError("Not a LeCroy .trc waveform: WAVEDESC not found.")

    # COMM_ORDER is 0 (big-endian) or 1 (little-endian); either byte order reads
    # such a small value the same way, so read it little-endian first.
    comm_order = int.from_bytes(data[position + _OFF_COMM_ORDER : position + _OFF_COMM_ORDER + 2], "little")
    byteorder = "big" if comm_order == 0 else "little"
    np_order = ">" if comm_order == 0 else "<"

    def read_i16(offset: int) -> int:
        start = position + offset
        return int.from_bytes(data[start : start + 2], byteorder, signed=True)

    def read_i32(offset: int) -> int:
        start = position + offset
        return int.from_bytes(data[start : start + 4], byteorder, signed=True)

    comm_type = read_i16(_OFF_COMM_TYPE)
    wave_descriptor = read_i32(_OFF_WAVE_DESCRIPTOR)
    user_text = read_i32(_OFF_USER_TEXT)
    trigtime_array = read_i32(_OFF_TRIGTIME_ARRAY)
    wave_array_1 = read_i32(_OFF_WAVE_ARRAY_1)

    start = position + wave_descriptor + user_text + trigtime_array
    dtype = np.dtype(np_order + ("i1" if comm_type == 0 else "i2"))
    count = wave_array_1 // dtype.itemsize
    return np.frombuffer(data[start : start + wave_array_1], dtype=dtype, count=count)


def _load_trc(path: Path) -> np.ndarray[Any, Any]:
    raw = path.read_bytes()
    if path.suffix.casefold() == ".bz2":
        raw = bz2.decompress(raw)
    return _read_lecroy(raw)


class _TrcTraces:
    """Lazy (n, samples) view over one LeCroy file per trace."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = list(paths)
        first = _load_trc(self._paths[0])
        self._trailing = (int(first.shape[0]),)
        self._dtype = np.dtype(first.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._paths), *self._trailing)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return np.stack(
                [_load_trc(self._paths[i]) for i in range(*index.indices(len(self._paths)))]
            )
        if index < 0:
            index += len(self._paths)
        return _load_trc(self._paths[index])


class LeCroyIndexDataset(SideChannelDataset):
    """Read a per-trace LeCroy dataset described by a text index file.

    ``columns`` maps names to 0-based column positions in the index:
    ``filename`` (required) and optionally ``directory`` locate each trace file;
    ``keys``/``plaintexts``/``ciphertexts`` name hex columns decoded to bytes.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        index_filename: str,
        columns: Mapping[str, int],
        split: str | None = None,
        algorithm: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if "filename" not in columns:
            raise ValueError("A LeCroy index must map the 'filename' column.")

        self._name = name.strip()
        self._split = split
        self._closed = False

        files = self._index_files(source)
        index_matches = [p for p in files if p.name == index_filename]
        if not index_matches:
            raise FileNotFoundError(f"Index file {index_filename!r} was not found.")
        index_path = index_matches[0]
        by_name = {p.name: p for p in files}

        rows = [
            line.split()
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Index file {index_filename!r} is empty.")

        filename_col = columns["filename"]
        directory_col = columns.get("directory")
        paths: list[Path] = []
        kept_rows: list[list[str]] = []
        for row in rows:
            trace_name = row[filename_col]
            resolved = by_name.get(trace_name)
            if resolved is None:
                base = index_path.parent
                resolved = (
                    base / row[directory_col] / trace_name
                    if directory_col is not None
                    else base / trace_name
                )
                if not resolved.is_file():
                    continue          # part of a partial download (files=): not on disk
            paths.append(resolved)
            kept_rows.append(row)
        if not paths:
            raise FileNotFoundError(
                f"None of the {len(rows)} traces listed in {index_filename!r} is on disk under "
                f"{index_path.parent}; download at least one archive (files=)."
            )
        if len(paths) < len(rows):
            warnings.warn(
                f"{name}: {len(paths)} of the {len(rows)} traces in the index are downloaded; "
                "loading those only.",
                stacklevel=3,
            )
        rows = kept_rows

        self._meta: dict[str, np.ndarray[Any, Any]] = {}
        for field in ("keys", "plaintexts", "ciphertexts", "masks"):
            column = columns.get(field)
            if column is not None:
                self._meta[field] = np.stack(
                    [np.frombuffer(bytes.fromhex(row[column]), dtype=np.uint8) for row in rows]
                )

        self._traces = _TrcTraces(paths)
        dataset_metadata: dict[str, Any] = {
            "format": "lecroy-trc",
            "index": str(index_path),
            "traces": len(paths),
        }
        if algorithm is not None:
            dataset_metadata["algorithm"] = algorithm
        dataset_metadata.update(metadata or {})
        self._metadata = MappingProxyType(dataset_metadata)
        validate_dataset(self)

    @staticmethod
    def _index_files(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> tuple[Path, ...]:
        if isinstance(source, (str, os.PathLike)):
            roots: tuple[Path, ...] = (Path(source).expanduser(),)
        else:
            roots = tuple(Path(path).expanduser() for path in source)
        files: list[Path] = []
        for root in roots:
            if root.is_dir():
                files.extend(item for item in root.rglob("*") if item.is_file())
            elif root.is_file():
                files.append(root)
        return tuple(files)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self._name!r} is closed.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> DatasetArray:
        self._ensure_open()
        return self._traces

    @property
    def plaintexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._meta.get("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._meta.get("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        self._ensure_open()
        return self._meta.get("keys")

    @property
    def masks(self) -> DatasetArray | None:
        self._ensure_open()
        return self._meta.get("masks")

    @property
    def labels(self) -> None:
        self._ensure_open()
        return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def __len__(self) -> int:
        return len(self._traces)

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

        def meta(field: str) -> np.ndarray[Any, Any] | None:
            array = self._meta.get(field)
            return None if array is None else np.asarray(array[index])

        return TraceSample(
            trace=np.asarray(self._traces[index]),
            plaintext=meta("plaintexts"),
            ciphertext=meta("ciphertexts"),
            key=meta("keys"),
            mask=meta("masks"),
        )


__all__ = ["LeCroyIndexDataset"]
