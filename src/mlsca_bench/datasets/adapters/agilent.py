# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for per-trace Agilent/Keysight ``.bin`` waveforms (DPA Contest DES).

Each trace is one Agilent "AG10" binary waveform file whose sample count is read
from the header (so no per-dataset sample-length config is needed). For these
datasets the cryptographic material is encoded in the filename, e.g.
``wave_DES_HW_..._k=<hex>_m=<hex>_c=<hex>.bin``; ``filename_fields`` maps our
fields to regexes that capture those hex strings, decoded to bytes.
"""

from __future__ import annotations

import os
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset

_BUFFER_DTYPES = {1: "<i1", 2: "<i2", 4: "<f4"}


def _read_agilent(raw: bytes) -> np.ndarray[Any, Any]:
    """Decode the first waveform buffer of an Agilent 'AG10' .bin file."""

    if raw[:2] != b"AG":
        raise ValueError("Not an Agilent .bin waveform (missing 'AG' cookie).")
    header_size = struct.unpack_from("<i", raw, 12)[0]
    buffer_header = 12 + header_size
    data_header_size = struct.unpack_from("<i", raw, buffer_header)[0]
    _buffer_type, bytes_per_point = struct.unpack_from("<hh", raw, buffer_header + 4)
    buffer_size = struct.unpack_from("<i", raw, buffer_header + 8)[0]
    data_offset = buffer_header + data_header_size
    dtype = _BUFFER_DTYPES.get(bytes_per_point)
    if dtype is None:
        raise ValueError(f"Unsupported Agilent bytes-per-point: {bytes_per_point}.")
    return np.frombuffer(raw[data_offset : data_offset + buffer_size], dtype=np.dtype(dtype))


class _AgilentTraces:
    """Lazy (n, samples) view over one Agilent .bin file per trace."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = list(paths)
        first = _read_agilent(self._paths[0].read_bytes())
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
                [_read_agilent(self._paths[i].read_bytes()) for i in range(*index.indices(len(self._paths)))]
            )
        if index < 0:
            index += len(self._paths)
        return _read_agilent(self._paths[index].read_bytes())


class AgilentWaveDataset(SideChannelDataset):
    """Read a directory of per-trace Agilent ``.bin`` waveforms.

    ``file_glob`` selects the trace files (ordered by filename). Optional
    ``filename_fields`` maps a public field to a regex whose first group is a hex
    string decoded to bytes (used to recover key/plaintext/ciphertext embedded
    in the filenames).
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        file_glob: str,
        filename_fields: Mapping[str, str] | None = None,
        split: str | None = None,
        algorithm: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        self._name = name.strip()
        self._split = split
        self._closed = False

        paths = sorted(self._match(source, file_glob), key=lambda p: p.name)
        if not paths:
            raise FileNotFoundError(f"No files matched {file_glob!r} in the supplied path.")

        self._meta: dict[str, np.ndarray[Any, Any]] = {}
        for field, pattern in dict(filename_fields or {}).items():
            compiled = re.compile(pattern)
            rows = []
            for path in paths:
                match = compiled.search(path.name)
                if match is None:
                    raise ValueError(
                        f"Filename {path.name!r} has no {field} match for {pattern!r}."
                    )
                rows.append(np.frombuffer(bytes.fromhex(match.group(1)), dtype=np.uint8))
            self._meta[field] = np.stack(rows)

        self._traces = _AgilentTraces(paths)
        dataset_metadata: dict[str, Any] = {"format": "agilent-bin", "traces": len(paths)}
        if algorithm is not None:
            dataset_metadata["algorithm"] = algorithm
        dataset_metadata.update(metadata or {})
        self._metadata = MappingProxyType(dataset_metadata)
        validate_dataset(self)

    @staticmethod
    def _match(
        source: str | os.PathLike[str] | Iterable[Path], file_glob: str
    ) -> list[Path]:
        if isinstance(source, (str, os.PathLike)):
            roots: tuple[Path, ...] = (Path(source).expanduser(),)
        else:
            roots = tuple(Path(path).expanduser() for path in source)
        matches: list[Path] = []
        for root in roots:
            if root.is_dir():
                matches.extend(p for p in root.rglob(file_glob) if p.is_file())
            elif root.is_file():
                import fnmatch

                if fnmatch.fnmatch(root.name, file_glob):
                    matches.append(root)
        return matches

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


__all__ = ["AgilentWaveDataset"]
