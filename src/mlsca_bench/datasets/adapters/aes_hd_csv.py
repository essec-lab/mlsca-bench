# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lazy multi-file text adapter for the original AES-HD GitHub release."""

from __future__ import annotations

import os
import re
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset


_TRACE_FILE_PATTERN = re.compile(r"^traces_(\d+)\.csv$", re.IGNORECASE)


class _TextTraceView:
    """Indexed lazy matrix spanning space-delimited text files."""

    def __init__(
        self,
        owner: AESHDCSVDataset,
        paths: Sequence[Path],
        *,
        dtype: np.dtype[Any] = np.dtype(np.float32),
    ) -> None:
        self._owner = owner
        self._paths = tuple(paths)
        self._dtype = np.dtype(dtype)
        self._offsets: list[array[int]] = []
        self._ends: list[int] = []
        self._width: int | None = None

        total = 0
        for path in self._paths:
            offsets = array("Q")
            first_row: np.ndarray[Any, Any] | None = None
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        raise ValueError(f"Blank trace row found in {path}.")
                    offsets.append(offset)
                    if first_row is None:
                        first_row = self._parse(line, path)

            if not offsets or first_row is None:
                raise ValueError(f"Trace file is empty: {path}")
            if self._width is None:
                self._width = len(first_row)
            elif len(first_row) != self._width:
                raise ValueError(
                    "AES-HD trace files have inconsistent trace lengths: "
                    f"expected {self._width}, got {len(first_row)} in {path}."
                )
            self._offsets.append(offsets)
            total += len(offsets)
            self._ends.append(total)

        if self._width is None:
            raise ValueError("At least one AES-HD trace file is required.")
        self._shape = (total, self._width)

    def _parse(self, line: bytes, path: Path) -> np.ndarray[Any, Any]:
        try:
            values = np.fromstring(line.decode("ascii"), sep=" ", dtype=self._dtype)
        except UnicodeDecodeError as error:
            raise ValueError(f"Non-ASCII data found in {path}.") from error
        if values.size == 0:
            raise ValueError(f"Invalid trace row found in {path}.")
        if self._width is not None and values.size != self._width:
            raise ValueError(
                f"Trace row in {path} has {values.size} samples; "
                f"expected {self._width}."
            )
        return values

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("AES-HD trace index out of range.")
        file_index = bisect_right(self._ends, index)
        previous_end = 0 if file_index == 0 else self._ends[file_index - 1]
        return file_index, index - previous_end

    def _read_indices(self, indices: Sequence[int]) -> np.ndarray[Any, Any]:
        if not indices:
            return np.empty((0, self._shape[1]), dtype=self._dtype)
        result = np.empty((len(indices), self._shape[1]), dtype=self._dtype)
        with ExitStack() as stack:
            streams: dict[int, Any] = {}
            for result_index, index in enumerate(indices):
                file_index, row_index = self._location(index)
                if file_index not in streams:
                    streams[file_index] = stack.enter_context(
                        self._paths[file_index].open("rb")
                    )
                stream = streams[file_index]
                stream.seek(self._offsets[file_index][row_index])
                result[result_index] = self._parse(
                    stream.readline(), self._paths[file_index]
                )
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return self._read_indices(indices)
        if isinstance(index, (list, tuple, np.ndarray)):
            return self._read_indices([int(item) for item in index])
        if not isinstance(index, (int, np.integer)):
            raise TypeError("AES-HD trace indexes must be integers or slices.")
        return self._read_indices([int(index)])[0]


class AESHDCSVDataset(SideChannelDataset):
    """Read the six-file AES-HD text release without materializing traces."""

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str = "aes-hd-git",
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        self._name = name.strip()
        self._closed = False
        files = self._resolve_files(source)
        labels_path = next(path for path in files if path.name == "labels.csv")
        trace_paths = sorted(
            (path for path in files if _TRACE_FILE_PATTERN.match(path.name)),
            key=lambda path: int(_TRACE_FILE_PATTERN.match(path.name).group(1)),
        )
        if not trace_paths:
            raise FileNotFoundError("No AES-HD files named traces_<n>.csv were found.")

        self._traces = _TextTraceView(self, trace_paths)
        labels = np.loadtxt(labels_path, dtype=np.uint16, ndmin=1)
        self._labels = np.asarray(labels)
        if len(self._labels) != len(self._traces):
            raise ValueError(
                "AES-HD traces and labels are not aligned: "
                f"expected {len(self._traces)}, got {len(self._labels)}."
            )
        self._metadata = MappingProxyType(
            {
                "format": "space-delimited-text",
                "algorithm": "AES",
                "trace_files": tuple(str(path) for path in trace_paths),
                "labels_file": str(labels_path),
            }
        )
        validate_dataset(self)

    @staticmethod
    def _resolve_files(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> tuple[Path, ...]:
        if isinstance(source, (str, os.PathLike)):
            path = Path(source).expanduser()
            supplied_paths = (path,)
        else:
            supplied_paths = tuple(Path(path).expanduser() for path in source)
        files: list[Path] = []
        for path in supplied_paths:
            if path.is_dir():
                files.extend(item for item in path.iterdir() if item.is_file())
            elif path.is_file():
                files.append(path)
        if not any(path.name == "labels.csv" for path in files):
            raise FileNotFoundError("AES-HD labels.csv was not found.")
        return tuple(files)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self.name!r} is closed.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> None:
        return None

    @property
    def traces(self) -> DatasetArray:
        self._ensure_open()
        return self._traces

    @property
    def plaintexts(self) -> None:
        self._ensure_open()
        return None

    @property
    def ciphertexts(self) -> None:
        self._ensure_open()
        return None

    @property
    def keys(self) -> None:
        self._ensure_open()
        return None

    @property
    def masks(self) -> None:
        self._ensure_open()
        return None

    @property
    def labels(self) -> np.ndarray[Any, Any]:
        self._ensure_open()
        return self._labels

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
            raise TypeError("AES-HD indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("AES-HD index out of range.")
        return TraceSample(
            trace=np.asarray(self._traces[index]),
            label=np.asarray(self._labels[index]),
        )


__all__ = ["AESHDCSVDataset"]
