# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for datasets stored as one ASCII waveform file per trace.

Used by the DPA Contest v2 public base, whose per-trace files are Agilent
oscilloscope ASCII exports: a run of ``#`` comment/header lines followed by one
integer sample per line. Files are enumerated (ordered by an index parsed from
the filename) and read lazily, one file per accessed trace.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset


class _AsciiTraces:
    """Lazy (n, samples) view over one ASCII waveform file per trace."""

    def __init__(
        self, paths: Sequence[Path], comment_prefix: str, dtype: np.dtype[Any]
    ) -> None:
        self._paths = list(paths)
        self._comment = comment_prefix
        self._dtype = dtype
        first = self._read(0)
        self._trailing = (int(first.shape[0]),)

    def _read(self, index: int) -> np.ndarray[Any, Any]:
        values = []
        with open(self._paths[index], "r", encoding="latin1") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith(self._comment):
                    values.append(int(line))
        return np.array(values, dtype=self._dtype)

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
            return np.stack([self._read(i) for i in range(*index.indices(len(self._paths)))])
        if index < 0:
            index += len(self._paths)
        return self._read(index)


class AsciiWaveDataset(SideChannelDataset):
    """Read a directory of per-trace ASCII waveform files as one trace matrix.

    ``file_glob`` selects the trace files; ``comment_prefix`` marks header lines
    to skip; ``index_regex`` (if given) extracts an integer from each filename to
    order the traces (otherwise a natural filename sort is used).

    ``filename_fields`` maps a public field (``plaintexts``/``keys``/...) to a
    regex whose first group is a hex string decoded to bytes, recovering
    cryptographic material embedded in the filename (as in the DPA Contest v2
    ``wave_..._k=<hex>_m=<hex>_c=<hex>.csv`` naming).
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        file_glob: str,
        comment_prefix: str = "#",
        dtype: str = "int16",
        index_regex: str | None = None,
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

        paths = self._match_files(source, file_glob)
        if not paths:
            raise FileNotFoundError(f"No files matched {file_glob!r} in the supplied path.")

        if index_regex is not None:
            pattern = re.compile(index_regex)

            def sort_key(path: Path) -> tuple[int, int, str]:
                match = pattern.search(path.name)
                return (0, int(match.group(1)), "") if match else (1, 0, path.name)

            paths = sorted(paths, key=sort_key)
        else:
            paths = sorted(
                paths,
                key=lambda p: [
                    int(t) if t.isdigit() else t.casefold()
                    for t in re.split(r"(\d+)", p.name)
                ],
            )

        # Cryptographic material carried in the filenames (kept aligned with the
        # sorted trace order), decoded hex -> per-trace byte rows.
        self._meta: dict[str, np.ndarray[Any, Any]] = {}
        for field, regex in dict(filename_fields or {}).items():
            compiled = re.compile(regex)
            rows = []
            for path in paths:
                match = compiled.search(path.name)
                if match is None:
                    raise ValueError(
                        f"Filename {path.name!r} has no {field} match for {regex!r}."
                    )
                rows.append(np.frombuffer(bytes.fromhex(match.group(1)), dtype=np.uint8))
            self._meta[field] = np.stack(rows)

        self._traces = _AsciiTraces(paths, comment_prefix, np.dtype(dtype))
        dataset_metadata: dict[str, Any] = {"format": "ascii-wave", "traces": len(paths)}
        if algorithm is not None:
            dataset_metadata["algorithm"] = algorithm
        dataset_metadata.update(metadata or {})
        self._metadata = MappingProxyType(dataset_metadata)
        validate_dataset(self)

    @staticmethod
    def _match_files(
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
    def labels(self) -> DatasetArray | None:
        self._ensure_open()
        return self._meta.get("labels")

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

        def field(name: str) -> np.ndarray[Any, Any] | None:
            array = self._meta.get(name)
            return None if array is None else np.asarray(array[index])

        return TraceSample(
            trace=np.asarray(self._traces[index]),
            plaintext=field("plaintexts"),
            ciphertext=field("ciphertexts"),
            key=field("keys"),
            mask=field("masks"),
            label=field("labels"),
        )


__all__ = ["AsciiWaveDataset"]
