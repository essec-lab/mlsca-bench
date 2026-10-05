# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the Chameleon HDF5 layout (one dataset per trace).

Chameleon does not store a single 2-D ``traces`` matrix. Each chunk file holds
``data/traces/trace_0 ... trace_{n-1}`` (one 1-D int16 dataset per trace) and a
parallel ``metadata/ciphers/ciphers_{i}`` subgroup whose ``key`` and
``plaintexts`` datasets carry the aligned key/plaintext (compound columns ``k``
and ``p``). Full subsets are spread over 16 chunk files; the chunked loader
stitches them with :class:`~mlsca_bench.datasets.adapters._concat.ConcatDataset`.

Each 134M-sample trace holds several AES executions interleaved with other code;
their ``start``/``end`` offsets live in ``metadata/pinpoints/pinpoints_{i}`` and
are exposed through the (Chameleon-specific) :attr:`ChameleonDataset.pinpoints`
attribute for segmenting a trace into per-operation attack windows. For the DFS
sub-dataset, ``metadata/frequencies/frequencies_{i}`` (compound ``sample`` /
``frequency`` fields) records the clock schedule and is exposed via
:attr:`ChameleonDataset.frequencies` for resampling to a constant clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset
from ..errors import MissingDependencyError

try:
    import h5py
except ImportError as error:  # pragma: no cover
    raise MissingDependencyError(
        "h5py", extra="hdf5", feature="Loading Chameleon HDF5 datasets"
    ) from error


class _PerTraceView:
    """Present ``trace_0 ... trace_{n-1}`` 1-D datasets as one (n, length) array."""

    def __init__(self, owner: "ChameleonDataset", group: Any, count: int) -> None:
        self._owner = owner
        self._group = group
        self._count = count
        first = group["trace_0"]
        self._trailing = (int(first.shape[0]),)
        self._dtype = np.dtype(first.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._count, *self._trailing)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        if isinstance(index, slice):
            return np.stack(
                [np.asarray(self._group[f"trace_{i}"]) for i in range(*index.indices(self._count))]
            )
        if index < 0:
            index += self._count
        return np.asarray(self._group[f"trace_{index}"])


class _CipherView:
    """Present ``ciphers_{i}/<name>`` compound datasets as one (n, width) array."""

    def __init__(
        self,
        owner: "ChameleonDataset",
        group: Any,
        dataset_name: str,
        column: str,
        count: int,
    ) -> None:
        self._owner = owner
        self._group = group
        self._dataset_name = dataset_name
        self._column = column
        self._count = count
        sample = self._read(0)
        self._trailing = tuple(sample.shape)
        self._dtype = np.dtype(sample.dtype)

    def _read(self, index: int) -> np.ndarray[Any, Any]:
        record = self._group[f"ciphers_{index}/{self._dataset_name}"]
        value = np.asarray(record[self._column])
        return value.reshape(-1)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._count, *self._trailing)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        if isinstance(index, slice):
            return np.stack([self._read(i) for i in range(*index.indices(self._count))])
        if index < 0:
            index += self._count
        return self._read(index)


class _PinpointView:
    """Present ``pinpoints_{i}`` compound datasets as per-trace ``(n_ops, 2)`` arrays.

    Each Chameleon trace contains several AES executions interleaved with other
    code; ``metadata/pinpoints/pinpoints_{i}`` is a compound dataset with
    ``start``/``end`` sample offsets, one record per execution. The op count
    varies per trace, so this view is ragged: ``[i]`` returns that trace's
    ``(n_ops_i, 2)`` ``[start, end]`` array and slicing returns a list.
    """

    def __init__(self, owner: "ChameleonDataset", group: Any, count: int) -> None:
        self._owner = owner
        self._group = group
        self._count = count
        self._dtype = np.dtype(self._read(0).dtype)

    def _read(self, index: int) -> np.ndarray[Any, Any]:
        record = np.asarray(self._group[f"pinpoints_{index}"][:])
        start = np.asarray(record["start"]).reshape(-1)
        end = np.asarray(record["end"]).reshape(-1)
        return np.stack([start, end], axis=-1)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._count,)          # ragged trailing dim (per-trace op count)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        if isinstance(index, slice):
            return [self._read(i) for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        return self._read(index)


class _FrequencyView:
    """Present ``frequencies_{i}`` compound datasets as per-trace ``(n_changes, 2)``.

    DFS-only. ``metadata/frequencies/frequencies_{i}`` is a compound dataset with
    ``sample`` (int offset where the clock changes) and ``frequency`` (new MHz)
    fields, one record per change. ``[i]`` returns that trace's
    ``(n_changes, 2)`` ``[sample, frequency]`` float array (the sample index is
    exact in float64); slicing returns a list. Use it to resample a DFS trace
    back to a constant clock before attacking.
    """

    def __init__(self, owner: "ChameleonDataset", group: Any, count: int) -> None:
        self._owner = owner
        self._group = group
        self._count = count
        self._dtype = np.dtype(self._read(0).dtype)

    def _read(self, index: int) -> np.ndarray[Any, Any]:
        record = np.asarray(self._group[f"frequencies_{index}"][:])
        sample = np.asarray(record["sample"]).reshape(-1)
        frequency = np.asarray(record["frequency"]).reshape(-1)
        return np.stack([sample.astype(np.float64), frequency.astype(np.float64)], axis=-1)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._count,)          # ragged trailing dim (per-trace change count)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        if isinstance(index, slice):
            return [self._read(i) for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        return self._read(index)


class ChameleonDataset(SideChannelDataset):
    """Read one Chameleon chunk file lazily."""

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "chameleon",
        split: str | None = None,
        algorithm: str = "AES-128",
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        self._name = name.strip()
        self._split = split
        self._path = Path(path).expanduser()
        if not self._path.is_file():
            raise FileNotFoundError(f"Chameleon file not found: {self._path}")

        self._closed = False
        self._file: h5py.File | None = None
        try:
            self._file = h5py.File(self._path, "r")
            traces_group = self._file.get("data/traces")
            if traces_group is None:
                raise ValueError("Chameleon file has no 'data/traces' group.")
            count = len(traces_group)
            if count == 0 or "trace_0" not in traces_group:
                raise ValueError("Chameleon 'data/traces' group is empty.")

            self._traces = _PerTraceView(self, traces_group, count)
            ciphers = self._file.get("metadata/ciphers")
            self._keys = (
                _CipherView(self, ciphers, "key", "k", count)
                if ciphers is not None and "ciphers_0" in ciphers
                else None
            )
            self._plaintexts = (
                _CipherView(self, ciphers, "plaintexts", "p", count)
                if ciphers is not None and "ciphers_0" in ciphers
                else None
            )
            pinpoints = self._file.get("metadata/pinpoints")
            self._pinpoints = (
                _PinpointView(self, pinpoints, count)
                if pinpoints is not None and "pinpoints_0" in pinpoints
                else None
            )
            frequencies = self._file.get("metadata/frequencies")
            self._frequencies = (
                _FrequencyView(self, frequencies, count)
                if frequencies is not None and "frequencies_0" in frequencies
                else None
            )
            self._count = count
            self._metadata = MappingProxyType(
                {
                    "format": "hdf5",
                    "structure": "chameleon",
                    "algorithm": algorithm,
                    "path": str(self._path),
                }
            )
            validate_dataset(self)
        except Exception:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed or self._file is None:
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
        return self._plaintexts

    @property
    def ciphertexts(self) -> None:
        self._ensure_open()
        return None

    @property
    def keys(self) -> DatasetArray | None:
        self._ensure_open()
        return self._keys

    @property
    def masks(self) -> None:
        self._ensure_open()
        return None

    @property
    def labels(self) -> None:
        self._ensure_open()
        return None

    @property
    def pinpoints(self) -> DatasetArray | None:
        """Per-trace ``(n_ops, 2)`` [start, end] AES-execution offsets, or ``None``.

        Chameleon-specific (not part of the base field set): use these to crop
        each 134M-sample trace into per-operation windows for a standard attack,
        e.g. ``trace[start:end]`` for each row of ``pinpoints[i]``.
        """
        self._ensure_open()
        return self._pinpoints

    @property
    def frequencies(self) -> DatasetArray | None:
        """Per-trace ``(n_changes, 2)`` [sample, frequency-MHz] DFS clock schedule.

        Chameleon-specific and present only for the DFS sub-dataset (``None``
        otherwise): the operating frequency changes to ``frequency`` starting at
        ``sample``. Use it to resample a trace back to a constant clock.
        """
        self._ensure_open()
        return self._frequencies

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        self._ensure_open()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Chameleon indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Chameleon index out of range.")
        return TraceSample(
            trace=np.asarray(self._traces[index]),
            plaintext=None if self._plaintexts is None else np.asarray(self._plaintexts[index]),
            key=None if self._keys is None else np.asarray(self._keys[index]),
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ChameleonDataset"]
