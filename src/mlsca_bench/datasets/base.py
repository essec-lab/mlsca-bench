# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Stable, format-independent interfaces for side-channel datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike, NDArray


Scalar = TypeVar("Scalar", bound=np.generic)
FIELD_NAMES = (
    "traces",
    "plaintexts",
    "ciphertexts",
    "keys",
    "masks",
    "labels",
)


@runtime_checkable
class DatasetArray(Protocol):
    """Minimal array interface exposed by dataset fields.

    NumPy arrays, memory maps, and HDF5 datasets can implement this protocol,
    allowing callers to slice large datasets without loading them completely.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> DTypeLike: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: Any) -> Any: ...


@dataclass(frozen=True)
class TraceSample:
    """One trace and the information aligned with that trace."""

    trace: NDArray[np.generic]
    plaintext: NDArray[np.generic] | None = None
    ciphertext: NDArray[np.generic] | None = None
    key: NDArray[np.generic] | None = None
    mask: NDArray[np.generic] | None = None
    label: NDArray[np.generic] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SideChannelDataset(ABC):
    """Common contract implemented by every loaded SCA dataset.

    Every present bulk field has the number of traces as its first dimension.
    Optional fields return ``None`` when the source dataset does not provide
    them. Field values are array-like and may be backed by lazy storage.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical registry name."""

    @property
    @abstractmethod
    def split(self) -> str | None:
        """Active split, normally ``profiling``, ``attack``, or ``validation``."""

    @property
    @abstractmethod
    def traces(self) -> DatasetArray:
        """Trace matrix with shape ``(number_of_traces, trace_length)``."""

    @property
    @abstractmethod
    def plaintexts(self) -> DatasetArray | None:
        """Plaintexts aligned with traces, or ``None`` when unavailable."""

    @property
    @abstractmethod
    def ciphertexts(self) -> DatasetArray | None:
        """Ciphertexts aligned with traces, or ``None`` when unavailable."""

    @property
    @abstractmethod
    def keys(self) -> DatasetArray | None:
        """Keys aligned with traces, or ``None`` when unavailable.

        A fixed key is represented as a broadcast view whose first dimension
        still equals the number of traces; it need not be copied in memory.
        """

    @property
    @abstractmethod
    def masks(self) -> DatasetArray | None:
        """Masks aligned with traces, or ``None`` when unavailable."""

    @property
    @abstractmethod
    def labels(self) -> DatasetArray | None:
        """Source-provided ML labels, or ``None`` when unavailable."""

    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, Any]:
        """Dataset-level metadata shared by all samples."""

    @property
    def available_fields(self) -> tuple[str, ...]:
        """Names of bulk fields supplied by this dataset."""

        return tuple(
            name for name in FIELD_NAMES if getattr(self, name) is not None
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the trace matrix."""

        return tuple(self.traces.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Data type of trace samples."""

        return np.dtype(self.traces.dtype)

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether the dataset can no longer be read."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Calling this method repeatedly is safe."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of traces in the active split."""

    @overload
    def __getitem__(self, index: int) -> TraceSample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TraceSample]: ...

    @abstractmethod
    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        """Read one sample or a sequence of samples."""

    def __enter__(self) -> SideChannelDataset:
        if self.closed:
            raise RuntimeError("Cannot reopen a closed dataset.")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class ArraySideChannelDataset(SideChannelDataset):
    """Dataset backed by aligned NumPy arrays."""

    def __init__(
        self,
        *,
        name: str,
        traces: NDArray[np.generic],
        split: str | None = None,
        plaintexts: NDArray[np.generic] | None = None,
        ciphertexts: NDArray[np.generic] | None = None,
        keys: NDArray[np.generic] | None = None,
        masks: NDArray[np.generic] | None = None,
        labels: NDArray[np.generic] | None = None,
        metadata: Mapping[str, Any] | None = None,
        sample_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        trace_array = np.asarray(traces)
        if trace_array.ndim != 2:
            raise ValueError(
                "traces must have shape (number_of_traces, trace_length)."
            )

        self._name = name.strip()
        self._split = split
        self._traces = trace_array
        self._plaintexts = self._validate_field("plaintexts", plaintexts)
        self._ciphertexts = self._validate_field("ciphertexts", ciphertexts)
        self._keys = self._validate_field("keys", keys)
        self._masks = self._validate_field("masks", masks)
        self._labels = self._validate_field("labels", labels)
        self._metadata = MappingProxyType(dict(metadata or {}))
        self._closed = False

        if sample_metadata is None:
            self._sample_metadata = tuple(
                MappingProxyType({}) for _ in range(len(self))
            )
        else:
            if len(sample_metadata) != len(self):
                raise ValueError(
                    "sample_metadata must contain one entry per trace: "
                    f"expected {len(self)}, got {len(sample_metadata)}."
                )
            if not all(isinstance(item, Mapping) for item in sample_metadata):
                raise TypeError("Each sample_metadata entry must be a mapping.")
            self._sample_metadata = tuple(
                MappingProxyType(dict(item)) for item in sample_metadata
            )

    def _validate_field(
        self, name: str, value: NDArray[np.generic] | None
    ) -> NDArray[np.generic] | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"{name} must have a trace dimension.")
        if len(array) != len(self._traces):
            raise ValueError(
                f"{name} must contain one entry per trace: "
                f"expected {len(self._traces)}, got {len(array)}."
            )
        return array

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self.name!r} is closed.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> NDArray[np.generic]:
        self._ensure_open()
        return self._traces

    @property
    def plaintexts(self) -> NDArray[np.generic] | None:
        self._ensure_open()
        return self._plaintexts

    @property
    def ciphertexts(self) -> NDArray[np.generic] | None:
        self._ensure_open()
        return self._ciphertexts

    @property
    def keys(self) -> NDArray[np.generic] | None:
        self._ensure_open()
        return self._keys

    @property
    def masks(self) -> NDArray[np.generic] | None:
        self._ensure_open()
        return self._masks

    @property
    def labels(self) -> NDArray[np.generic] | None:
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
            raise TypeError("Dataset indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range.")

        return TraceSample(
            trace=np.asarray(self._traces[index]),
            plaintext=self._sample(self._plaintexts, index),
            ciphertext=self._sample(self._ciphertexts, index),
            key=self._sample(self._keys, index),
            mask=self._sample(self._masks, index),
            label=self._sample(self._labels, index),
            metadata=self._sample_metadata[index],
        )

    @staticmethod
    def _sample(
        field: NDArray[np.generic] | None, index: int
    ) -> NDArray[np.generic] | None:
        return None if field is None else np.asarray(field[index])


def broadcast_field(
    value: NDArray[Scalar] | Sequence[Any], number_of_traces: int
) -> NDArray[Scalar]:
    """Return a zero-copy, read-only per-trace view of a constant field."""

    if not isinstance(number_of_traces, int) or number_of_traces < 0:
        raise ValueError("number_of_traces must be a non-negative integer.")
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("A constant field must have at least one dimension.")
    return np.broadcast_to(array, (number_of_traces, *array.shape))


def validate_dataset(dataset: SideChannelDataset) -> None:
    """Validate the format-independent invariants of a loaded dataset.

    This deliberately inspects only field metadata and the leading dimension;
    it does not materialize lazy arrays.
    """

    if dataset.closed:
        raise RuntimeError("Cannot validate a closed dataset.")
    if not isinstance(dataset.name, str) or not dataset.name.strip():
        raise ValueError("Dataset name must be a non-empty string.")
    if dataset.split is not None and (
        not isinstance(dataset.split, str) or not dataset.split.strip()
    ):
        raise ValueError("Dataset split must be a non-empty string or None.")
    if not isinstance(dataset.metadata, Mapping):
        raise TypeError("Dataset metadata must be a mapping.")

    traces = dataset.traces
    if len(traces.shape) != 2:
        raise ValueError(
            "traces must have shape (number_of_traces, trace_length)."
        )
    if len(traces) != len(dataset):
        raise ValueError(
            "traces and dataset length are not aligned: "
            f"expected {len(dataset)}, got {len(traces)}."
        )

    for field_name in FIELD_NAMES[1:]:
        value = getattr(dataset, field_name)
        if value is None:
            continue
        if len(value.shape) == 0:
            raise ValueError(f"{field_name} must have a trace dimension.")
        if len(value) != len(dataset):
            raise ValueError(
                f"{field_name} must contain one entry per trace: "
                f"expected {len(dataset)}, got {len(value)}."
            )


__all__ = [
    "ArraySideChannelDataset",
    "DatasetArray",
    "FIELD_NAMES",
    "SideChannelDataset",
    "TraceSample",
    "broadcast_field",
    "validate_dataset",
]
