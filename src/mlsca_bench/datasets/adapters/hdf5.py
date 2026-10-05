# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Reusable lazy adapter for compound-metadata HDF5 side-channel datasets.

Many public SCA datasets share the same on-disk shape: one HDF5 group per
split, a two-dimensional ``traces`` dataset, an optional compound ``metadata``
record whose columns hold the aligned plaintext/ciphertext/key/mask fields,
and (for some datasets) a separate top-level ``labels`` dataset. This adapter
captures that shape once and is configured per dataset through the registry;
:class:`~mlsca_bench.datasets.adapters.ascad.ASCADDataset` and
:class:`~mlsca_bench.datasets.adapters.ascon.AsconHDF5Dataset` are thin
subclasses that only pin the configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset
from ..errors import MissingDependencyError
from ._hdf5 import CompoundFieldView

try:
    import h5py
except ImportError as error:  # pragma: no cover - exercised without the extra.
    raise MissingDependencyError(
        "h5py", extra="hdf5", feature="Loading HDF5 side-channel datasets"
    ) from error


# Fields carried inside a compound ``metadata`` record, keyed by the public
# field name and mapped to the source column names that may hold them.
_ALIAS_FIELDS = ("plaintexts", "ciphertexts", "keys", "masks")


class HDF5CompoundDataset(SideChannelDataset):
    """Read one split of a compound-metadata HDF5 dataset lazily.

    The HDF5 file stays open until :meth:`close` is called (normally through
    the context-manager protocol). Bulk properties return lazy HDF5-backed
    arrays and never load the whole dataset into memory.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str,
        splits: Mapping[str, str],
        split: str,
        field_aliases: Mapping[str, Sequence[str]],
        traces_dataset: str = "traces",
        labels_dataset: str | None = None,
        metadata_dataset: str = "metadata",
        require_labels: bool = False,
        algorithm: str | None = None,
        label: str = "HDF5",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if not splits:
            raise ValueError("An HDF5 adapter must define at least one split.")
        if split not in splits:
            choices = ", ".join(sorted(splits))
            raise ValueError(f"{label} split must be one of: {choices}.")

        self._name = name.strip()
        self._split = split
        self._label = label
        self._labels_dataset_name = labels_dataset
        self._metadata_dataset_name = metadata_dataset
        self._field_aliases = {
            field: tuple(aliases) for field, aliases in field_aliases.items()
        }
        self._path = Path(path).expanduser()
        if not self._path.is_file():
            raise FileNotFoundError(f"{label} file not found: {self._path}")

        self._closed = False
        self._file: h5py.File | None = None
        self._label_dataset: h5py.Dataset | None = None
        self._field_views: dict[str, CompoundFieldView] = {}

        try:
            self._file = h5py.File(self._path, "r")
            group_name = splits[split]
            # ``get`` accepts nested paths ("Impl/Profiling") and the root ("/"),
            # so datasets that store traces at the top level are supported too.
            group = self._file.get(group_name)
            if group is None:
                # Some datasets nest every split under a single top-level device
                # group (e.g. AES-PTv2: /D1/Unprotected/Profiling). Prepend the
                # sole top-level group so the config stays device-agnostic.
                roots = list(self._file.keys())
                if len(roots) == 1:
                    group = self._file.get(f"{roots[0]}/{group_name}")
            if group is None:
                raise ValueError(f"{label} group {group_name!r} was not found.")
            if traces_dataset not in group:
                raise ValueError(
                    f"{label} group does not contain {traces_dataset!r}."
                )

            self._trace_dataset = group[traces_dataset]
            if self._trace_dataset.ndim != 2:
                raise ValueError(f"{label} traces must be a two-dimensional array.")

            if labels_dataset is not None:
                if labels_dataset not in group:
                    if require_labels:
                        raise ValueError(
                            f"{label} group does not contain {labels_dataset!r}."
                        )
                else:
                    self._label_dataset = group[labels_dataset]

            # A compound "metadata" record is read column-wise; but some formats
            # (e.g. eShard .ets) use a "metadata" *group* of per-field datasets,
            # which is handled through nested-path field aliases instead.
            records = group.get(self._metadata_dataset_name)
            self._records: h5py.Dataset | None = (
                records if isinstance(records, h5py.Dataset) else None
            )
            for field_name, value in (
                ("labels", self._label_dataset),
                ("metadata", self._records),
            ):
                if value is not None and len(value) != len(self._trace_dataset):
                    raise ValueError(
                        f"{label} traces and {field_name} are not aligned."
                    )

            record_fields = (
                tuple(self._records.dtype.names or ())
                if self._records is not None
                else ()
            )
            # Each field resolves either to a column of the compound "metadata"
            # record ("compound", <column>) or to a separate child dataset in
            # the group ("dataset", <h5py.Dataset>). Compound columns take
            # priority; dataset lookup supports nested paths (e.g. "MetaData/Key").
            self._field_sources: dict[str, tuple[str, Any]] = {}
            for public_name, aliases in self._field_aliases.items():
                column = next(
                    (alias for alias in aliases if alias in record_fields), None
                )
                if column is not None:
                    self._field_sources[public_name] = ("compound", column)
                    continue
                for alias in aliases:
                    child = group.get(alias)
                    if isinstance(child, h5py.Dataset):
                        if len(child) != len(self._trace_dataset):
                            raise ValueError(
                                f"{label} traces and {public_name} are not aligned."
                            )
                        self._field_sources[public_name] = ("dataset", child)
                        break
            used_columns = {
                reference
                for kind, reference in self._field_sources.values()
                if kind == "compound"
            }
            self._sample_metadata_fields = tuple(
                field for field in record_fields if field not in used_columns
            )

            dataset_metadata: dict[str, Any] = {
                "format": "hdf5",
                "path": str(self._path),
                "group": group_name,
            }
            if algorithm is not None:
                dataset_metadata["algorithm"] = algorithm
            if extra_metadata:
                dataset_metadata.update(extra_metadata)
            self._metadata = MappingProxyType(dataset_metadata)
            validate_dataset(self)
        except Exception:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed or self._file is None:
            raise RuntimeError(f"Dataset {self.name!r} is closed.")

    def _field(self, public_name: str) -> DatasetArray | None:
        self._ensure_open()
        source = self._field_sources.get(public_name)
        if source is None:
            return None
        kind, reference = source
        if kind == "dataset":
            return reference
        if self._records is None:
            return None
        if public_name not in self._field_views:
            self._field_views[public_name] = CompoundFieldView(
                self, self._records, reference
            )
        return self._field_views[public_name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str:
        return self._split

    @property
    def path(self) -> Path:
        return self._path

    @property
    def traces(self) -> h5py.Dataset:
        self._ensure_open()
        return self._trace_dataset

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
        self._ensure_open()
        return self._label_dataset

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
        return len(self._trace_dataset)

    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        self._ensure_open()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"{self._label} indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"{self._label} index out of range.")

        record = self._records[index] if self._records is not None else None

        def read(public_name: str) -> np.ndarray[Any, Any] | None:
            source = self._field_sources.get(public_name)
            if source is None:
                return None
            kind, reference = source
            if kind == "dataset":
                return np.asarray(reference[index])
            if record is None:
                return None
            return np.asarray(record[reference])

        if record is None:
            sample_metadata: Mapping[str, Any] = MappingProxyType({})
        elif self._records is not None and self._records.dtype.names is None:
            sample_metadata = MappingProxyType({"raw_metadata": np.asarray(record)})
        else:
            sample_metadata = MappingProxyType(
                {
                    field: np.asarray(record[field])
                    for field in self._sample_metadata_fields
                }
            )

        label = (
            None
            if self._label_dataset is None
            else np.asarray(self._label_dataset[index])
        )
        return TraceSample(
            trace=np.asarray(self._trace_dataset[index]),
            plaintext=read("plaintexts"),
            ciphertext=read("ciphertexts"),
            key=read("keys"),
            mask=read("masks"),
            label=label,
            metadata=sample_metadata,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _ColumnView:
    """Lazy view of a column range of a 2-D dataset (e.g. bytes 0-15 = plaintext)."""

    def __init__(self, source: Any, start: int, stop: int) -> None:
        self._source, self._start, self._stop = source, start, stop

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._source), self._stop - self._start)

    @property
    def dtype(self) -> Any:
        return self._source.dtype

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: Any) -> Any:
        return np.asarray(self._source[index])[..., self._start:self._stop]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        values = self[:]
        return values if dtype is None else values.astype(dtype)


class FlatHDF5Dataset(SideChannelDataset):
    """HDF5 where each split is a top-level 2-D dataset (no per-split group).

    Used by datasets such as the Curve25519 ECC release, which store
    ``profiling_traces``/``attacking_traces`` as top-level datasets alongside
    sibling ``*_data`` label datasets. ``splits`` maps a split name to its
    traces dataset; ``labels`` optionally maps a split to its label dataset.
    ``data`` maps a split to a 2-D byte dataset whose column ranges hold the
    plaintexts, ciphertexts, keys or masks, given by ``data_columns`` as
    ``{"plaintexts": [0, 16], ...}`` (CHES CTF 2018: plaintext | ciphertext | key).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str,
        splits: Mapping[str, str],
        split: str,
        labels: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        data_columns: Mapping[str, Sequence[int]] | None = None,
        algorithm: str | None = None,
        label: str = "HDF5",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if split not in splits:
            choices = ", ".join(sorted(splits))
            raise ValueError(f"{label} split must be one of: {choices}.")

        self._name = name.strip()
        self._split = split
        self._path = Path(path).expanduser()
        if not self._path.is_file():
            raise FileNotFoundError(f"{label} file not found: {self._path}")

        self._closed = False
        self._file: h5py.File | None = None
        self._label_dataset: h5py.Dataset | None = None
        self._columns: dict[str, _ColumnView] = {}
        try:
            self._file = h5py.File(self._path, "r")
            traces = self._file.get(splits[split])
            if not isinstance(traces, h5py.Dataset) or traces.ndim != 2:
                raise ValueError(
                    f"{label} split {splits[split]!r} is not a 2-D dataset."
                )
            self._trace_dataset = traces

            if labels is not None and split in labels:
                label_ds = self._file.get(labels[split])
                if isinstance(label_ds, h5py.Dataset):
                    if len(label_ds) != len(traces):
                        raise ValueError(f"{label} traces and labels are not aligned.")
                    self._label_dataset = label_ds

            if data is not None and split in data and data_columns:
                data_ds = self._file.get(data[split])
                if not isinstance(data_ds, h5py.Dataset) or data_ds.ndim != 2:
                    raise ValueError(f"{label} data {data[split]!r} is not a 2-D dataset.")
                if len(data_ds) != len(traces):
                    raise ValueError(f"{label} traces and {data[split]!r} are not aligned.")
                for field, (start, stop) in data_columns.items():
                    if field not in ("plaintexts", "ciphertexts", "keys", "masks"):
                        raise ValueError(f"{label} data_columns: unknown field {field!r}.")
                    if not 0 <= start < stop <= data_ds.shape[1]:
                        raise ValueError(f"{label} data_columns[{field!r}] is outside the {data_ds.shape[1]} columns.")
                    self._columns[field] = _ColumnView(data_ds, int(start), int(stop))

            dataset_metadata: dict[str, Any] = {
                "format": "hdf5",
                "path": str(self._path),
                "dataset": splits[split],
            }
            if algorithm is not None:
                dataset_metadata["algorithm"] = algorithm
            if extra_metadata:
                dataset_metadata.update(extra_metadata)
            self._metadata = MappingProxyType(dataset_metadata)
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
    def split(self) -> str:
        return self._split

    @property
    def traces(self) -> h5py.Dataset:
        self._ensure_open()
        return self._trace_dataset

    @property
    def plaintexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._columns.get("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._columns.get("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        self._ensure_open()
        return self._columns.get("keys")

    @property
    def masks(self) -> DatasetArray | None:
        self._ensure_open()
        return self._columns.get("masks")

    @property
    def labels(self) -> DatasetArray | None:
        self._ensure_open()
        return self._label_dataset

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
        return len(self._trace_dataset)

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
        columns = {
            {"plaintexts": "plaintext", "ciphertexts": "ciphertext", "keys": "key", "masks": "mask"}[field]: view[index]
            for field, view in self._columns.items()
        }
        return TraceSample(
            trace=np.asarray(self._trace_dataset[index]),
            label=None
            if self._label_dataset is None
            else np.asarray(self._label_dataset[index]),
            **columns,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["FlatHDF5Dataset", "HDF5CompoundDataset"]
