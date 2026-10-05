# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for SCAAML datasets (sharded TFRecord + info.json).

SCAAML stores traces as ``tf.train.Example`` records in ``.tfrec`` shards under
``train``/``test``/``holdout``, described by an ``info.json``. This adapter
reads that format WITHOUT TensorFlow: it decodes the TFRecord framing, the
(optionally GZIP-compressed) stream, and the Example protobuf directly. Each
shard becomes a lazily-parsed dataset, and shards are concatenated into one
split; shapes come from ``info.json`` so nothing is read until a trace is.
"""

from __future__ import annotations

import gzip
import json
import os
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset
from ._concat import ConcatDataset
from ..errors import describe_paths


# --- dependency-free TFRecord + tf.train.Example reader -------------------

def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes, start: int, end: int) -> Iterator[tuple[int, int, Any]]:
    pos = start
    while pos < end:
        tag, pos = _varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value = buf[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}.")
        yield field, wire, value


def _parse_feature(buf: bytes) -> Any:
    for field, wire, value in _iter_fields(buf, 0, len(buf)):
        if field == 1 and wire == 2:  # BytesList { repeated bytes value = 1; }
            return [v for f, w, v in _iter_fields(value, 0, len(value)) if f == 1 and w == 2]
        if field == 2 and wire == 2:  # FloatList { repeated float value = 1 [packed]; }
            floats = [
                np.frombuffer(v, dtype="<f4")
                for f, w, v in _iter_fields(value, 0, len(value))
                if f == 1 and w in (2, 5)
            ]
            return np.concatenate(floats) if floats else np.array([], np.float32)
        if field == 3 and wire == 2:  # Int64List { repeated int64 value = 1 [packed]; }
            ints: list[int] = []
            for f, w, v in _iter_fields(value, 0, len(value)):
                if f == 1 and w == 2:
                    pos = 0
                    while pos < len(v):
                        item, pos = _varint(v, pos)
                        ints.append(item)
                elif f == 1 and w == 0:
                    ints.append(v)
            return np.array(ints, dtype=np.int64)
    return None


def _parse_example(buf: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, wire, value in _iter_fields(buf, 0, len(buf)):
        if field != 1 or wire != 2:  # Example { Features features = 1; }
            continue
        for f, w, entry in _iter_fields(value, 0, len(value)):  # Features.feature map
            if f != 1 or w != 2:
                continue
            key = feature = None
            for ff, ww, vv in _iter_fields(entry, 0, len(entry)):
                if ff == 1 and ww == 2:
                    key = vv.decode("utf-8")
                elif ff == 2 and ww == 2:
                    feature = vv
            if key is not None and feature is not None:
                out[key] = _parse_feature(feature)
    return out


def _iter_tfrecords(data: bytes) -> Iterator[bytes]:
    pos, total = 0, len(data)
    while pos + 12 <= total:
        length = int.from_bytes(data[pos : pos + 8], "little")
        pos += 8 + 4  # length + its CRC (CRC not verified)
        if pos + length + 4 > total:
            break  # truncated tail
        yield data[pos : pos + length]
        pos += length + 4


# --- adapter --------------------------------------------------------------


class _ShardFieldView:
    """Lazy (n, width) view of one feature within a shard; parses on access."""

    def __init__(
        self, owner: "_ScaamlShard", feature: str, shape: tuple[int, int], dtype: np.dtype[Any]
    ) -> None:
        self._owner = owner
        self._feature = feature
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        return self._owner._decode()[self._feature][index]


class _ScaamlShard(SideChannelDataset):
    """One SCAAML ``.tfrec`` shard, parsed lazily into aligned field arrays."""

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        count: int,
        traces_feature: str,
        trace_length: int,
        measurement_dtype: str,
        field_map: Mapping[str, str],
        ap_specs: Mapping[str, tuple[int, int]],
        compression: str | None,
    ) -> None:
        self._path = path
        self._name = name
        self._split: str | None = None
        self._count = count
        self._traces_feature = traces_feature
        self._field_map = dict(field_map)
        self._ap_specs = dict(ap_specs)
        self._compression = compression
        self._closed = False
        self._decoded: dict[str, np.ndarray[Any, Any]] | None = None

        trace_dtype = np.dtype(measurement_dtype)
        self._views: dict[str, _ShardFieldView] = {
            "traces": _ShardFieldView(self, traces_feature, (count, trace_length), trace_dtype)
        }
        for public_name, source in self._field_map.items():
            if source == traces_feature:
                self._views[public_name] = _ShardFieldView(
                    self, source, (count, trace_length), trace_dtype
                )
            else:
                width, max_val = self._ap_specs[source]
                dtype = np.dtype(np.uint8 if max_val <= 256 else np.int64)
                self._views[public_name] = _ShardFieldView(self, source, (count, width), dtype)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self._name!r} is closed.")

    def _feature_dtype(self, feature: str) -> np.dtype[Any]:
        if feature == self._traces_feature:
            return self._views["traces"].dtype
        _width, max_val = self._ap_specs[feature]
        return np.dtype(np.uint8 if max_val <= 256 else np.int64)

    def _decode(self) -> dict[str, np.ndarray[Any, Any]]:
        if self._decoded is None:
            raw = self._path.read_bytes()
            if self._compression == "GZIP":
                raw = gzip.decompress(raw)
            needed = {self._traces_feature, *self._field_map.values()}
            decoded: dict[str, np.ndarray[Any, Any]] = {}
            for row, payload in enumerate(_iter_tfrecords(raw)):
                example = _parse_example(payload)
                for feature in needed:
                    value = example[feature]
                    if feature not in decoded:
                        decoded[feature] = np.empty(
                            (self._count, value.shape[0]), dtype=self._feature_dtype(feature)
                        )
                    decoded[feature][row] = value
            self._decoded = decoded
        return self._decoded

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> DatasetArray:
        self._ensure_open()
        return self._views["traces"]

    @property
    def plaintexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._views.get("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        self._ensure_open()
        return self._views.get("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        self._ensure_open()
        return self._views.get("keys")

    @property
    def masks(self) -> DatasetArray | None:
        self._ensure_open()
        return self._views.get("masks")

    @property
    def labels(self) -> DatasetArray | None:
        self._ensure_open()
        return self._views.get("labels")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType({"format": "scaaml-tfrecord", "path": str(self._path)})

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._decoded = None
        self._closed = True

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        self._ensure_open()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("Dataset index out of range.")

        def read(field: str) -> np.ndarray[Any, Any] | None:
            view = self._views.get(field)
            return None if view is None else np.asarray(view[index])

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


def _find_info(source: str | os.PathLike[str] | Iterable[Path]) -> Path:
    if isinstance(source, (str, os.PathLike)):
        roots: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        roots = tuple(Path(path).expanduser() for path in source)
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "info.json":
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob("info.json"))
    if not found:
        raise FileNotFoundError(f"No SCAAML info.json was found in {describe_paths(roots)}.")
    if len(found) > 1:
        raise RuntimeError("Multiple info.json files were found; point at one dataset dir.")
    return found[0]


def open_scaaml(
    source: str | os.PathLike[str] | Iterable[Path],
    *,
    name: str,
    traces_feature: str,
    field_map: Mapping[str, str],
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SideChannelDataset:
    """Build a concatenated SCAAML split from its sharded TFRecord files."""

    info_path = _find_info(source)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    base = info_path.parent
    shards_list = info["shards_list"]
    active_split = split or "train"
    if active_split not in shards_list:
        choices = ", ".join(sorted(shards_list))
        raise ValueError(f"SCAAML split must be one of: {choices}.")

    measurements = info["measurements_info"]
    if traces_feature not in measurements:
        raise KeyError(f"Measurement {traces_feature!r} is not in info.json.")
    trace_length = int(measurements[traces_feature]["len"])
    measurement_dtype = info.get("measurement_dtype", "float32")
    if measurement_dtype != "float32":
        raise ValueError(
            f"Only float32 SCAAML measurements are supported, got {measurement_dtype!r}."
        )
    ap_specs = {
        ap_name: (int(spec["len"]), int(spec["max_val"]))
        for ap_name, spec in info["attack_points_info"].items()
    }
    compression = info.get("compression")

    # A partial download (files= / split=) holds only some shards: use those.
    entries = list(shards_list[active_split])
    present = [entry for entry in entries if (base / entry["path"]).is_file()]
    if entries and not present:
        raise FileNotFoundError(
            f"None of the {len(entries)} {active_split} shards of {name} is under {base} "
            f"(e.g. {entries[0]['path']}). Download them with split={active_split!r} or files=."
        )
    shards = [
        _ScaamlShard(
            base / entry["path"],
            name=name,
            count=int(entry["examples"]),
            traces_feature=traces_feature,
            trace_length=trace_length,
            measurement_dtype=measurement_dtype,
            field_map=field_map,
            ap_specs=ap_specs,
            compression=compression,
        )
        for entry in present
    ]
    if not shards:
        raise ValueError(f"SCAAML split {active_split!r} has no shards.")
    if len(present) < len(entries):
        warnings.warn(
            f"{name}: loaded {len(present)} of the {len(entries)} {active_split} shards; the others "
            "are not downloaded (download them with files= or split=).",
            stacklevel=3,
        )
    dataset = ConcatDataset(shards, name=name, split=active_split)
    validate_dataset(dataset)
    return dataset


__all__ = ["open_scaaml"]
