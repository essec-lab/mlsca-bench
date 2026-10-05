# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import ConcatDataset
from mlsca_bench.datasets.adapters.scaaml import _parse_example, open_scaaml
from mlsca_bench.datasets.registry import DatasetSpec


# --- minimal protobuf/TFRecord encoder (mirrors what SCAAML writes) --------

def _uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _len_delim(field: int, payload: bytes) -> bytes:
    return _uvarint((field << 3) | 2) + _uvarint(len(payload)) + payload


def _float_feature(values: np.ndarray) -> bytes:
    packed = np.asarray(values, dtype="<f4").tobytes()
    return _len_delim(2, _len_delim(1, packed))  # Feature.FloatList.value[packed]


def _int_feature(values: np.ndarray) -> bytes:
    packed = b"".join(_uvarint(int(v)) for v in values)
    return _len_delim(3, _len_delim(1, packed))  # Feature.Int64List.value[packed]


def _example(features: dict[str, bytes]) -> bytes:
    entries = b"".join(
        _len_delim(1, _len_delim(1, name.encode()) + _len_delim(2, feat))
        for name, feat in features.items()
    )
    return _len_delim(1, entries)  # Example.features


def _tfrecord_frame(payload: bytes) -> bytes:
    return struct.pack("<Q", len(payload)) + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00"


def _write_shard(path: Path, traces: np.ndarray, k: np.ndarray, k_bits: np.ndarray) -> None:
    frames = b"".join(
        _tfrecord_frame(
            _example(
                {
                    "trace1": _float_feature(traces[i]),
                    "k": _int_feature(k[i]),
                    "k_bits": _int_feature(k_bits[i]),
                }
            )
        )
        for i in range(len(traces))
    )
    path.write_bytes(gzip.compress(frames))


@pytest.fixture
def scaaml_dir(tmp_path: Path) -> Path:
    (tmp_path / "train").mkdir()
    shards = []
    for shard_index, base in enumerate([0, 100]):
        traces = (np.arange(2 * 3, dtype=np.float32).reshape(2, 3) + base)
        k = (np.arange(2 * 2, dtype=np.int64).reshape(2, 2) + base) % 256
        k_bits = (np.arange(2 * 4, dtype=np.int64).reshape(2, 4)) % 2
        rel = f"train/{shard_index}_deadbeef_0.tfrec"
        _write_shard(tmp_path / rel, traces, k, k_bits)
        shards.append({"path": rel, "examples": 2})
    info = {
        "compression": "GZIP",
        "measurement_dtype": "float32",
        "measurements_info": {"trace1": {"type": "power", "len": 3}},
        "attack_points_info": {"k": {"len": 2, "max_val": 256}, "k_bits": {"len": 4, "max_val": 2}},
        "shards_list": {"train": shards},
    }
    (tmp_path / "info.json").write_text(json.dumps(info))
    return tmp_path


def test_scaaml_reads_sharded_tfrecords(scaaml_dir: Path) -> None:
    dataset = open_scaaml(
        scaaml_dir,
        name="scaaml-cm0",
        traces_feature="trace1",
        field_map={"keys": "k", "labels": "k_bits"},
        split="train",
    )
    with dataset:
        validate_dataset(dataset)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 4  # 2 shards x 2 examples
        assert dataset.shape == (4, 3)
        assert dataset.available_fields == ("traces", "keys", "labels")
        assert dataset.keys.dtype == np.uint8  # max_val 256 -> uint8
        # Global index 2 is the first example of shard 2 (base 100).
        np.testing.assert_array_equal(dataset[2].trace, np.arange(3, dtype=np.float32) + 100)
        np.testing.assert_array_equal(dataset[2].key, (np.arange(2) + 100).astype(np.uint8))
        np.testing.assert_array_equal(dataset[0].label, [0, 1, 0, 1])


def test_scaaml_public_loader(monkeypatch: pytest.MonkeyPatch, scaaml_dir: Path) -> None:
    spec = DatasetSpec(
        name="scaaml-ecc-gpam-cm0",
        display_name="SCAAML CM0",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="scaaml",
        adapter_config={
            "traces_feature": "trace1",
            "field_map": {"keys": "k", "labels": "k_bits"},
            "default_split": "train",
        },
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)
    with loading.load_dataset("scaaml-ecc-gpam-cm0", path=scaaml_dir) as dataset:
        assert len(dataset) == 4
        assert dataset.name == "scaaml-ecc-gpam-cm0"


def test_parse_example_roundtrip() -> None:
    payload = _example({"m": _float_feature(np.array([1.5, -2.0])), "k": _int_feature(np.array([7, 255]))})
    parsed = _parse_example(payload)
    np.testing.assert_array_equal(parsed["m"], np.array([1.5, -2.0], dtype=np.float32))
    np.testing.assert_array_equal(parsed["k"], np.array([7, 255], dtype=np.int64))
