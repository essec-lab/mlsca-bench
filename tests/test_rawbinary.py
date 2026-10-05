# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import RawBinaryDataset


@pytest.fixture
def present_like_directory(tmp_path: Path) -> Path:
    traces = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    # Prepend an 8-byte header to exercise the offset handling.
    with open(tmp_path / "traces.dat", "wb") as handle:
        handle.write(b"\x00" * 8)
        handle.write(traces.tobytes())
    np.arange(4 * 8, dtype=np.uint8).reshape(4, 8).tofile(tmp_path / "labels.dat")
    return tmp_path


def test_raw_binary_reads_with_offset(present_like_directory: Path) -> None:
    with RawBinaryDataset(
        present_like_directory,
        name="present",
        layout={
            "traces": {
                "filename": "traces.dat",
                "dtype": "float32",
                "item_length": 6,
                "offset": 8,
            },
            "labels": {
                "filename": "labels.dat",
                "dtype": "uint8",
                "item_length": 8,
            },
        },
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (4, 6)
        assert dataset.available_fields == ("traces", "labels")
        assert isinstance(dataset.traces, np.memmap)
        np.testing.assert_array_equal(
            dataset.traces[1], np.arange(6, 12, dtype=np.float32)
        )
        np.testing.assert_array_equal(dataset[3].label, np.arange(24, 32))


def test_raw_binary_rejects_ragged_file(tmp_path: Path) -> None:
    np.arange(6, dtype=np.float32).tofile(tmp_path / "traces.dat")
    with pytest.raises(ValueError, match="whole number of traces records"):
        RawBinaryDataset(
            tmp_path,
            name="present",
            layout={
                "traces": {"filename": "traces.dat", "dtype": "float32", "item_length": 4}
            },
        )


def test_raw_binary_record_mode_interleaved(tmp_path: Path) -> None:
    # One record = 5 int8 samples followed by 1 uint8 group byte (DL-LA shape).
    record_dtype = np.dtype([("trace", "i1", (5,)), ("group", "u1")])
    records = np.zeros(4, dtype=record_dtype)
    for index in range(4):
        records[index]["trace"] = np.arange(5) + index
        records[index]["group"] = index % 2
    records.tofile(tmp_path / "Traces_1.dat")

    with RawBinaryDataset(
        tmp_path,
        name="present",
        record={
            "filename": "Traces_1.dat",
            "dtype": [["trace", "i1", [5]], ["group", "u1"]],
            "fields": {"traces": "trace", "labels": "group"},
        },
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (4, 5)
        assert dataset.available_fields == ("traces", "labels")
        np.testing.assert_array_equal(dataset.traces[2], np.arange(5) + 2)
        assert dataset[3].label.item() == 1


def test_raw_binary_requires_exactly_one_of_layout_or_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        RawBinaryDataset(tmp_path, name="x")


def test_raw_binary_rejects_misaligned_fields(tmp_path: Path) -> None:
    np.arange(4 * 6, dtype=np.float32).reshape(4, 6).tofile(tmp_path / "traces.dat")
    np.arange(3 * 8, dtype=np.uint8).reshape(3, 8).tofile(tmp_path / "labels.dat")
    with pytest.raises(ValueError, match="one entry per trace"):
        RawBinaryDataset(
            tmp_path,
            name="present",
            layout={
                "traces": {"filename": "traces.dat", "dtype": "float32", "item_length": 6},
                "labels": {"filename": "labels.dat", "dtype": "uint8", "item_length": 8},
            },
        )
