# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import ConcatDataset, TRSDataset
from mlsca_bench.datasets.registry import DatasetSpec


def _write_trs(
    path: Path,
    *,
    traces: np.ndarray,
    data: np.ndarray,
    title_length: int = 0,
) -> None:
    """Write a minimal Riscure .trs file (float32 samples, uint8 data)."""

    number_of_traces, number_of_samples = traces.shape
    data_length = data.shape[1] if data.size else 0

    def tlv(tag: int, payload: bytes) -> bytes:
        assert len(payload) < 0x80
        return bytes([tag, len(payload)]) + payload

    header = b"".join(
        (
            tlv(0x41, struct.pack("<I", number_of_traces)),
            tlv(0x42, struct.pack("<I", number_of_samples)),
            tlv(0x43, bytes([0x14])),  # coding: float (0x10) | 4 bytes
            tlv(0x44, struct.pack("<H", data_length)),
            tlv(0x45, bytes([title_length])),
            bytes([0x5F, 0x00]),  # trace block
        )
    )
    with open(path, "wb") as handle:
        handle.write(header)
        for index in range(number_of_traces):
            handle.write(b"\x00" * title_length)
            if data_length:
                handle.write(data[index].astype(np.uint8).tobytes())
            handle.write(traces[index].astype(np.float32).tobytes())


@pytest.fixture
def trs_file(tmp_path: Path) -> Path:
    path = tmp_path / "traces.trs"
    traces = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    data = np.zeros((3, 32), dtype=np.uint8)
    for index in range(3):
        data[index, 0:16] = np.arange(16) + index  # plaintext
        data[index, 16:32] = np.arange(16) + 100 + index  # ciphertext
    _write_trs(path, traces=traces, data=data)
    return path


def test_trs_reads_samples_and_data_fields(trs_file: Path) -> None:
    with TRSDataset(
        trs_file,
        name="ches-c2",
        data_fields={"plaintexts": [0, 16], "ciphertexts": [16, 32]},
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 5)
        assert dataset.available_fields == ("traces", "plaintexts", "ciphertexts")
        assert dataset.metadata["format"] == "trs"
        np.testing.assert_array_equal(
            dataset.traces[1], np.arange(5, 10, dtype=np.float32)
        )
        sample = dataset[2]
        np.testing.assert_array_equal(sample.plaintext, np.arange(16) + 2)
        np.testing.assert_array_equal(sample.ciphertext, np.arange(16) + 102)


def test_trs_without_data_fields_exposes_only_traces(trs_file: Path) -> None:
    with TRSDataset(trs_file, name="ches-c2") as dataset:
        assert dataset.available_fields == ("traces",)
        assert dataset.plaintexts is None


def test_trs_rejects_out_of_range_data_field(trs_file: Path) -> None:
    with pytest.raises(ValueError, match="outside the 32-byte"):
        TRSDataset(trs_file, name="ches-c2", data_fields={"keys": [16, 48]})


def test_trs_label_data_field(tmp_path: Path) -> None:
    path = tmp_path / "reassure.trs"
    traces = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    data = np.array([[0], [1], [0]], dtype=np.uint8)  # 1-byte ladder swap bit
    _write_trs(path, traces=traces, data=data)
    with TRSDataset(path, name="reassure", data_fields={"labels": [0, 1]}) as dataset:
        validate_dataset(dataset)
        assert dataset.available_fields == ("traces", "labels")
        assert dataset[1].label[0] == 1


def test_trs_preferred_filename(tmp_path: Path) -> None:
    for name, base in (("a.trs", 0.0), ("b.trs", 100.0)):
        _write_trs(
            tmp_path / name,
            traces=np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + base,
            data=np.zeros((2, 0), dtype=np.uint8),
        )
    with pytest.raises(RuntimeError, match="preferred_filename"):
        TRSDataset(tmp_path, name="x")
    with TRSDataset(tmp_path, name="x", preferred_filename="b.trs") as dataset:
        assert dataset.traces[0, 0] == 100.0


def _spec_ches_c2() -> DatasetSpec:
    return DatasetSpec(
        name="ches-ctf-2018-challenge-2",
        display_name="CHES CTF 2018 C2",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="trs",
        adapter_config={
            "chunked": True,
            "chunk_glob": "*.trs",
            "data_fields": {"plaintexts": [0, 16], "ciphertexts": [16, 32], "keys": [32, 48]},
        },
    )


def test_chunked_trs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for index, base in ((1, 0), (2, 50)):
        traces = np.arange(2 * 5, dtype=np.float32).reshape(2, 5) + base
        data = np.zeros((2, 48), dtype=np.uint8)
        for row in range(2):
            data[row, 0:16] = np.arange(16) + base + row  # plaintext
            data[row, 16:32] = np.arange(16) + 200  # ciphertext
            data[row, 32:48] = np.arange(16)  # key
        _write_trs(tmp_path / f"PinataAcqTask2.{index}_upload.trs", traces=traces, data=data)
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_ches_c2())

    with loading.load_dataset("ches-ctf-2018-challenge-2", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 4
        assert dataset.available_fields == ("traces", "plaintexts", "ciphertexts", "keys")
        np.testing.assert_array_equal(dataset[2].trace, np.arange(5, dtype=np.float32) + 50)
        np.testing.assert_array_equal(dataset[2].plaintext, (np.arange(16) + 50).astype(np.uint8))
        np.testing.assert_array_equal(dataset[2].key, np.arange(16).astype(np.uint8))


def test_trs_handles_title_space(tmp_path: Path) -> None:
    path = tmp_path / "titled.trs"
    traces = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    data = np.zeros((2, 0), dtype=np.uint8)
    _write_trs(path, traces=traces, data=data, title_length=8)
    with TRSDataset(path, name="titled") as dataset:
        assert dataset.shape == (2, 4)
        np.testing.assert_array_equal(dataset[1].trace, np.arange(4, 8))
