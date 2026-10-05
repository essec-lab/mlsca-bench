# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bz2
import struct
from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import LeCroyIndexDataset
from mlsca_bench.datasets.adapters.lecroy import _read_lecroy


def _make_trc_bytes(samples: np.ndarray, preamble: bytes = b"##") -> bytes:
    """Build a minimal little-endian, int8 LeCroy .trc payload."""

    descriptor = bytearray(346)
    descriptor[0:8] = b"WAVEDESC"
    struct.pack_into("<h", descriptor, 32, 0)  # COMM_TYPE: byte (int8)
    struct.pack_into("<h", descriptor, 34, 1)  # COMM_ORDER: little-endian
    struct.pack_into("<i", descriptor, 36, 346)  # WAVE_DESCRIPTOR length
    struct.pack_into("<i", descriptor, 40, 0)  # USER_TEXT
    struct.pack_into("<i", descriptor, 48, 0)  # TRIGTIME_ARRAY
    struct.pack_into("<i", descriptor, 60, samples.size)  # WAVE_ARRAY_1 (bytes)
    return preamble + bytes(descriptor) + samples.astype(np.int8).tobytes()


def test_read_lecroy_parses_wavedesc() -> None:
    samples = np.array([-3, -1, 0, 5, 127, -128], dtype=np.int8)
    decoded = _read_lecroy(_make_trc_bytes(samples))
    np.testing.assert_array_equal(decoded, samples)


@pytest.fixture
def dpav4_like_dir(tmp_path: Path) -> Path:
    key = "00112233445566778899aabbccddeeff"
    rows = []
    for i in range(3):
        name = f"DPACV42_{i:06d}.trc.bz2"
        samples = (np.arange(6) + i).astype(np.int8)
        (tmp_path / name).write_bytes(bz2.compress(_make_trc_bytes(samples)))
        plaintext = f"{i:032x}"
        ciphertext = f"{(i + 1):032x}"
        rows.append(f"{key} {plaintext} {ciphertext} 0000 1111 2222 k00 {name}")
    (tmp_path / "dpav4_2_index").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_lecroy_index_dataset(dpav4_like_dir: Path) -> None:
    with LeCroyIndexDataset(
        dpav4_like_dir,
        name="dpav4.2",
        index_filename="dpav4_2_index",
        columns={"keys": 0, "plaintexts": 1, "ciphertexts": 2, "directory": 6, "filename": 7},
        algorithm="AES-128",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 6)
        assert dataset.available_fields == ("traces", "plaintexts", "ciphertexts", "keys")
        np.testing.assert_array_equal(dataset[2].trace, (np.arange(6) + 2).astype(np.int8))
        assert dataset[0].key[0] == 0x00 and dataset[0].key[-1] == 0xFF
        assert dataset[1].plaintext[-1] == 1
        assert dataset[1].ciphertext[-1] == 2
