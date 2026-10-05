# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import AgilentWaveDataset
from mlsca_bench.datasets.adapters.agilent import _read_agilent
from mlsca_bench.datasets.registry import DatasetSpec


def _make_ag10(samples: np.ndarray) -> bytes:
    data = samples.astype("<f4").tobytes()
    waveform_header = bytearray(140)
    struct.pack_into("<i", waveform_header, 0, 140)  # header size
    struct.pack_into("<i", waveform_header, 12, samples.size)  # points (informational)
    data_header = struct.pack("<ihhi", 12, 1, 4, len(data))  # size, type=1, bpp=4, bufsize
    return b"AG10" + struct.pack("<ii", 0, 1) + bytes(waveform_header) + data_header + data


def test_read_agilent_header() -> None:
    samples = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    decoded = _read_agilent(_make_ag10(samples))
    np.testing.assert_allclose(decoded, samples, rtol=1e-6)


@pytest.fixture
def dpa_des_dir(tmp_path: Path) -> Path:
    for index, (k, m, c) in enumerate(
        [
            ("6b64796b64796b64", "1c534561330470ab", "0032d948aa7f152a"),
            ("6b64796b64796b64", "0ea0e9f4f1a38bac", "833f2dd1cc81c540"),
        ]
    ):
        samples = (np.arange(5, dtype=np.float32) + index)
        fname = f"wave_DES_HW_2007-09-2{index}_k={k}_m={m}_c={c}.bin"
        (tmp_path / fname).write_bytes(_make_ag10(samples))
    return tmp_path


def test_agilent_wave_with_filename_metadata(dpa_des_dir: Path) -> None:
    with AgilentWaveDataset(
        dpa_des_dir,
        name="dpa-v3",
        file_glob="wave_*.bin",
        filename_fields={
            "keys": r"k=([0-9a-fA-F]+)",
            "plaintexts": r"m=([0-9a-fA-F]+)",
            "ciphertexts": r"c=([0-9a-fA-F]+)",
        },
        algorithm="DES",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (2, 5)
        assert dataset.available_fields == ("traces", "plaintexts", "ciphertexts", "keys")
        np.testing.assert_allclose(dataset[1].trace, np.arange(5, dtype=np.float32) + 1)
        def as_bytes(hex_string: str) -> np.ndarray:
            return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8)

        np.testing.assert_array_equal(dataset[0].key, as_bytes("6b64796b64796b64"))
        np.testing.assert_array_equal(dataset[0].plaintext, as_bytes("1c534561330470ab"))
        np.testing.assert_array_equal(dataset[1].ciphertext, as_bytes("833f2dd1cc81c540"))


def test_agilent_public_loader(monkeypatch: pytest.MonkeyPatch, dpa_des_dir: Path) -> None:
    spec = DatasetSpec(
        name="dpacontest-v3-secmatv3-des",
        display_name="DPA v3 DES",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="agilent-wave",
        adapter_config={
            "file_glob": "wave_*.bin",
            "filename_fields": {
                "keys": r"k=([0-9a-fA-F]+)",
                "plaintexts": r"m=([0-9a-fA-F]+)",
                "ciphertexts": r"c=([0-9a-fA-F]+)",
            },
        },
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)
    with loading.load_dataset("dpacontest-v3-secmatv3-des", path=dpa_des_dir) as dataset:
        assert len(dataset) == 2
        assert dataset.keys is not None
