# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import AsciiWaveDataset
from mlsca_bench.datasets.registry import DatasetSpec

_HEADER = [
    "# Format : WORD",
    "# Type : AVERage type",
    "# Points : 3",
    "# Y increment : 1.41351E-05",
]


def _write_wave(path: Path, samples: list[int]) -> None:
    lines = _HEADER + [str(v) for v in samples]
    path.write_text("\n".join(lines) + "\n", encoding="latin1")


@pytest.fixture
def dpav2_like_dir(tmp_path: Path) -> Path:
    # Deliberately out of filesystem order to exercise index_regex ordering.
    _write_wave(tmp_path / "wave_aist_2009_n=10_k=00", [700, -354, 714])
    _write_wave(tmp_path / "wave_aist_2009_n=2_k=00", [-67, 5, 9])
    _write_wave(tmp_path / "wave_aist_2009_n=1_k=00", [1, 2, 3])
    return tmp_path


def test_ascii_wave_orders_by_index(dpav2_like_dir: Path) -> None:
    with AsciiWaveDataset(
        dpav2_like_dir,
        name="dpav2",
        file_glob="wave_*",
        index_regex=r"n=(\d+)",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 3)
        assert dataset.available_fields == ("traces",)
        assert dataset.traces.dtype == np.int16
        # Ordered n=1, n=2, n=10 (numeric, not lexicographic).
        np.testing.assert_array_equal(dataset[0].trace, [1, 2, 3])
        np.testing.assert_array_equal(dataset[1].trace, [-67, 5, 9])
        np.testing.assert_array_equal(dataset[2].trace, [700, -354, 714])


def test_ascii_wave_parses_filename_fields(tmp_path: Path) -> None:
    # DPA Contest v2 shape: k/m/c are hex in the filename (16 bytes each here 2).
    _write_wave(tmp_path / "wave_n=1_k=e0a1_m=1122_c=aabb.csv", [1, 2, 3])
    _write_wave(tmp_path / "wave_n=0_k=e0a1_m=3344_c=ccdd.csv", [4, 5, 6])
    fields = {
        "keys": r"_k=([0-9a-f]+)_",
        "plaintexts": r"_m=([0-9a-f]+)_",
        "ciphertexts": r"_c=([0-9a-f]+)",
    }
    with AsciiWaveDataset(
        tmp_path, name="dpav2", file_glob="wave_*",
        index_regex=r"n=(\d+)", filename_fields=fields,
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.available_fields == ("traces", "plaintexts", "ciphertexts", "keys")
        # Ordered by n: row 0 is n=0 (m=3344), row 1 is n=1 (m=1122).
        np.testing.assert_array_equal(dataset.plaintexts[:], [[0x33, 0x44], [0x11, 0x22]])
        np.testing.assert_array_equal(dataset.ciphertexts[:], [[0xCC, 0xDD], [0xAA, 0xBB]])
        # Fixed key on both rows.
        np.testing.assert_array_equal(dataset.keys[:], [[0xE0, 0xA1], [0xE0, 0xA1]])
        # Per-sample access stays aligned with the trace ordering.
        assert dataset[0].plaintext.tolist() == [0x33, 0x44]
        np.testing.assert_array_equal(dataset[0].trace, [4, 5, 6])


def test_ascii_wave_missing_filename_field_errors(tmp_path: Path) -> None:
    _write_wave(tmp_path / "wave_n=0_m=1122.csv", [1, 2, 3])
    with pytest.raises(ValueError, match="no keys match"):
        AsciiWaveDataset(
            tmp_path, name="x", file_glob="wave_*",
            filename_fields={"keys": r"_k=([0-9a-f]+)_"},
        )


def test_ascii_wave_public_loader(
    monkeypatch: pytest.MonkeyPatch, dpav2_like_dir: Path
) -> None:
    spec = DatasetSpec(
        name="dpacontest-v2",
        display_name="DPA Contest v2",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="ascii-wave",
        adapter_config={"file_glob": "wave_*", "index_regex": r"n=(\d+)"},
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)
    with loading.load_dataset("dpacontest-v2", path=dpav2_like_dir) as dataset:
        assert len(dataset) == 3
        np.testing.assert_array_equal(dataset[2].trace, [700, -354, 714])
