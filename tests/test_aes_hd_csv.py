# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench import load_dataset
from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import AESHDCSVDataset


@pytest.fixture
def aes_hd_directory(tmp_path: Path) -> Path:
    np.savetxt(
        tmp_path / "traces_1.csv",
        np.array([[1, 2, 3], [4, 5, 6]]),
        fmt="%d",
        delimiter=" ",
    )
    np.savetxt(
        tmp_path / "traces_2.csv",
        np.array([[7, 8, 9]]),
        fmt="%d",
        delimiter=" ",
    )
    np.savetxt(tmp_path / "labels.csv", np.array([10, 11, 12]), fmt="%d")
    return tmp_path


def test_text_adapter_satisfies_contract_lazily(aes_hd_directory: Path) -> None:
    with AESHDCSVDataset(aes_hd_directory) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 3)
        assert dataset.available_fields == ("traces", "labels")
        assert not isinstance(dataset.traces, np.ndarray)
        assert dataset.plaintexts is None
        assert dataset.keys is None


def test_reads_across_files_in_numeric_order(aes_hd_directory: Path) -> None:
    with AESHDCSVDataset(aes_hd_directory) as dataset:
        np.testing.assert_array_equal(
            dataset.traces[:],
            np.arange(1, 10, dtype=np.float32).reshape(3, 3),
        )
        np.testing.assert_array_equal(
            dataset.traces[[2, 0]],
            np.array([[7, 8, 9], [1, 2, 3]], dtype=np.float32),
        )
        sample = dataset[-1]
        np.testing.assert_array_equal(sample.trace, [7, 8, 9])
        assert sample.label.item() == 12


def test_rejects_misaligned_labels(aes_hd_directory: Path) -> None:
    np.savetxt(aes_hd_directory / "labels.csv", np.array([1, 2]), fmt="%d")
    with pytest.raises(ValueError, match="not aligned"):
        AESHDCSVDataset(aes_hd_directory)


def test_rejects_inconsistent_trace_lengths(aes_hd_directory: Path) -> None:
    np.savetxt(
        aes_hd_directory / "traces_2.csv",
        np.array([[7, 8]]),
        fmt="%d",
        delimiter=" ",
    )
    with pytest.raises(ValueError, match="trace lengths|expected 3"):
        AESHDCSVDataset(aes_hd_directory)


def test_public_loader_accepts_directory(aes_hd_directory: Path) -> None:
    with load_dataset("aes-hd-csv", path=aes_hd_directory) as dataset:
        assert isinstance(dataset, AESHDCSVDataset)
        assert dataset.name == "aes-hd-git"
        assert len(dataset) == 3


def test_public_loader_rejects_invented_split(aes_hd_directory: Path) -> None:
    with pytest.raises(ValueError, match="does not define splits"):
        load_dataset("aes-hd-git", path=aes_hd_directory, split="profiling")


def test_close_invalidates_text_view(aes_hd_directory: Path) -> None:
    dataset = AESHDCSVDataset(aes_hd_directory)
    traces = dataset.traces
    dataset.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = traces[0]
