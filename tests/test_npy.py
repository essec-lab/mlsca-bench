# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench import load_dataset
from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import AESHDZaidDataset, NpyDirectoryDataset


@pytest.fixture
def aes_hd_zaid_directory(tmp_path: Path) -> Path:
    for split, count, offset in (("profiling", 3, 0), ("attack", 2, 100)):
        np.save(
            tmp_path / f"{split}_traces_AES_HD.npy",
            np.arange(count * 5, dtype=np.float64).reshape(count, 5) + offset,
        )
        np.save(
            tmp_path / f"{split}_ciphertext_AES_HD.npy",
            np.arange(count * 16, dtype=np.float64).reshape(count, 16),
        )
        np.save(
            tmp_path / f"{split}_labels_AES_HD.npy",
            np.arange(count, dtype=np.float64).reshape(count, 1),
        )
    return tmp_path


def test_zaid_adapter_uses_memory_maps(aes_hd_zaid_directory: Path) -> None:
    with AESHDZaidDataset(aes_hd_zaid_directory) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 5)
        assert dataset.available_fields == ("traces", "ciphertexts", "labels")
        assert isinstance(dataset.traces, np.memmap)
        assert isinstance(dataset.ciphertexts, np.memmap)
        assert dataset.plaintexts is None
        assert dataset.keys is None


def test_zaid_attack_split_and_sample_alignment(
    aes_hd_zaid_directory: Path,
) -> None:
    with AESHDZaidDataset(aes_hd_zaid_directory, split="attack") as dataset:
        assert len(dataset) == 2
        assert dataset.traces[0, 0] == 100
        np.testing.assert_array_equal(dataset[1].ciphertext, dataset.ciphertexts[1])
        np.testing.assert_array_equal(dataset[1].label, dataset.labels[1])


def test_public_loader_for_zaid(aes_hd_zaid_directory: Path) -> None:
    with load_dataset(
        "aes-hd-preprocessed",
        path=aes_hd_zaid_directory,
        split="attack",
    ) as dataset:
        assert isinstance(dataset, AESHDZaidDataset)
        assert dataset.name == "aes-hd-zaid"
        assert dataset.split == "attack"


def test_generic_npy_layout_rejects_misalignment(tmp_path: Path) -> None:
    np.save(tmp_path / "traces.npy", np.zeros((3, 5)))
    np.save(tmp_path / "labels.npy", np.zeros(2))
    with pytest.raises(ValueError, match="one entry per trace"):
        NpyDirectoryDataset(
            tmp_path,
            name="invalid",
            layout={"traces": "traces.npy", "labels": "labels.npy"},
        )
