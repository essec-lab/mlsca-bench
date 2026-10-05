# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import NpzArchiveDataset


@pytest.fixture
def spook_like_archive(tmp_path: Path) -> Path:
    path = tmp_path / "traces.npz"
    np.savez(
        path,
        traces=np.arange(3 * 5, dtype=np.float32).reshape(3, 5),
        plaintext=np.arange(3 * 16, dtype=np.uint8).reshape(3, 16),
        labels=np.arange(3, dtype=np.uint8),
        unused=np.zeros(99),
    )
    return path


def test_npz_reads_named_members(spook_like_archive: Path) -> None:
    with NpzArchiveDataset(
        spook_like_archive,
        name="spook",
        layout={"traces": "traces", "plaintexts": "plaintext", "labels": "labels"},
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 5)
        assert dataset.available_fields == ("traces", "plaintexts", "labels")
        assert dataset.ciphertexts is None
        np.testing.assert_array_equal(dataset[1].plaintext, np.arange(16, 32))
        assert dataset[1].label.item() == 1


def test_npz_directory_resolution(spook_like_archive: Path) -> None:
    with NpzArchiveDataset(
        spook_like_archive.parent,
        name="spook",
        layout={"traces": "traces"},
    ) as dataset:
        assert len(dataset) == 3


def test_npz_member_reads_header_before_materializing(
    spook_like_archive: Path,
) -> None:
    with NpzArchiveDataset(
        spook_like_archive, name="spook", layout={"traces": "traces"}
    ) as dataset:
        member = dataset.traces
        # shape/len come from the .npy header, with no array loaded yet.
        assert member.shape == (3, 5)
        assert len(member) == 3
        assert member._cache is None
        _ = member[0]
        assert member._cache is not None


def test_npz_missing_member_is_rejected(spook_like_archive: Path) -> None:
    with pytest.raises(KeyError, match="ciphertext"):
        NpzArchiveDataset(
            spook_like_archive,
            name="spook",
            layout={"traces": "traces", "ciphertexts": "ciphertext"},
        )
