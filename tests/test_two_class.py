# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import TwoClassNpyDataset
from mlsca_bench.datasets.registry import DatasetSpec


@pytest.fixture
def reenc_like_dir(tmp_path: Path) -> Path:
    for impl in ("aes_nonprotect_sw", "aes_masked_hw"):
        for split, base in (("train", 0), ("test", 500)):
            folder = tmp_path / "dataset" / impl / split
            folder.mkdir(parents=True)
            np.save(folder / "fixed.npy", np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + base)
            np.save(
                folder / "random.npy",
                np.arange(3 * 4, dtype=np.float32).reshape(3, 4) + base + 100,
            )
    return tmp_path


def test_two_class_concatenates_and_labels(reenc_like_dir: Path) -> None:
    with TwoClassNpyDataset(
        reenc_like_dir,
        name="re-encryption",
        class0_file="fixed.npy",
        class1_file="random.npy",
        base_dir="aes_nonprotect_sw",
        split="train",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (5, 4)  # 2 fixed + 3 random
        assert dataset.available_fields == ("traces", "labels")
        # First two are fixed (label 0), last three random (label 1).
        assert [dataset[i].label.item() for i in range(5)] == [0, 0, 1, 1, 1]
        np.testing.assert_array_equal(dataset[0].trace, np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(dataset[2].trace, np.arange(4, dtype=np.float32) + 100)


def test_two_class_base_dir_selects_implementation(reenc_like_dir: Path) -> None:
    with TwoClassNpyDataset(
        reenc_like_dir,
        name="re-encryption",
        class0_file="fixed.npy",
        class1_file="random.npy",
        base_dir="aes_masked_hw",
        split="test",
    ) as dataset:
        # test/fixed.npy starts at base 500.
        np.testing.assert_array_equal(dataset[0].trace, np.arange(4, dtype=np.float32) + 500)


def test_two_class_public_loader(
    monkeypatch: pytest.MonkeyPatch, reenc_like_dir: Path
) -> None:
    spec = DatasetSpec(
        name="re-encryption",
        display_name="Re-encryption",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="two-class-npy",
        adapter_config={
            "base_dir": "aes_nonprotect_sw",
            "default_split": "train",
            "class0_file": "fixed.npy",
            "class1_file": "random.npy",
        },
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)
    with loading.load_dataset("re-encryption", path=reenc_like_dir) as dataset:
        assert len(dataset) == 5
        assert dataset[4].label.item() == 1
