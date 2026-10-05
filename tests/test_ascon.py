# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mlsca_bench import load_dataset
from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import AsconHDF5Dataset


@pytest.fixture
def ascon_file(tmp_path: Path) -> Path:
    path = tmp_path / "ascon_cw_unprotected.h5"
    metadata_dtype = np.dtype(
        [
            ("key", np.uint8, (16,)),
            ("nonce", np.uint8, (16,)),
            ("plaintext", np.uint8, (4,)),
            ("associated_data", np.uint8, (4,)),
            ("ciphertext", np.uint8, (4,)),
            ("tag", np.uint8, (16,)),
        ]
    )
    with h5py.File(path, "w") as h5_file:
        for group_name, count, offset in (
            ("fixed_keys", 3, 0),
            ("random_keys", 2, 100),
        ):
            group = h5_file.create_group(group_name)
            group.create_dataset(
                "traces",
                data=np.arange(count * 6, dtype=np.float32).reshape(count, 6)
                + offset,
            )
            group.create_dataset(
                "labels",
                data=np.arange(count * 64, dtype=np.uint8).reshape(count, 64),
            )
            records = np.zeros(count, dtype=metadata_dtype)
            for index in range(count):
                records[index]["key"] = np.arange(16) + index
                records[index]["nonce"] = np.arange(16) + 20
                records[index]["plaintext"] = np.arange(4) + 40
                records[index]["associated_data"] = np.arange(4) + 50
                records[index]["ciphertext"] = np.arange(4) + 60
                records[index]["tag"] = np.arange(16) + 70
            group.create_dataset("metadata", data=records)
    return path


def test_ascon_adapter_satisfies_contract_lazily(ascon_file: Path) -> None:
    with AsconHDF5Dataset(ascon_file) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 6)
        assert dataset.available_fields == (
            "traces",
            "plaintexts",
            "ciphertexts",
            "keys",
            "labels",
        )
        assert isinstance(dataset.traces, h5py.Dataset)
        assert isinstance(dataset.labels, h5py.Dataset)
        assert not isinstance(dataset.keys, np.ndarray)
        assert dataset.masks is None


def test_ascon_sample_contains_standard_and_extra_fields(ascon_file: Path) -> None:
    with AsconHDF5Dataset(ascon_file) as dataset:
        sample = dataset[1]
        np.testing.assert_array_equal(sample.trace, dataset.traces[1])
        np.testing.assert_array_equal(sample.plaintext, dataset.plaintexts[1])
        np.testing.assert_array_equal(sample.ciphertext, dataset.ciphertexts[1])
        np.testing.assert_array_equal(sample.key, dataset.keys[1])
        np.testing.assert_array_equal(sample.label, dataset.labels[1])
        # The nonce (init-attack public input) is surfaced as ``plaintexts``;
        # the AEAD message plaintext falls through to per-sample metadata.
        np.testing.assert_array_equal(sample.plaintext, np.arange(16) + 20)   # nonce
        assert set(sample.metadata) == {"plaintext", "associated_data", "tag"}


def test_random_key_split(ascon_file: Path) -> None:
    with AsconHDF5Dataset(ascon_file, split="random") as dataset:
        assert dataset.split == "random"
        assert len(dataset) == 2
        assert dataset.traces[0, 0] == 100


def test_public_loader_accepts_ascon_alias(ascon_file: Path) -> None:
    with load_dataset(
        "ascon-unprotected", path=ascon_file, split="random"
    ) as dataset:
        assert isinstance(dataset, AsconHDF5Dataset)
        assert dataset.name == "ascon-cw-unprotected"
        assert dataset.split == "random"


def test_rejects_misaligned_labels(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("fixed_keys")
        group.create_dataset("traces", data=np.zeros((2, 5)))
        group.create_dataset("labels", data=np.zeros((1, 64)))
    with pytest.raises(ValueError, match="labels are not aligned"):
        AsconHDF5Dataset(path)
