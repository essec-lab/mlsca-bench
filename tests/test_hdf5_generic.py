# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import FlatHDF5Dataset, HDF5CompoundDataset
from mlsca_bench.datasets.registry import DatasetSpec


@pytest.fixture
def generic_file(tmp_path: Path) -> Path:
    """An HDF5 file with non-ASCAD group and column names."""

    path = tmp_path / "generic.h5"
    metadata_dtype = np.dtype(
        [
            ("pt", np.uint8, (16,)),
            ("ky", np.uint8, (16,)),
            ("trial", np.uint8),
        ]
    )
    with h5py.File(path, "w") as h5_file:
        for group_name, count, offset in (("train", 4, 0), ("test", 2, 100)):
            group = h5_file.create_group(group_name)
            group.create_dataset(
                "traces",
                data=np.arange(count * 6, dtype=np.float32).reshape(count, 6) + offset,
            )
            group.create_dataset(
                "labels", data=np.arange(count, dtype=np.uint8) + offset
            )
            records = np.zeros(count, dtype=metadata_dtype)
            for index in range(count):
                records[index]["pt"] = np.arange(16) + index
                records[index]["ky"] = np.arange(16)
                records[index]["trial"] = index
            group.create_dataset("metadata", data=records)
    return path


_SPLITS = {"profiling": "train", "attack": "test"}
_ALIASES = {
    "plaintexts": ["pt"],
    "keys": ["ky"],
}


def test_generic_hdf5_reads_custom_schema(generic_file: Path) -> None:
    with HDF5CompoundDataset(
        generic_file,
        name="demo",
        splits=_SPLITS,
        split="profiling",
        field_aliases=_ALIASES,
        labels_dataset="labels",
        require_labels=True,
        label="Demo",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.name == "demo"
        assert dataset.shape == (4, 6)
        assert dataset.available_fields == ("traces", "plaintexts", "keys", "labels")
        assert dataset.ciphertexts is None
        assert dataset.masks is None
        assert isinstance(dataset.labels, h5py.Dataset)

        sample = dataset[2]
        np.testing.assert_array_equal(sample.plaintext, np.arange(16) + 2)
        np.testing.assert_array_equal(sample.key, np.arange(16))
        assert sample.label.item() == 2
        assert set(sample.metadata) == {"trial"}
        assert sample.metadata["trial"].item() == 2


def test_generic_hdf5_unknown_split_is_rejected(generic_file: Path) -> None:
    with pytest.raises(ValueError, match="Demo split must be one of"):
        HDF5CompoundDataset(
            generic_file,
            name="demo",
            splits=_SPLITS,
            split="validation",
            field_aliases=_ALIASES,
            label="Demo",
        )


def test_generic_hdf5_missing_required_labels(tmp_path: Path) -> None:
    path = tmp_path / "no-labels.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("train")
        group.create_dataset("traces", data=np.zeros((2, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="does not contain 'labels'"):
        HDF5CompoundDataset(
            path,
            name="demo",
            splits=_SPLITS,
            split="profiling",
            field_aliases=_ALIASES,
            labels_dataset="labels",
            require_labels=True,
        )


@pytest.fixture
def aesptv2_like_file(tmp_path: Path) -> Path:
    """Nested impl/split groups, capitalized 'Traces', metadata as datasets."""

    path = tmp_path / "device.h5"
    with h5py.File(path, "w") as h5_file:
        for split_name, count, offset in (("Profiling", 3, 0), ("Attack", 2, 100)):
            group = h5_file.create_group(f"Unprotected/{split_name}")
            group.create_dataset(
                "Traces",
                data=np.arange(count * 6, dtype=np.float32).reshape(count, 6) + offset,
            )
            group.create_dataset("Labels", data=np.arange(count, dtype=np.uint8))
            meta = group.create_group("MetaData")
            meta.create_dataset(
                "Plaintext",
                data=np.arange(count * 16, dtype=np.uint8).reshape(count, 16),
            )
            meta.create_dataset(
                "Key", data=np.arange(count * 16, dtype=np.uint8).reshape(count, 16)
            )
    return path


def test_hdf5_reads_separate_metadata_datasets(aesptv2_like_file: Path) -> None:
    with HDF5CompoundDataset(
        aesptv2_like_file,
        name="aes-ptv2",
        splits={"profiling": "Unprotected/Profiling", "attack": "Unprotected/Attack"},
        split="profiling",
        traces_dataset="Traces",
        field_aliases={
            "plaintexts": ["MetaData/Plaintext"],
            "keys": ["MetaData/Key"],
            "masks": ["MetaData/Masks"],
        },
        labels_dataset="Labels",
        label="AESPTv2",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 6)
        assert dataset.available_fields == (
            "traces",
            "plaintexts",
            "keys",
            "labels",
        )
        assert dataset.masks is None  # MetaData/Masks absent in Unprotected
        assert isinstance(dataset.plaintexts, h5py.Dataset)
        np.testing.assert_array_equal(dataset[1].plaintext, np.arange(16, 32))
        np.testing.assert_array_equal(dataset[2].key, np.arange(32, 48))
        assert dataset[0].label.item() == 0


def test_hdf5_missing_custom_traces_name(aesptv2_like_file: Path) -> None:
    with pytest.raises(ValueError, match="does not contain 'traces'"):
        HDF5CompoundDataset(
            aesptv2_like_file,
            name="aes-ptv2",
            splits={"profiling": "Unprotected/Profiling"},
            split="profiling",
            field_aliases={},
        )


@pytest.fixture
def ets_like_file(tmp_path: Path) -> Path:
    """Root-level 'traces' with a 'metadata' GROUP of per-field datasets (.ets)."""

    path = tmp_path / "nucleo.ets"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "traces", data=np.arange(4 * 6, dtype=np.int16).reshape(4, 6)
        )
        meta = h5_file.create_group("metadata")
        meta.create_dataset(
            "plaintext", data=np.arange(4 * 16, dtype=np.uint8).reshape(4, 16)
        )
        meta.create_dataset(
            "key", data=np.arange(4 * 16, dtype=np.uint8).reshape(4, 16)
        )
    return path


def test_hdf5_root_split_with_metadata_group(ets_like_file: Path) -> None:
    with HDF5CompoundDataset(
        ets_like_file,
        name="eshard",
        splits={"all": "/"},
        split="all",
        field_aliases={
            "plaintexts": ["metadata/plaintext"],
            "keys": ["metadata/key"],
        },
        label="eShard",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (4, 6)
        assert dataset.available_fields == ("traces", "plaintexts", "keys")
        np.testing.assert_array_equal(dataset[2].plaintext, np.arange(32, 48))


@pytest.fixture
def ecc_like_file(tmp_path: Path) -> Path:
    """Top-level per-split trace datasets with sibling *_data label datasets."""

    path = tmp_path / "cswap_pointer.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "profiling_traces", data=np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
        )
        h5_file.create_dataset(
            "attacking_traces",
            data=np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 100,
        )
        h5_file.create_dataset(
            "profiling_data", data=np.arange(3 * 2, dtype=np.uint8).reshape(3, 2)
        )
        h5_file.create_dataset(
            "attacking_data", data=np.arange(2 * 2, dtype=np.uint8).reshape(2, 2)
        )
    return path


def test_flat_hdf5_top_level_splits(ecc_like_file: Path) -> None:
    with FlatHDF5Dataset(
        ecc_like_file,
        name="ecc",
        splits={"profiling": "profiling_traces", "attack": "attacking_traces"},
        split="attack",
        labels={"profiling": "profiling_data", "attack": "attacking_data"},
        label="ECC",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (2, 4)
        assert dataset.available_fields == ("traces", "labels")
        assert dataset.plaintexts is None
        np.testing.assert_array_equal(dataset[0].trace, np.arange(4) + 100)
        np.testing.assert_array_equal(dataset[1].label, [2, 3])


def test_public_loader_uses_adapter_config(
    monkeypatch: pytest.MonkeyPatch, generic_file: Path
) -> None:
    spec = DatasetSpec(
        name="demo-hdf5",
        display_name="Demo HDF5",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="hdf5",
        adapter_config={
            "splits": _SPLITS,
            "default_split": "attack",
            "field_aliases": _ALIASES,
            "labels_dataset": "labels",
            "require_labels": True,
            "algorithm": "AES",
            "preferred_filename": "generic.h5",
        },
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)

    with loading.load_dataset("demo-hdf5", path=generic_file) as dataset:
        assert isinstance(dataset, HDF5CompoundDataset)
        assert dataset.name == "demo-hdf5"
        assert dataset.split == "attack"
        assert len(dataset) == 2
        assert dataset.metadata["algorithm"] == "AES"
