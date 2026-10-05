# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import validate_dataset
from mlsca_bench.datasets.adapters import ManifestNpyDataset


@pytest.fixture
def manifest_dataset(tmp_path: Path) -> Path:
    """A minimal SIMPLE-DATASET manifest with two chunks and per-field .npy."""

    fields = {"traces": None, "umsk_key": None}
    chunks = {}
    for chunk_index, (count, base) in enumerate([(3, 0), (2, 100)]):
        files = {}
        for field, width, dtype in (("traces", 5, np.int16), ("umsk_key", 4, np.uint8)):
            rel = f"{field}/{chunk_index:04d}.npy"
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                path,
                (np.arange(count * width, dtype=dtype).reshape(count, width) + base),
            )
            files[field] = {"path": rel, "hash": "sha256-0"}
        chunks[f"chunk_{chunk_index}"] = {"nexec": count, "files": files}

    manifest = {
        "about": "SIMPLE-DATASET-MANIFEST",
        "version": "1.0",
        "id": "test",
        "metadata": {},
        "fields": {"traces": {"shape": [5], "dtype": "int16"}},
        "chunks": chunks,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_manifest_concatenates_chunks(manifest_dataset: Path) -> None:
    with ManifestNpyDataset(
        manifest_dataset,
        name="smaesh",
        field_map={"traces": "traces", "keys": "umsk_key"},
    ) as dataset:
        validate_dataset(dataset)
        assert len(dataset) == 5  # 3 + 2 across two chunks
        assert dataset.shape == (5, 5)
        assert dataset.available_fields == ("traces", "keys")
        # Global index 3 is the first row of chunk 2 (base 100).
        np.testing.assert_array_equal(dataset[3].trace, (np.arange(5) + 100).astype(np.int16))
        np.testing.assert_array_equal(dataset[3].key, (np.arange(4) + 100).astype(np.uint8))


def test_manifest_requires_traces_mapping(manifest_dataset: Path) -> None:
    with pytest.raises(ValueError, match="map the traces field"):
        ManifestNpyDataset(manifest_dataset, name="x", field_map={"keys": "umsk_key"})


def _write_manifest(root: Path, n: int, *, base: int, include_masks: bool = True) -> None:
    """Write a minimal SMAesH-shaped manifest at ``root``.

    ``include_masks`` mirrors the real dataset: the fixed-key attack set (fk0)
    ships traces + umsk_plaintext + umsk_key only, without msk_key.
    """

    root.mkdir(parents=True, exist_ok=True)
    files = {}
    layout = [
        ("traces", 6, np.int16),
        ("umsk_plaintext", 16, np.uint8),
        ("umsk_key", 16, np.uint8),
    ]
    if include_masks:
        layout.append(("msk_key", 16, np.uint8))
    for field, width, dtype in layout:
        rel = f"{field}/0000.npy"
        (root / field).mkdir(parents=True, exist_ok=True)
        arr = (np.arange(n * width, dtype=np.int64).reshape(n, width) + base) % 251
        np.save(root / rel, arr.astype(dtype))
        files[field] = {"path": rel, "hash": "sha256-0"}
    manifest = {
        "about": "SIMPLE-DATASET-MANIFEST",
        "version": "1.0",
        "id": root.name,
        "metadata": {},
        "fields": {"traces": {"shape": [6], "dtype": "int16"}},
        "chunks": {"chunk_0": {"nexec": n, "files": files}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def smaesh_layout(tmp_path: Path) -> Path:
    """Parent dir holding extracted vk0 (profiling) and fk0 (attack) splits."""

    _write_manifest(tmp_path / "smaesh-dataset-A7_d2-vk0", 40, base=0, include_masks=True)
    # fk0 (attack) has no msk_key, matching the real SMAesH dataset.
    _write_manifest(tmp_path / "smaesh-dataset-A7_d2-fk0", 15, base=100, include_masks=False)
    return tmp_path


def test_manifest_subdir_selects_split(smaesh_layout: Path) -> None:
    field_map = {"traces": "traces", "plaintexts": "umsk_plaintext", "keys": "umsk_key"}
    with ManifestNpyDataset(smaesh_layout, name="smaesh", field_map=field_map, subdir="vk0") as ds:
        assert len(ds) == 40
    with ManifestNpyDataset(smaesh_layout, name="smaesh", field_map=field_map, subdir="fk0") as ds:
        assert len(ds) == 15
    # Ambiguous without a split hint -> refuses rather than guessing.
    with pytest.raises(RuntimeError, match="Multiple manifest.json"):
        ManifestNpyDataset(smaesh_layout, name="smaesh", field_map=field_map)


def test_smaesh_make_splits_honors_author_split(smaesh_layout: Path) -> None:
    from mlsca_bench.benchmark.splits import make_splits

    with make_splits("smaesh-a7", path=str(smaesh_layout)) as splits:
        # test == the fixed-key attack set (fk0), untouched; train+val == vk0.
        assert len(splits.test) == 15
        assert len(splits.train) + len(splits.val) == 40
        assert splits.policy is not None
        # msk_key exists only in vk0: profiling exposes masks, the fk0 attack
        # set does not (the field is legitimately absent, not an error).
        assert splits.train.masks is not None
        assert splits.test.masks is None
        # both splits still carry plaintext + key for label building
        assert splits.test.plaintexts is not None and splits.test.keys is not None
