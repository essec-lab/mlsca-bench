# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets.adapters._concat import ConcatArray

h5py = pytest.importorskip("h5py")

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import ConcatDataset
from mlsca_bench.datasets.registry import DatasetSpec


def test_concat_array_is_lazy_and_aligned() -> None:
    parts = [np.arange(0, 6).reshape(3, 2), np.arange(100, 104).reshape(2, 2)]
    concat = ConcatArray(parts)
    assert concat.shape == (5, 2)
    assert len(concat) == 5
    np.testing.assert_array_equal(concat[0], [0, 1])
    np.testing.assert_array_equal(concat[3], [100, 101])  # crosses into part 2
    np.testing.assert_array_equal(concat[-1], [102, 103])
    np.testing.assert_array_equal(concat[2:5], [[4, 5], [100, 101], [102, 103]])


def _make_chunk(path: Path, count: int, offset: int) -> None:
    metadata_dtype = np.dtype([("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,))])
    with h5py.File(path, "w") as h5_file:
        for split_name in ("profiling", "attack"):
            group = h5_file.create_group(split_name)
            group.create_dataset(
                "traces",
                data=np.arange(count * 4, dtype=np.float32).reshape(count, 4) + offset,
            )
            records = np.zeros(count, dtype=metadata_dtype)
            for index in range(count):
                records[index]["plaintext"] = np.arange(16) + offset + index
                records[index]["key"] = np.arange(16)
            group.create_dataset("metadata", data=records)


def _spec_dfs() -> DatasetSpec:
    return DatasetSpec(
        name="dfs-desynch",
        display_name="DFS",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="hdf5",
        adapter_config={
            "chunked": True,
            "chunk_glob": "*.h5",
            "splits": {"profiling": "profiling", "attack": "attack"},
            "default_split": "profiling",
            "field_aliases": {"plaintexts": ["plaintext"], "keys": ["key"]},
            "algorithm": "AES",
        },
    )


def test_chunked_hdf5_concatenates_all_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_chunk(tmp_path / "chunk_1.h5", count=3, offset=0)
    _make_chunk(tmp_path / "chunk_2.h5", count=2, offset=1000)
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_dfs())

    with loading.load_dataset("dfs-desynch", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 5  # 3 + 2 across the two chunks
        assert dataset.shape == (5, 4)
        assert dataset.metadata["chunks"] == 2
        # First sample of chunk 2 sits at global index 3.
        np.testing.assert_array_equal(dataset[3].trace, np.arange(4) + 1000)
        np.testing.assert_array_equal(dataset.traces[3], np.arange(4) + 1000)
        np.testing.assert_array_equal(
            dataset[3].plaintext, (np.arange(16) + 1000).astype(np.uint8)
        )


def _spec_present() -> DatasetSpec:
    return DatasetSpec(
        name="present-2021-ti-misaligned",
        display_name="PRESENT TI",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="raw-binary",
        adapter_config={
            "chunked": True,
            "chunk_glob": "Traces_*.dat",
            "record": {
                "filename": "Traces_1.dat",
                "dtype": [["trace", "i1", [4]], ["group", "u1"]],
                "fields": {"traces": "trace", "labels": "group"},
            },
        },
    )


def test_chunked_raw_binary_record_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_dtype = np.dtype([("trace", "i1", (4,)), ("group", "u1")])
    for file_index, base in ((1, 0), (2, 50)):
        records = np.zeros(3, dtype=record_dtype)
        for index in range(3):
            records[index]["trace"] = np.arange(4) + base + index
            records[index]["group"] = (base + index) % 2
        records.tofile(tmp_path / f"Traces_{file_index}.dat")
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_present())

    with loading.load_dataset("present-2021-ti-misaligned", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 6  # 3 + 3
        np.testing.assert_array_equal(dataset[4].trace, np.arange(4) + 51)


def _spec_dilithium() -> DatasetSpec:
    return DatasetSpec(
        name="dtds-dilithium2",
        display_name="DTDS Dilithium2",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="npy",
        adapter_config={
            "chunked": True,
            "layout": {
                "traces": "polyz_unpack_traces.npy",
                "labels": "polyz_unpack_y.npy",
            },
        },
    )


def test_chunked_npy_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two numbered chunk directories, each with the two .npy files (as on disk).
    for chunk, base in (("0", 0), ("1", 100)):
        chunk_dir = tmp_path / chunk
        chunk_dir.mkdir()
        np.save(
            chunk_dir / "polyz_unpack_traces.npy",
            (np.arange(2 * 5, dtype=np.int16).reshape(2, 5) + base),
        )
        np.save(
            chunk_dir / "polyz_unpack_y.npy",
            (np.arange(2 * 3, dtype=np.int32).reshape(2, 3) + base),
        )
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_dilithium())

    with loading.load_dataset("dtds-dilithium2", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 4
        assert dataset.available_fields == ("traces", "labels")
        np.testing.assert_array_equal(dataset[2].trace, np.arange(5).astype(np.int16) + 100)
        np.testing.assert_array_equal(dataset[0].label, np.arange(3).astype(np.int32))


def _make_chameleon_chunk(path: Path, count: int, offset: int, *, with_frequencies: bool = False) -> None:
    key_dtype = np.dtype([("k", np.uint8, (16,))])
    pt_dtype = np.dtype([("p", np.uint8, (16,))])
    with h5py.File(path, "w") as h5_file:
        traces = h5_file.create_group("data/traces")
        for i in range(count):
            traces.create_dataset(
                f"trace_{i}", data=(np.arange(5) + offset + i).astype(np.int16)
            )
        ciphers = h5_file.create_group("metadata/ciphers")
        for i in range(count):
            sub = ciphers.create_group(f"ciphers_{i}")
            key_rec = np.zeros((1,), dtype=key_dtype)
            key_rec[0]["k"] = np.arange(16)
            sub.create_dataset("key", data=key_rec)
            pt_rec = np.zeros((1,), dtype=pt_dtype)
            pt_rec[0]["p"] = np.arange(16) + offset + i
            sub.create_dataset("plaintexts", data=pt_rec)
        # Two AES executions per trace, with start/end offsets.
        pin_dtype = np.dtype([("start", np.int64), ("end", np.int64)])
        pinpoints = h5_file.create_group("metadata/pinpoints")
        for i in range(count):
            rec = np.zeros((2,), dtype=pin_dtype)
            rec["start"] = [10 + offset + i, 100 + offset + i]
            rec["end"] = [20 + offset + i, 200 + offset + i]
            pinpoints.create_dataset(f"pinpoints_{i}", data=rec)
        if with_frequencies:  # DFS sub-dataset only
            freq_dtype = np.dtype([("sample", np.int64), ("frequency", np.float64)])
            freqs = h5_file.create_group("metadata/frequencies")
            for i in range(count):
                rec = np.zeros((2,), dtype=freq_dtype)
                rec["sample"] = [0, 500 + offset + i]
                rec["frequency"] = [100.0, 50.0]
                freqs.create_dataset(f"frequencies_{i}", data=rec)


def _spec_chameleon() -> DatasetSpec:
    return DatasetSpec(
        name="chameleon-base",
        display_name="Chameleon BASE",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="chameleon",
        adapter_config={"chunked": True, "chunk_glob": "chameleon_base_chunk_*.h5"},
    )


def test_chunked_chameleon_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_chameleon_chunk(tmp_path / "chameleon_base_chunk_1.h5", count=2, offset=0)
    _make_chameleon_chunk(tmp_path / "chameleon_base_chunk_2.h5", count=3, offset=50)
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_chameleon())

    with loading.load_dataset("chameleon-base", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 5  # 2 + 3
        assert dataset.shape == (5, 5)
        assert dataset.available_fields == ("traces", "plaintexts", "keys")
        # Global index 2 is the first trace of chunk 2 (chunk 1 held 2 traces).
        np.testing.assert_array_equal(dataset[2].trace, (np.arange(5) + 50).astype(np.int16))

        # Pinpoints (Chameleon-specific) survive chunking and route to the right
        # chunk: global index 2 == chunk-2 trace 0 (offset 50).
        assert dataset.pinpoints is not None
        assert len(dataset.pinpoints) == 5
        np.testing.assert_array_equal(
            np.asarray(dataset.pinpoints[2]), np.array([[60, 70], [150, 250]])
        )
        # trace 0 of chunk 1 (offset 0): [[10,20],[100,200]]
        np.testing.assert_array_equal(
            np.asarray(dataset.pinpoints[0]), np.array([[10, 20], [100, 200]])
        )
        # BASE has no DFS clock schedule.
        assert dataset.frequencies is None


def test_chunked_chameleon_dfs_frequencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_chameleon_chunk(tmp_path / "chameleon_dfs_chunk_1.h5", count=2, offset=0, with_frequencies=True)
    _make_chameleon_chunk(tmp_path / "chameleon_dfs_chunk_2.h5", count=3, offset=50, with_frequencies=True)

    def _spec_dfs() -> DatasetSpec:
        return DatasetSpec(
            name="chameleon-dfs", display_name="Chameleon DFS", aliases=(),
            availability="manual", files=(), homepage=None, paper=None,
            license=None, notes=None, adapter="chameleon",
            adapter_config={"chunked": True, "chunk_glob": "chameleon_dfs_chunk_*.h5"},
        )

    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_dfs())
    with loading.load_dataset("chameleon-dfs", path=tmp_path) as dataset:
        assert len(dataset) == 5
        # frequencies survive chunking; global index 2 == chunk-2 trace 0 (offset 50).
        assert dataset.frequencies is not None
        np.testing.assert_array_equal(
            np.asarray(dataset.frequencies[2]), np.array([[0.0, 100.0], [550.0, 50.0]])
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.frequencies[0]), np.array([[0.0, 100.0], [500.0, 50.0]])
        )
        # pinpoints still work alongside frequencies
        assert dataset.pinpoints is not None
        np.testing.assert_array_equal(dataset[2].plaintext, (np.arange(16) + 50).astype(np.uint8))
        np.testing.assert_array_equal(dataset[2].key, np.arange(16))


def _spec_spook() -> DatasetSpec:
    return DatasetSpec(
        name="ches-ctf-2020-spook-sw3",
        display_name="Spook SW3",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="npz",
        adapter_config={
            "chunked": True,
            "chunk_glob": "rkey_sw3_*.npz",
            "layout": {"traces": "traces", "keys": "umsk_keys", "masks": "msk_keys"},
        },
    )


def test_chunked_npz_random_key_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Mirror the real Spook random-key member set.
    for file_index, base in ((1, 0), (2, 100)):
        np.savez(
            tmp_path / f"rkey_sw3_10000_{file_index}.npz",
            traces=(np.arange(2 * 7, dtype=np.int16).reshape(2, 7) + base),
            umsk_keys=np.arange(2 * 4, dtype=np.uint32).reshape(2, 4) + base,
            msk_keys=np.arange(2 * 8, dtype=np.uint32).reshape(2, 8),
            nonces=np.zeros((2, 4), dtype=np.uint32),
            seeds=np.zeros((2, 4), dtype=np.uint32),
        )
    # A fixed-key file that must be excluded by the glob.
    np.savez(tmp_path / "fkey_sw3_K0_1_0.npz", traces=np.zeros((1, 7), np.int16))
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_spook())

    with loading.load_dataset("ches-ctf-2020-spook-sw3", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 4  # only the two rkey files, 2 traces each
        assert dataset.available_fields == ("traces", "keys", "masks")
        assert dataset.plaintexts is None
        np.testing.assert_array_equal(dataset[2].trace, np.arange(7).astype(np.int16) + 100)
        np.testing.assert_array_equal(dataset[2].key, np.arange(4).astype(np.uint32) + 100)
