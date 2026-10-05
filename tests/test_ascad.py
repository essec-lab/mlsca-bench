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
from mlsca_bench.datasets.adapters import ASCADDataset


@pytest.fixture
def ascad_file(tmp_path: Path) -> Path:
    path = tmp_path / "ASCAD.h5"
    metadata_dtype = np.dtype(
        [
            ("plaintext", np.uint8, (16,)),
            ("key", np.uint8, (16,)),
            ("masks", np.uint8, (16,)),
            ("desync", np.uint8),
        ]
    )

    with h5py.File(path, "w") as h5_file:
        for group_name, count, offset in (
            ("Profiling_traces", 3, 0),
            ("Attack_traces", 2, 100),
        ):
            group = h5_file.create_group(group_name)
            group.create_dataset(
                "traces",
                data=np.arange(count * 5, dtype=np.float32).reshape(count, 5)
                + offset,
            )
            records = np.zeros(count, dtype=metadata_dtype)
            for index in range(count):
                records[index]["plaintext"] = np.arange(16) + index
                records[index]["key"] = np.arange(16)
                records[index]["masks"] = np.arange(16) + index + 1
                records[index]["desync"] = index
            group.create_dataset("metadata", data=records)
    return path


def test_ascad_implements_dataset_contract_lazily(ascad_file: Path) -> None:
    with ASCADDataset(ascad_file) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 5)
        assert dataset.available_fields == (
            "traces",
            "plaintexts",
            "keys",
            "masks",
        )
        assert isinstance(dataset.traces, h5py.Dataset)
        assert not isinstance(dataset.plaintexts, np.ndarray)
        np.testing.assert_array_equal(
            dataset.traces[:2],
            np.arange(10, dtype=np.float32).reshape(2, 5),
        )
        assert dataset.ciphertexts is None
        assert dataset.labels is None


def test_bulk_fields_and_samples_are_aligned(ascad_file: Path) -> None:
    with ASCADDataset(ascad_file) as dataset:
        assert dataset.plaintexts is not None
        assert dataset.keys is not None
        assert dataset.masks is not None
        np.testing.assert_array_equal(
            dataset.plaintexts[1], np.arange(16, dtype=np.uint8) + 1
        )
        np.testing.assert_array_equal(dataset.keys[1], np.arange(16))

        sample = dataset[1]
        np.testing.assert_array_equal(sample.trace, dataset.traces[1])
        np.testing.assert_array_equal(sample.plaintext, dataset.plaintexts[1])
        np.testing.assert_array_equal(sample.key, dataset.keys[1])
        np.testing.assert_array_equal(sample.mask, dataset.masks[1])
        assert sample.metadata["desync"].item() == 1


def test_attack_split_and_negative_indexing(ascad_file: Path) -> None:
    with ASCADDataset(ascad_file, split="attack") as dataset:
        assert dataset.split == "attack"
        assert len(dataset) == 2
        np.testing.assert_array_equal(dataset[-1].trace, dataset.traces[1])
        assert len(dataset[::-1]) == 2


def test_close_is_idempotent_and_invalidates_lazy_views(ascad_file: Path) -> None:
    dataset = ASCADDataset(ascad_file)
    plaintexts = dataset.plaintexts
    assert plaintexts is not None
    dataset.close()
    dataset.close()
    assert dataset.closed
    with pytest.raises(RuntimeError, match="closed"):
        _ = dataset.traces
    with pytest.raises(RuntimeError, match="closed"):
        _ = plaintexts[0]


def test_missing_metadata_fields_are_none(tmp_path: Path) -> None:
    path = tmp_path / "traces-only.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("Profiling_traces")
        group.create_dataset("traces", data=np.zeros((2, 5), dtype=np.float32))

    with ASCADDataset(path) as dataset:
        validate_dataset(dataset)
        assert dataset.plaintexts is None
        assert dataset.keys is None
        assert dataset[0].metadata == {}


def test_rejects_misaligned_metadata(tmp_path: Path) -> None:
    path = tmp_path / "misaligned.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("Profiling_traces")
        group.create_dataset("traces", data=np.zeros((2, 5)))
        group.create_dataset(
            "metadata", data=np.zeros(1, dtype=[("plaintext", "u1", (16,))])
        )

    with pytest.raises(ValueError, match="not aligned"):
        ASCADDataset(path)


def test_public_loader_accepts_alias_and_directory(ascad_file: Path) -> None:
    with load_dataset(
        "ascad-fixed", path=ascad_file.parent, split="attack"
    ) as dataset:
        assert isinstance(dataset, ASCADDataset)
        assert dataset.name == "ascadf"
        assert dataset.split == "attack"


def test_explicit_path_does_not_download(
    monkeypatch: pytest.MonkeyPatch, ascad_file: Path
) -> None:
    from mlsca_bench.datasets import loading

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("download_dataset should not be called")

    monkeypatch.setattr(loading, "download_dataset", fail_if_called)
    with loading.load_dataset("ascadf", path=ascad_file) as dataset:
        assert len(dataset) == 3


def test_downloaded_file_is_forwarded_to_adapter(
    monkeypatch: pytest.MonkeyPatch, ascad_file: Path, tmp_path: Path
) -> None:
    from mlsca_bench.datasets import loading

    def fake_download(
        name: str,
        destination: str | Path | None,
        **kwargs: object,
    ) -> tuple[Path, ...]:
        assert name == "ascadf"
        assert destination == tmp_path
        assert kwargs["progress"] is False
        return (ascad_file,)

    monkeypatch.setattr(loading, "download_dataset", fake_download)
    with loading.load_dataset(
        "ascadf", destination=tmp_path, progress=False
    ) as dataset:
        assert len(dataset) == 3


def test_dataset_without_adapter_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mlsca_bench.datasets import loading
    from mlsca_bench.datasets.registry import DatasetSpec

    spec = DatasetSpec(
        name="no-adapter",
        display_name="No Adapter",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter=None,
    )
    monkeypatch.setattr(loading, "get_dataset", lambda name: spec)
    with pytest.raises(ValueError, match="downloadable.*loading adapter"):
        loading.load_dataset("no-adapter", path=tmp_path)


def test_ascad_compatible_adapter_is_reused_for_ge_wars(
    ascad_file: Path,
) -> None:
    with load_dataset("ge-wars", path=ascad_file) as dataset:
        assert isinstance(dataset, ASCADDataset)
        assert dataset.name == "ge-wars"
        assert len(dataset) == 3


def test_ascad_variable_key_family_is_registered() -> None:
    from mlsca_bench.datasets.registry import get_dataset

    for name in ("ascadv", "ascadv-desync50", "ascadv-desync100"):
        spec = get_dataset(name)
        assert spec.adapter == "ascad"
        assert spec.availability == "automatic"
        assert spec.files[0].known_hash.startswith("sha1:")
        assert len(spec.files[0].known_hash.split(":")[1]) == 40

    raw = get_dataset("ascadv-raw")
    assert raw.adapter == "hdf5"
    assert raw.adapter_config["splits"] == {"all": "/"}

    v2 = get_dataset("ascadv2")
    assert v2.adapter == "ascad"
    # Verified against the authoritative ANSSI sha1 (anssi/ascadv2/sha1.txt).
    assert v2.availability == "automatic"
    assert v2.files[0].known_hash.startswith("sha1:")

    v2r = get_dataset("ascadv2r")
    assert v2r.availability == "automatic"          # all 8 raw files sha1-pinned
    assert all(f.known_hash.startswith("sha1:") for f in v2r.files)
    assert len(v2r.files) == 8


def test_ascadv_aliases_resolve() -> None:
    from mlsca_bench.datasets.registry import get_dataset

    assert get_dataset("ascad-variable").name == "ascadv"
    assert get_dataset("ascad-variable-desync100").name == "ascadv-desync100"
    assert get_dataset("ascad-v2").name == "ascadv2"


def test_load_ascadv_uses_ascad_adapter(ascad_file: Path) -> None:
    with load_dataset("ascadv", path=ascad_file, split="attack") as dataset:
        assert isinstance(dataset, ASCADDataset)
        assert dataset.name == "ascadv"
        assert dataset.split == "attack"


def test_load_ascadv_raw_root_layout(tmp_path: Path) -> None:
    from mlsca_bench.datasets.adapters import HDF5CompoundDataset

    path = tmp_path / "atmega8515-raw-traces.h5"
    metadata_dtype = np.dtype(
        [
            ("plaintext", np.uint8, (16,)),
            ("key", np.uint8, (16,)),
            ("masks", np.uint8, (16,)),
        ]
    )
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "traces", data=np.arange(4 * 6, dtype=np.int8).reshape(4, 6)
        )
        records = np.zeros(4, dtype=metadata_dtype)
        for index in range(4):
            records[index]["plaintext"] = np.arange(16) + index
            records[index]["key"] = np.arange(16)
        h5_file.create_dataset("metadata", data=records)

    with load_dataset("ascadv-raw", path=path) as dataset:
        assert isinstance(dataset, HDF5CompoundDataset)
        validate_dataset(dataset)
        assert dataset.shape == (4, 6)
        assert dataset.available_fields == ("traces", "plaintexts", "keys", "masks")
        np.testing.assert_array_equal(dataset[2].plaintext, np.arange(16) + 2)
