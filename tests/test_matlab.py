# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.io import savemat

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import ConcatDataset
from mlsca_bench.datasets.adapters.matlab import MatlabDataset, MatlabMultiFileDataset
from mlsca_bench.datasets.registry import DatasetSpec


def _obj(*strings: str) -> np.ndarray:
    array = np.empty((len(strings),), dtype=object)
    for index, value in enumerate(strings):
        array[index] = value
    return array


@pytest.fixture
def aes_rd_like_file(tmp_path: Path) -> Path:
    path = tmp_path / "ctraces.mat"
    savemat(
        path,
        {
            # MATLAB SCA files commonly store traces column-per-trace.
            "CompressedTraces": np.arange(5 * 3, dtype=np.float64).reshape(5, 3),
            "plaintext": np.arange(3 * 16, dtype=np.uint8).reshape(3, 16),
            "key": np.arange(3 * 16, dtype=np.uint8).reshape(3, 16),
        },
    )
    return path


def test_matlab_reads_and_transposes(aes_rd_like_file: Path) -> None:
    with MatlabDataset(
        aes_rd_like_file,
        name="aes-rd",
        fields={
            "traces": "CompressedTraces",
            "plaintexts": "plaintext",
            "keys": "key",
        },
        transpose_traces=True,
        metadata={"algorithm": "AES"},
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 5)
        assert dataset.available_fields == ("traces", "plaintexts", "keys")
        assert dataset.metadata["algorithm"] == "AES"
        np.testing.assert_array_equal(dataset[0].plaintext, np.arange(16))


def test_matlab_preferred_filename_selects_one(tmp_path: Path) -> None:
    for device, base in (("cw308XGD2.mat", 0), ("cw308XGD3.mat", 100)):
        savemat(
            tmp_path / device,
            {
                "traces": np.arange(3 * 4, dtype=np.float64).reshape(3, 4) + base,
                "key": np.arange(3 * 16, dtype=np.uint8).reshape(3, 16),
            },
        )
    # Without a preference, the ambiguous directory is rejected.
    with pytest.raises(RuntimeError, match="preferred_filename"):
        MatlabDataset(tmp_path, name="x-deepsca", fields={"traces": "traces"})
    # With a preference, the named device loads.
    with MatlabDataset(
        tmp_path,
        name="x-deepsca",
        fields={"traces": "traces", "keys": "key"},
        preferred_filename="cw308XGD3.mat",
    ) as dataset:
        assert dataset.shape == (3, 4)
        assert dataset.traces[0, 0] == 100


def test_matlab_multifile_hex_and_base_dir(tmp_path: Path) -> None:
    for device, base in (("c1_k1", 0.0), ("c2_k1", 99.0)):
        folder = tmp_path / device
        folder.mkdir()
        savemat(
            folder / "traces.mat",
            {"traces": np.arange(2 * 4, dtype=np.float64).reshape(2, 4) + base},
        )
        savemat(
            folder / "ptext.mat",
            {"ptext": _obj("000102030405060708090a0b0c0d0e0f", "10" * 16)},
        )
        savemat(folder / "ctext.mat", {"ctext": _obj("ff" * 16, "ee" * 16)})

    with MatlabMultiFileDataset(
        tmp_path,
        name="portability",
        field_files={
            "traces": {"filename": "traces.mat", "var": "traces"},
            "plaintexts": {"filename": "ptext.mat", "var": "ptext", "hex": True},
            "ciphertexts": {"filename": "ctext.mat", "var": "ctext", "hex": True},
        },
        base_dir="c1_k1",
    ) as dataset:
        validate_dataset(dataset)
        assert dataset.shape == (2, 4)
        assert dataset.traces[0, 0] == 0.0  # picked c1_k1, not c2_k1
        np.testing.assert_array_equal(dataset[0].plaintext, np.arange(16))
        assert dataset[1].ciphertext[0] == 0xEE


def _spec_kyber() -> DatasetSpec:
    return DatasetSpec(
        name="kyber-reference-ppm",
        display_name="Kyber PPM",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="matlab",
        adapter_config={
            "chunked": True,
            "chunk_glob": "tracesA*.mat",
            "field_files": {
                "traces": {"var": "tracesA", "anchor": True},
                "labels": {"var": "noncesA", "from_anchor": ["tracesA", "noncesA"]},
            },
        },
    )


def test_matlab_paired_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for index, base in ((0, 0), (1, 100)):
        savemat(
            tmp_path / f"tracesA{index}.mat",
            {"tracesA": np.arange(2 * 5, dtype=np.float64).reshape(2, 5) + base},
        )
        savemat(
            tmp_path / f"noncesA{index}.mat",
            {"noncesA": np.arange(2 * 12, dtype=np.int32).reshape(2, 12) + base},
        )
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec_kyber())

    with loading.load_dataset("kyber-reference-ppm", path=tmp_path) as dataset:
        assert isinstance(dataset, ConcatDataset)
        validate_dataset(dataset)
        assert len(dataset) == 4
        assert dataset.available_fields == ("traces", "labels")
        np.testing.assert_array_equal(dataset[2].trace, np.arange(5) + 100)
        np.testing.assert_array_equal(dataset[2].label, np.arange(12) + 100)


def test_matlab_missing_variable_is_rejected(aes_rd_like_file: Path) -> None:
    with pytest.raises(KeyError, match="ciphertext"):
        MatlabDataset(
            aes_rd_like_file,
            name="aes-rd",
            fields={"traces": "CompressedTraces", "ciphertexts": "ciphertext"},
        )
