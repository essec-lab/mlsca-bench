# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mlsca_bench.datasets import loading, validate_dataset
from mlsca_bench.datasets.adapters import PickleDataset, PickleTrustError
from mlsca_bench.datasets.registry import DatasetSpec


@pytest.fixture
def pickle_dir(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        np.arange(3 * 4, dtype=np.float64).reshape(3, 4),
        columns=["s0", "s1", "s2", "s3"],
    )
    frame.to_pickle(tmp_path / "Kx_const_cw_data.pickle")
    return tmp_path


def test_pickle_refuses_without_trust(pickle_dir: Path) -> None:
    with pytest.raises(PickleTrustError, match="trust_pickle=True"):
        PickleDataset(pickle_dir, name="galactics", member="Kx_const_cw_data.pickle")


def test_pickle_loads_with_trust(pickle_dir: Path) -> None:
    with pytest.warns(UserWarning, match="executes code"):
        dataset = PickleDataset(
            pickle_dir,
            name="galactics",
            member="Kx_const_cw_data.pickle",
            trust_pickle=True,
        )
    with dataset:
        validate_dataset(dataset)
        assert dataset.shape == (3, 4)
        np.testing.assert_array_equal(dataset[1].trace, [4, 5, 6, 7])


def test_pickle_traces_column(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"trace": [np.arange(4), np.arange(4) + 10, np.arange(4) + 20], "nonce": [0, 1, 2]}
    )
    frame.to_pickle(tmp_path / "yu_cw_data.pickle")
    with pytest.warns(UserWarning, match="executes code"):
        dataset = PickleDataset(
            tmp_path,
            name="galactics",
            member="yu_cw_data.pickle",
            traces_column="trace",
            trust_pickle=True,
        )
    with dataset:
        assert dataset.shape == (3, 4)
        np.testing.assert_array_equal(dataset[2].trace, np.arange(4) + 20)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        name="galactics-attack-data",
        display_name="GALACTICS",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter="pickle",
        adapter_config={"member": "Kx_const_cw_data.pickle"},
    )


def test_public_loader_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, pickle_dir: Path
) -> None:
    monkeypatch.setattr(loading, "get_dataset", lambda name: _spec())
    # Default call must NOT unpickle.
    with pytest.raises(PickleTrustError):
        loading.load_dataset("galactics-attack-data", path=pickle_dir)
    # Explicit opt-in works.
    with pytest.warns(UserWarning):
        dataset = loading.load_dataset(
            "galactics-attack-data", path=pickle_dir, trust_pickle=True
        )
    with dataset:
        assert len(dataset) == 3
