# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from mlsca_bench.benchmark import (
    DatasetSplits,
    SplitPolicy,
    Subset,
    describe_splits,
    make_splits,
    resolve_policy,
    verify_splits,
)
from mlsca_bench.benchmark import splits as splits_mod
from mlsca_bench.datasets.base import ArraySideChannelDataset
from mlsca_bench.datasets.registry import DatasetSpec


def _dataset(n: int, *, base: int = 0, split: str | None = None) -> ArraySideChannelDataset:
    traces = (np.arange(n * 4, dtype=np.float32).reshape(n, 4)) + base * 1000
    meta = (np.arange(n * 16, dtype=np.uint8).reshape(n, 16))
    return ArraySideChannelDataset(
        name="synthetic", split=split, traces=traces, plaintexts=meta, keys=meta
    )


def _spec(adapter: str, config: dict | None = None) -> DatasetSpec:
    return DatasetSpec(
        name="synthetic",
        display_name="Synthetic",
        aliases=(),
        availability="manual",
        files=(),
        homepage=None,
        paper=None,
        license=None,
        notes=None,
        adapter=adapter,
        adapter_config=config,
    )


# --- policy & pure-numpy partitioning -------------------------------------

def test_policy_validates() -> None:
    with pytest.raises(ValueError):
        SplitPolicy(val_ratio=1.0)
    with pytest.raises(ValueError):
        SplitPolicy(ratios=(0.5, 0.3, 0.3))


def test_partition_is_disjoint_deterministic_and_sized() -> None:
    a = splits_mod._partition(100, SplitPolicy())
    b = splits_mod._partition(100, SplitPolicy())
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)  # deterministic
    train, val, test = a
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    union = np.concatenate([train, val, test])
    assert sorted(union.tolist()) == list(range(100))  # partition, no overlap
    for part in a:
        assert list(part) == sorted(part)  # stored sorted


def test_partition_changes_with_seed() -> None:
    assert not np.array_equal(
        splits_mod._partition(100, SplitPolicy(seed=0))[0],
        splits_mod._partition(100, SplitPolicy(seed=1))[0],
    )


# --- Subset view -----------------------------------------------------------

def test_subset_view_maps_indices() -> None:
    parent = _dataset(10)
    sub = Subset(parent, [9, 3, 7], split_label="train")
    assert len(sub) == 3
    assert sub.shape == (3, 4)
    assert sub.available_fields == ("traces", "plaintexts", "keys")
    np.testing.assert_array_equal(sub[0].trace, parent[9].trace)
    np.testing.assert_array_equal(sub.traces[2], parent.traces[7])
    np.testing.assert_array_equal(sub.traces[0:2], parent.traces[[9, 3]])


# --- make_splits: no-native-split case -------------------------------------

def test_make_splits_none_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("matlab"))
    monkeypatch.setattr(
        splits_mod, "load_dataset", lambda n, split=None, path=None, **k: _dataset(100)
    )
    with make_splits("synthetic") as s:
        assert (len(s.train), len(s.val), len(s.test)) == (70, 15, 15)
        allidx = np.concatenate([s.indices["train"], s.indices["val"], s.indices["test"]])
        assert sorted(allidx.tolist()) == list(range(100))  # disjoint cover


# --- make_splits: 2-way case (honor attack set, carve val from profiling) --

def test_make_splits_2way_honors_attack(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(name, split=None, path=None, **k):
        if split == "attack":
            return _dataset(20, base=9, split="attack")  # traces >= 9000
        return _dataset(100, base=0, split=split)          # traces < 9000

    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("ascad"))
    monkeypatch.setattr(splits_mod, "load_dataset", fake_load)

    with make_splits("synthetic") as s:
        # test == the full author attack set
        assert len(s.test) == 20
        assert s.test[0].trace[0] >= 9000
        # val is 10% of the 100-trace profiling pool; train the rest
        assert (len(s.train), len(s.val)) == (90, 10)
        # no attack traces leaked into train/val
        assert all(s.train[i].trace[0] < 9000 for i in range(len(s.train)))
        # train ∪ val exactly cover the profiling pool, disjoint
        pool = np.concatenate([s.indices["train"], s.indices["val"]])
        assert sorted(pool.tolist()) == list(range(100))


def test_make_splits_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("matlab"))
    monkeypatch.setattr(
        splits_mod, "load_dataset", lambda n, split=None, path=None, **k: _dataset(50)
    )
    with make_splits("synthetic") as a, make_splits("synthetic") as b:
        for key in ("train", "val", "test"):
            np.testing.assert_array_equal(a.indices[key], b.indices[key])


# --- make_splits: 3-way native (SCAAML-like) -------------------------------

def test_make_splits_3way_uses_native(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(name, split=None, path=None, **k):
        sizes = {"train": 30, "test": 10, "holdout": 8}
        return _dataset(sizes[split], split=split)

    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("scaaml"))
    monkeypatch.setattr(splits_mod, "load_dataset", fake_load)
    with make_splits("synthetic") as s:
        assert (len(s.train), len(s.val), len(s.test)) == (30, 8, 10)  # holdout -> val


# --- canonical config (splits.json) ----------------------------------------

def test_resolve_policy_default_matches_packaged() -> None:
    p = resolve_policy("anything")
    assert (p.seed, p.val_ratio, p.ratios) == (0, 0.1, (0.7, 0.15, 0.15))


def test_resolve_policy_applies_overrides() -> None:
    cfg = {
        "default": {"seed": 5, "val_ratio": 0.1, "ratios": [0.7, 0.15, 0.15]},
        "datasets": {"x": {"val_ratio": 0.2, "ratios": [0.8, 0.1, 0.1]}},
    }
    p = resolve_policy("x", config=cfg)
    assert p.seed == 5 and p.val_ratio == 0.2 and p.ratios == (0.8, 0.1, 0.1)


# --- manifest export / verify ----------------------------------------------

def test_describe_and_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("matlab"))
    monkeypatch.setattr(
        splits_mod, "load_dataset", lambda n, split=None, path=None, **k: _dataset(50)
    )
    with make_splits("synthetic") as s:
        manifest = describe_splits(s)
        assert sum(manifest["sizes"].values()) == 50
        assert set(manifest["checksums"]) == {"train", "val", "test"}
        assert verify_splits(s, manifest) is True
        tampered = {**manifest, "checksums": {**manifest["checksums"], "train": "deadbeef"}}
        assert verify_splits(s, tampered) is False


# --- lifecycle & edge cases ------------------------------------------------

def test_context_manager_closes_loaded_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _dataset(40)
    monkeypatch.setattr(splits_mod, "get_dataset", lambda n: _spec("matlab"))
    monkeypatch.setattr(splits_mod, "load_dataset", lambda n, split=None, path=None, **k: pool)
    with make_splits("synthetic") as s:
        assert not pool.closed
        assert s.train.closed is False
    assert pool.closed  # DatasetSplits closed the owned dataset on exit


def test_subset_traces_only_and_bad_index() -> None:
    parent = ArraySideChannelDataset(
        name="t", traces=np.arange(6, dtype=np.float32).reshape(3, 2)
    )
    sub = Subset(parent, [2, 0])
    assert sub.available_fields == ("traces",)
    assert sub.plaintexts is None
    np.testing.assert_array_equal(sub[-1].trace, parent[0].trace)
    with pytest.raises(IndexError):
        sub[5]
