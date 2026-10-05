# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the classical baselines (CPA, DPA, Gaussian templates).

Pure NumPy — no torch needed. A synthetic dataset leaks the Hamming weight of
the S-box output at one sample (and one bit at another), so every attack must
recover the planted key; on pure noise the correct key must stay ~random.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsca_bench.benchmark import classical as C
from mlsca_bench.benchmark.classical import _checkpoints
from mlsca_bench.benchmark.leakage import HAMMING_WEIGHT, SBOX, LeakageModel
from mlsca_bench.benchmark.splits import DatasetSplits, SplitPolicy, Subset
from mlsca_bench.datasets.base import ArraySideChannelDataset

TRUE_KEY = 0xE0


def _dataset(n: int, noise: float, rng: np.random.Generator, *, leaky: bool) -> ArraySideChannelDataset:
    pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    keys = np.full((n, 16), TRUE_KEY, dtype=np.uint8)
    inter = SBOX[pt[:, 2] ^ keys[:, 2]]
    traces = (rng.standard_normal((n, 16)) * noise).astype(np.float32)
    if leaky:
        traces[:, 5] += HAMMING_WEIGHT[inter]           # HW leak (helps CPA/template)
        traces[:, 10] += ((inter >> 7) & 1)             # bit leak (helps DPA)
    return ArraySideChannelDataset(name="syn", traces=traces, plaintexts=pt, keys=keys)


def _install_splits(monkeypatch: pytest.MonkeyPatch, *, leaky: bool = True, noise: float = 3.0) -> None:
    def fake(name, *, policy=None, path=None, **kw):
        rng = np.random.default_rng(0)
        tr, te = _dataset(4000, noise, rng, leaky=leaky), _dataset(3000, noise, rng, leaky=leaky)
        return DatasetSplits(
            Subset(tr, np.arange(len(tr)), split_label="train"),
            Subset(tr, np.arange(50), split_label="val"),
            Subset(te, np.arange(len(te)), split_label="test"),
            policy or SplitPolicy(),
            {"train": np.arange(len(tr)), "val": np.arange(50), "test": np.arange(len(te))},
            (tr, te),
        )

    monkeypatch.setattr(C, "make_splits", fake)


def test_checkpoints() -> None:
    cp = _checkpoints(3000, 40)
    assert cp[0] >= 1 and cp[-1] == 3000
    assert np.all(np.diff(cp) > 0)                 # strictly increasing
    assert len(cp) <= 45
    assert np.array_equal(_checkpoints(20, 40), np.arange(1, 21))  # small -> contiguous


@pytest.mark.parametrize("attack", ["cpa", "dpa", "template"])
def test_classical_recovers_key(monkeypatch: pytest.MonkeyPatch, attack: str) -> None:
    _install_splits(monkeypatch, leaky=True)
    leak = LeakageModel(leakage="hw", byte=2)
    res = C.run_classical(attack, "syn", leakage=leak, n_experiments=20, seed=0)
    assert res.n_train == 4000 and res.n_test == 3000
    assert res.guessing_entropy[-1] < 2.0          # correct key at/near the top
    assert res.traces_to_disclosure is not None
    assert res.guessing_entropy.shape == res.success_rate.shape


def test_cpa_without_leak_stays_random(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_splits(monkeypatch, leaky=False)
    res = C.run_cpa("syn", leakage=LeakageModel(byte=2), n_experiments=20, seed=0)
    # No leakage -> the correct key should be nowhere near rank 0 on average.
    assert res.guessing_entropy[-1] > 20.0


def test_dpa_bit_selects_intermediate_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bit leak is on bit 7; attacking bit 7 works, a non-leaking bit is weaker.
    _install_splits(monkeypatch, leaky=True)
    good = C.run_dpa("syn", leakage=LeakageModel(byte=2), dpa_bit=7,
                     n_experiments=20, seed=0)
    assert good.guessing_entropy[-1] < 5.0
