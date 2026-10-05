# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from mlsca_bench.benchmark import (
    HAMMING_WEIGHT,
    LeakageModel,
    SBOX,
    evaluate_key_rank,
    guessing_entropy,
)
from mlsca_bench.datasets.base import ArraySideChannelDataset


# --- leakage model ---------------------------------------------------------

def test_sbox_known_values() -> None:
    assert SBOX[0x00] == 0x63
    assert SBOX[0x53] == 0xED
    assert SBOX[0xFF] == 0x16
    assert len(np.unique(SBOX)) == 256  # a permutation


def test_identity_and_hw_labels() -> None:
    idm = LeakageModel(leakage="id", byte=0)
    hwm = LeakageModel(leakage="hw", byte=0)
    assert idm.n_classes == 256 and hwm.n_classes == 9
    pt = np.array([0x12, 0x00], dtype=np.uint8)
    k = np.array([0x34, 0x00], dtype=np.uint8)
    np.testing.assert_array_equal(idm.labels(pt, k), SBOX[pt ^ k].astype(np.int64))
    np.testing.assert_array_equal(hwm.labels(pt, k), HAMMING_WEIGHT[SBOX[pt ^ k]])


def test_hypothesis_labels_match_constant_key() -> None:
    model = LeakageModel(leakage="id", byte=1)
    pt = (np.arange(20 * 16, dtype=np.uint8).reshape(20, 16))
    hyps = model.hypothesis_labels(pt[:, 1])
    assert hyps.shape == (20, 256)
    for k in (0, 7, 200, 255):
        np.testing.assert_array_equal(hyps[:, k], model.labels(pt[:, 1], k))


def test_dataset_helpers() -> None:
    n = 8
    pt = (np.arange(n * 16, dtype=np.uint8).reshape(n, 16))
    key = np.tile(np.arange(16, dtype=np.uint8), (n, 1))  # fixed key 0..15
    ds = ArraySideChannelDataset(
        name="t", traces=np.zeros((n, 4), np.float32), plaintexts=pt, keys=key
    )
    model = LeakageModel(leakage="id", byte=3)
    np.testing.assert_array_equal(
        model.dataset_labels(ds), SBOX[pt[:, 3] ^ 3].astype(np.int64)
    )
    assert model.true_key_byte(ds) == 3
    assert model.dataset_hypotheses(ds).shape == (n, 256)


# --- metrics ---------------------------------------------------------------

def _attack(n: int, key: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    model = LeakageModel(leakage="id", byte=0)
    labels = model.labels(pt[:, 0], key)          # true class per trace
    hyps = model.hypothesis_labels(pt[:, 0])       # (n, 256)
    return labels, hyps


def test_leaky_model_breaks_the_key() -> None:
    n, key = 300, 42
    labels, hyps = _attack(n, key)
    # A informative model: most probability mass on the correct class.
    probs = np.full((n, 256), 0.001)
    probs[np.arange(n), labels] = 0.744
    probs /= probs.sum(axis=1, keepdims=True)

    result = evaluate_key_rank(probs, hyps, key, n_experiments=50, seed=0)
    assert result.guessing_entropy[-1] == 0.0        # correct key ranked first
    assert result.success_rate[-1] == 1.0
    ttd = result.traces_to_disclosure()
    assert ttd is not None and ttd < n


def test_uninformative_model_does_not_break_key() -> None:
    n, key = 300, 42
    _labels, hyps = _attack(n, key)
    probs = np.full((n, 256), 1.0 / 256)             # uniform → no information
    ge = guessing_entropy(probs, hyps, key, n_experiments=50, seed=0)
    assert ge[-1] > 50                               # stays near random (~127)


def test_metrics_are_deterministic() -> None:
    n, key = 120, 7
    labels, hyps = _attack(n, key)
    probs = np.full((n, 256), 0.002)
    probs[np.arange(n), labels] = 0.5
    a = evaluate_key_rank(probs, hyps, key, seed=3)
    b = evaluate_key_rank(probs, hyps, key, seed=3)
    np.testing.assert_array_equal(a.guessing_entropy, b.guessing_entropy)
    np.testing.assert_array_equal(a.success_rate, b.success_rate)


# --- new leakage models: sbox_in target + single-bit transform -------------

def test_sbox_in_target() -> None:
    pt = np.array([0x53, 0x10, 0xff], np.uint8)
    key = np.uint8(0xE0)
    # sbox_in is the AddRoundKey output (before SubBytes): pt ^ k.
    m = LeakageModel(target="sbox_in", leakage="id", byte=0)
    np.testing.assert_array_equal(m.labels(pt, key), (pt ^ key).astype(np.int64))
    # vs sbox_out which applies the S-box.
    out = LeakageModel(target="sbox_out", leakage="id").labels(pt, key)
    np.testing.assert_array_equal(out, SBOX[pt ^ key].astype(np.int64))
    # hypotheses: column g is pt ^ g
    hyp = m.hypothesis_labels(pt)
    assert hyp.shape == (3, 256)
    np.testing.assert_array_equal(hyp[:, 0x11], (pt ^ 0x11).astype(np.int64))


def test_bit_leakage_transform() -> None:
    pt = np.arange(20, dtype=np.uint8)
    key = np.uint8(7)
    inter = SBOX[pt ^ key]
    for bit in (0, 3, 7):
        m = LeakageModel(leakage="bit", bit=bit)
        assert m.n_classes == 2
        np.testing.assert_array_equal(m.labels(pt, key), ((inter >> bit) & 1).astype(np.int64))
    # sbox_in + bit takes the bit of pt^k directly
    m2 = LeakageModel(target="sbox_in", leakage="bit", bit=2)
    np.testing.assert_array_equal(m2.labels(pt, key), (((pt ^ key) >> 2) & 1).astype(np.int64))
    with pytest.raises(ValueError):
        LeakageModel(leakage="bit", bit=8)
    with pytest.raises(ValueError):
        LeakageModel(target="nope")


def test_bit_leakage_on_last_round_hd() -> None:
    # The bit transform composes with any target, including last_round_hd.
    m = LeakageModel(target="last_round_hd", leakage="bit", bit=0, byte=3)
    assert m.n_classes == 2
    ct = np.array([[i % 256 for i in range(16)]], np.uint8)
    # one attack trace -> hypotheses over 256 last-round subkey guesses, all 0/1
    hyp = m.hd_hypothesis_labels(ct[:, 3], ct[:, 11])
    assert hyp.shape == (1, 256)
    assert set(np.unique(hyp)).issubset({0, 1})


@pytest.mark.parametrize("target,leak", [("sbox_in", "id"), ("sbox_out", "bit"), ("sbox_in", "hw")])
def test_new_models_recover_key(target: str, leak: str) -> None:
    rng = np.random.default_rng(0)
    n, key = 4000, 123
    m = LeakageModel(target=target, leakage=leak, byte=0, bit=0)
    pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    labels = m.labels(pt[:, 0], np.uint8(key))
    probs = np.eye(m.n_classes, dtype=np.float64)[labels]   # perfect model
    hyp = m.hypothesis_labels(pt[:, 0])
    res = evaluate_key_rank(probs, hyp, key, n_experiments=30, seed=0)
    assert res.guessing_entropy[-1] < 5.0


def test_plots_accept_single_results_and_dicts(tmp_path):
    pytest.importorskip("matplotlib")
    from mlsca_bench.benchmark import plots
    from mlsca_bench.benchmark.metrics import RankResult

    ranks = RankResult(n_traces=np.arange(1, 11), guessing_entropy=np.linspace(100, 0, 10),
                       success_rate=np.linspace(0, 1, 10))

    class Result:                     # what run_attack / run_classical return: an object with .ranks
        def __init__(self, r):
            self.ranks = r

    plots.plot_guessing_entropy(Result(ranks), save=tmp_path / "ge.png")
    plots.plot_success_rate({"mlp": Result(ranks), "cpa": ranks}, save=tmp_path / "sr.png")
    assert (tmp_path / "ge.png").stat().st_size > 0 and (tmp_path / "sr.png").stat().st_size > 0
