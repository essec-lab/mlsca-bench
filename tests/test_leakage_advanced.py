# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AES leakage targets beyond the first-round S-box output:
the last-round Hamming distance (DPA Contest v2 / AES-HD) and AES-256.

They are checked against FIPS-197 and end to end via key recovery on
synthetic leaky data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsca_bench.benchmark.leakage import (
    HAMMING_WEIGHT,
    SBOX,
    SBOX_INV,
    LeakageModel,
    aes128_round_keys,
    aes_round1_output,
)
from mlsca_bench.benchmark.metrics import evaluate_key_rank
from mlsca_bench.datasets.base import ArraySideChannelDataset


# --------------------------------------------------------------------------
# AES last-round Hamming-distance model (DPA Contest v2 / AES-HD)
# --------------------------------------------------------------------------
def test_aes_key_schedule_fips197() -> None:
    master = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    rks = aes128_round_keys(master)
    assert bytes(rks[0]).hex() == master.hex()
    # FIPS-197 Appendix A.1 final (round-10) key.
    assert bytes(rks[10]).hex() == "d014f9a8c9ee2589e13f0cc8b6630ca6"
    assert (SBOX_INV[SBOX] == np.arange(256)).all()


def test_hd_label_formula() -> None:
    model = LeakageModel(target="last_round_hd", leakage="hw", byte=15)
    # HW(InvSbox[ct15 ^ rk] ^ ct11), partner of 15 is 11.
    ct15, ct11, rk = np.uint8(0x3C), np.uint8(0xA5), np.uint8(0x11)
    expected = HAMMING_WEIGHT[SBOX_INV[ct15 ^ rk] ^ ct11]
    got = model.hd_labels(ct15, ct11, rk)
    assert int(got) == int(expected)


def test_hd_recovers_last_round_key() -> None:
    rng = np.random.default_rng(0)
    n, target = 4000, 5
    master = rng.integers(0, 256, size=16, dtype=np.uint8)
    ct = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    ds = ArraySideChannelDataset(
        name="hd", traces=np.zeros((n, 1), np.float32),
        ciphertexts=ct, keys=np.tile(master, (n, 1)),
    )
    model = LeakageModel(target="last_round_hd", leakage="id", byte=target)
    true_rk = model.true_key_byte(ds)
    assert true_rk == int(aes128_round_keys(master)[10][target])
    # A perfect model: one-hot of the true HD label -> correct key ranks first.
    labels = model.dataset_labels(ds)
    probs = np.eye(256, dtype=np.float64)[labels]
    hyp = model.dataset_hypotheses(ds)
    res = evaluate_key_rank(probs, hyp, true_rk, n_experiments=20, seed=0)
    assert res.guessing_entropy[-1] < 1.0


def test_hd_requires_ciphertext() -> None:
    ds = ArraySideChannelDataset(
        name="x", traces=np.zeros((3, 1), np.float32),
        plaintexts=np.zeros((3, 16), np.uint8), keys=np.zeros((3, 16), np.uint8),
    )
    with pytest.raises(ValueError, match="no ciphertexts"):
        LeakageModel(target="last_round_hd").dataset_labels(ds)


# --------------------------------------------------------------------------
def test_aes_round1_output_matches_fips197() -> None:
    # FIPS-197 Appendix B: state after round-1 MixColumns.
    pt = np.array([[0x32, 0x43, 0xF6, 0xA8, 0x88, 0x5A, 0x30, 0x8D,
                    0x31, 0x31, 0x98, 0xA2, 0xE0, 0x37, 0x07, 0x34]], np.uint8)
    rk0 = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
                    0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], np.uint8)
    expected = [0x04, 0x66, 0x81, 0xE5, 0xE0, 0xCB, 0x19, 0x9A,
                0x48, 0xF8, 0xD3, 0x7A, 0x28, 0x06, 0x26, 0x4C]
    assert aes_round1_output(pt, rk0)[0].tolist() == expected


def test_sbox_out_r2_validation_and_true_key_byte() -> None:
    with pytest.raises(ValueError):
        LeakageModel(target="bogus")
    rng = np.random.default_rng(0)
    n = 8
    key = rng.integers(0, 256, size=32, dtype=np.uint8)          # AES-256 master
    pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    ds = ArraySideChannelDataset(name="aes256", traces=np.zeros((n, 1), np.float32),
                                 plaintexts=pt, keys=np.tile(key, (n, 1)))
    # round0_key is required to form round-2 hypotheses
    with pytest.raises(ValueError):
        LeakageModel(target="sbox_out_r2", byte=0).dataset_hypotheses(ds)
    m = LeakageModel(target="sbox_out_r2", byte=5, round0_key=key[:16])
    assert m.true_key_byte(ds) == int(key[16 + 5])              # RK1 byte = master[21]
    assert m.dataset_hypotheses(ds).shape == (n, 256)


def test_aes256_recovers_full_key_both_stages() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    key = rng.integers(0, 256, size=32, dtype=np.uint8)          # 32-byte master
    pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    ds = ArraySideChannelDataset(name="aes256", traces=np.zeros((n, 1), np.float32),
                                 plaintexts=pt, keys=np.tile(key, (n, 1)))

    def ge(model: LeakageModel, true: int) -> float:
        labels = model.dataset_labels(ds)
        probs = np.eye(model.n_classes, dtype=np.float64)[labels]
        hyp = model.dataset_hypotheses(ds)
        return evaluate_key_rank(probs, hyp, true, n_experiments=10, seed=0).guessing_entropy[-1]

    # Stage 1: round key 0 = master bytes 0-15 (round-1 S-box).
    for b in range(16):
        assert ge(LeakageModel(target="sbox_out", byte=b), int(key[b])) < 2.0
    # Stage 2: round key 1 = master bytes 16-31 (round-2 S-box, RK0 known).
    for b in range(16):
        m = LeakageModel(target="sbox_out_r2", byte=b, round0_key=key[:16])
        assert ge(m, int(key[16 + b])) < 2.0



def _keyless_hd_dataset():
    """A dataset without keys that declares its attacked last-round key byte (as AES-HD does)."""

    from mlsca_bench.datasets.base import ArraySideChannelDataset

    rng = np.random.default_rng(0)
    return ArraySideChannelDataset(
        name="keyless-hd",
        traces=rng.normal(size=(20, 5)).astype(np.float32),
        ciphertexts=rng.integers(0, 256, (20, 16), dtype=np.uint8),
        metadata={"last_round_key_bytes": {11: 0x2A}},
    )


def test_declared_last_round_key_byte_is_used():
    lm = LeakageModel(target="last_round_hd", byte=11)
    ds = _keyless_hd_dataset()
    assert lm.true_key_byte(ds) == 0x2A
    assert lm.dataset_labels(ds).shape == (20,)


def test_undeclared_byte_error_names_the_known_ones():
    with pytest.raises(ValueError, match=r"known last-round key byte\(s\): \[11\].*byte=11"):
        LeakageModel(target="last_round_hd", byte=0).true_key_byte(_keyless_hd_dataset())
