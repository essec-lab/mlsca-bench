# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Spook 2-way split: random-key profiling glob + fixed-key attack glob.

Builds synthetic npz files matching the real Spook layout (rkey files carry
traces/nonces/umsk_keys/msk_keys; fkey files carry only traces/nonces, with the
key supplied via the registry's `fixed_key`) and drives them through
load_dataset and make_splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import load_dataset
from mlsca_bench.benchmark.splits import make_splits

SW3_K0_HEX = "d11a61da644e5d360eefa2fec96c1115"


@pytest.fixture
def spook_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    ns = 32
    # random-key profiling file
    n = 40
    np.savez(
        tmp_path / "rkey_sw3_40_0.npz",
        traces=rng.integers(-2000, 2000, size=(n, ns), dtype=np.int16),
        nonces=rng.integers(0, 2**32, size=(n, 4)).astype(np.uint32),
        umsk_keys=rng.integers(0, 2**32, size=(n, 4)).astype(np.uint32),
        msk_keys=rng.integers(0, 2**32, size=(n, 12)).astype(np.uint32),
    )
    # fixed-key attack file: traces + nonces ONLY (no key field)
    m = 15
    np.savez(
        tmp_path / "fkey_sw3_K0_15_0.npz",
        traces=rng.integers(-2000, 2000, size=(m, ns), dtype=np.int16),
        nonces=rng.integers(0, 2**32, size=(m, 4)).astype(np.uint32),
    )
    return tmp_path


def test_profiling_split_loads_random_keys(spook_dir: Path) -> None:
    with load_dataset("ches-ctf-2020-spook-sw3", path=str(spook_dir), split="profiling") as ds:
        assert len(ds) == 40
        assert "keys" in ds.available_fields and "plaintexts" in ds.available_fields
        keys = np.asarray(ds.keys[:])
        assert keys.shape == (40, 4) and not (keys == keys[0]).all()   # random keys vary


def test_attack_split_injects_fixed_key(spook_dir: Path) -> None:
    with load_dataset("ches-ctf-2020-spook-sw3", path=str(spook_dir), split="attack") as ds:
        assert len(ds) == 15
        keys = np.asarray(ds.keys[:])
        assert keys.shape == (15, 4)
        expected = np.frombuffer(bytes.fromhex(SW3_K0_HEX), dtype="<u4")
        assert (keys == expected).all()                      # constant published key
        assert np.asarray(ds[0].key).tolist() == expected.tolist()
        # nonce (public input) is present; per-trace key matches too
        assert np.asarray(ds.plaintexts[:]).shape == (15, 4)


def test_make_splits_two_way(spook_dir: Path) -> None:
    with make_splits("ches-ctf-2020-spook-sw3", path=str(spook_dir)) as sp:
        assert len(sp.test) == 15                       # fixed-key attack set, untouched
        assert len(sp.train) + len(sp.val) == 40        # carved from random-key profiling
        expected_key = np.frombuffer(bytes.fromhex(SW3_K0_HEX), dtype="<u4")
        assert (np.asarray(sp.test.keys[:]) == expected_key).all()

HW2_K0_HEX = "90414ffdd837c411c5b06152ddaea7ff"


def _s16(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random 16-byte bytestrings, as the HW npz stores nonce/key (|S16)."""
    return np.array([rng.integers(0, 256, 16, dtype=np.uint8).tobytes() for _ in range(n)], dtype="S16")


@pytest.fixture
def spook_hw_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    ns = 32
    # HW random-key profiling: nonce/umsk_keys are |S16 bytestrings.
    n = 40
    np.savez(
        tmp_path / "rkey_hw2_40_0.npz",
        traces=rng.integers(-2000, 2000, size=(n, ns), dtype=np.int16),
        nonce=_s16(rng, n),
        umsk_keys=_s16(rng, n),
    )
    # HW fixed-key attack: traces + nonce (|S16) only.
    m = 15
    np.savez(
        tmp_path / "fkey_hw2_K0_15_0.npz",
        traces=rng.integers(-2000, 2000, size=(m, ns), dtype=np.int16),
        nonce=_s16(rng, m),
    )
    return tmp_path


def test_hw_bytestring_fields_become_uint32(spook_hw_dir: Path) -> None:
    with load_dataset("ches-ctf-2020-spook-hw2", path=str(spook_hw_dir), split="profiling") as ds:
        assert len(ds) == 40
        keys = np.asarray(ds.keys[:])
        nonce = np.asarray(ds.plaintexts[:])
        assert keys.shape == (40, 4) and keys.dtype == np.uint32
        assert nonce.shape == (40, 4) and nonce.dtype == np.uint32


def test_hw_make_splits_two_way(spook_hw_dir: Path) -> None:
    with make_splits("ches-ctf-2020-spook-hw2", path=str(spook_hw_dir)) as sp:
        assert len(sp.test) == 15
        assert len(sp.train) + len(sp.val) == 40
        expected = np.frombuffer(bytes.fromhex(HW2_K0_HEX), dtype="<u4")
        assert (np.asarray(sp.test.keys[:]) == expected).all()
