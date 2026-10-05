# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for dataset wirings corrected by the verification sweep.

- DTDS Dilithium3/5 store fields in parallel per-chunk directory trees
  (traces under Traces/NNN, labels under Median/NNN) with a different label
  filename than Dilithium2; exercised via the `field_subdirs` mechanism.
- X-DeepSCA .mat files carry `textin` (plaintext), which must be exposed so the
  attack pipeline can build S-box labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsca_bench.datasets import load_dataset


@pytest.fixture
def dilithium3_dir(tmp_path: Path) -> Path:
    """DTDS Dilithium3/5 nested layout: 3_Data/{Traces,Median}/NNN/<file>.npy."""
    rng = np.random.default_rng(0)
    base = tmp_path / "3_Data"
    for i in range(3):                       # three chunks
        n, ns = 20, 40
        tdir = base / "Traces" / f"{i:03d}"
        mdir = base / "Median" / f"{i:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        mdir.mkdir(parents=True, exist_ok=True)
        np.save(tdir / "polyz_unpack_traces.npy",
                rng.integers(-2000, 2000, size=(n, ns), dtype=np.int16))
        np.save(mdir / "polyvecl_uniform_gamma1_y.npy",
                rng.integers(0, 256, size=n, dtype=np.int64))
        # A sibling Sign/ dir with key material that must NOT be loaded.
        (base / "Sign" / f"{i:03d}").mkdir(parents=True, exist_ok=True)
        (base / "Sign" / f"{i:03d}" / "mate.req").write_bytes(b"\x00" * 8)
    return tmp_path


def test_dilithium3_nested_field_subdirs(dilithium3_dir: Path) -> None:
    with load_dataset("dtds-dilithium3", path=str(dilithium3_dir)) as ds:
        assert len(ds) == 60                 # 3 chunks x 20 traces, concatenated
        assert ds.shape == (60, 40)
        assert "labels" in ds.available_fields
        assert np.asarray(ds.labels[:]).shape == (60,)
        # traces and labels stay aligned across the parallel Traces/Median trees
        sample = ds[25]
        assert sample.trace.shape == (40,) and sample.label is not None


def test_dilithium2_flat_layout_still_works(tmp_path: Path) -> None:
    # Dilithium2 keeps the co-located layout (no field_subdirs) — must not regress.
    rng = np.random.default_rng(1)
    for i in range(2):
        d = tmp_path / f"{i:03d}"
        d.mkdir()
        np.save(d / "polyz_unpack_traces.npy", rng.integers(-100, 100, (10, 30), dtype=np.int16))
        np.save(d / "polyz_unpack_y.npy", rng.integers(0, 256, 10, dtype=np.int64))
    with load_dataset("dtds-dilithium2", path=str(tmp_path)) as ds:
        assert len(ds) == 20 and ds.shape == (20, 30)
        assert "labels" in ds.available_fields


def test_xdeepsca_exposes_plaintext(tmp_path: Path) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    rng = np.random.default_rng(2)
    mat = tmp_path / "cw308XGD2_10k_nov5_1447.mat"
    scipy_io.savemat(mat, {
        "traces": rng.standard_normal((50, 100)),
        "textin": rng.integers(0, 256, (50, 16), dtype=np.uint8),
        "key": np.tile(np.arange(16, dtype=np.uint8), (50, 1)),
    })
    with load_dataset("x-deepsca", path=str(mat)) as ds:
        assert len(ds) == 50
        assert "plaintexts" in ds.available_fields      # the fix: textin -> plaintexts
        assert np.asarray(ds.plaintexts[:]).shape == (50, 16)
        assert np.asarray(ds.keys[:]).shape == (50, 16)
