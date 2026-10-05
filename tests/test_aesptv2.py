# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the AES-PTv2 wiring, corrected after inspecting the real
(RAR-extracted) HDF5.

The real files nest every split under a single per-device top-level group
(/D1/Unprotected/Profiling, /D1/MS1/Attack, ...), store traces/labels as
`Traces`/`Labels`, and carry plaintext/key/masks in a compound dataset named
`MetaData` (columns `plaintext`/`key`/`masks`). This builds a small file with
that shape and loads it through the registry entries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mlsca_bench import load_dataset


def _make_group(parent, n, ns, *, with_masks: bool) -> None:
    parent.create_dataset("Traces", data=np.zeros((n, ns), dtype=np.float32))
    parent.create_dataset("Labels", data=np.arange(n, dtype=np.int32) % 256)
    cols = [("plaintext", "u1", (16,)), ("key", "u1", (16,))]
    if with_masks:
        cols.append(("masks", "u1", (2,)))
    md = np.zeros(n, dtype=np.dtype(cols))
    md["plaintext"] = np.arange(n * 16, dtype=np.uint8).reshape(n, 16)
    md["key"] = np.tile(np.arange(16, dtype=np.uint8), (n, 1))
    parent.create_dataset("MetaData", data=md)


@pytest.fixture
def aesptv2_file(tmp_path: Path) -> Path:
    path = tmp_path / "AES_PTv2_D1.h5"
    with h5py.File(path, "w") as f:
        dev = f.create_group("D1")               # single per-device top-level group
        for scheme, masks in (("Unprotected", False), ("MS1", True), ("MS2", True)):
            g = dev.create_group(scheme)
            _make_group(g.create_group("Profiling"), 30, 20, with_masks=masks)
            _make_group(g.create_group("Attack"), 12, 20, with_masks=masks)
    return path


def test_aesptv2_unprotected_loads(aesptv2_file: Path) -> None:
    # Device-agnostic split path ("Unprotected/Attack") + auto device-group prefix.
    with load_dataset("aes-ptv2-stm32f4", path=str(aesptv2_file), split="attack") as ds:
        assert len(ds) == 12 and ds.shape == (12, 20)
        assert ds.available_fields == ("traces", "plaintexts", "keys", "labels")
        assert np.asarray(ds.keys[:]).shape == (12, 16)
        assert ds.masks is None                    # Unprotected has no masks column


def test_aesptv2_masked_loads_with_masks(aesptv2_file: Path) -> None:
    with load_dataset("aes-ptv2-stm32f4-ms1", path=str(aesptv2_file), split="profiling") as ds:
        assert len(ds) == 30
        assert "masks" in ds.available_fields
        assert np.asarray(ds.masks[:]).shape == (30, 2)

    with load_dataset("aes-ptv2-stm32f4-ms2", path=str(aesptv2_file), split="attack") as ds:
        assert len(ds) == 12
        assert np.asarray(ds.plaintexts[:]).shape == (12, 16)


def test_aesptv2_entries_are_automatic_rar(aesptv2_file: Path) -> None:
    # The gdrive files are RAR archives; extraction is handled by the RAR
    # processor (needs unar/unrar), so the entries are automatic.
    from mlsca_bench.datasets.registry import get_dataset

    for name in ("aes-ptv2-pinata", "aes-ptv2-stm32f4", "aes-ptv2-stm32f4-ms1"):
        spec = get_dataset(name)
        assert spec.availability == "automatic"
        assert all(f.archive == "rar" for f in spec.files)
