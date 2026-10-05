# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lazy adapter for ASCAD-compatible HDF5 files."""

from __future__ import annotations

from pathlib import Path

from .hdf5 import HDF5CompoundDataset


class ASCADDataset(HDF5CompoundDataset):
    """Read an ASCAD split lazily from an HDF5 file.

    ASCAD stores each split in a ``Profiling_traces`` or ``Attack_traces``
    group with a two-dimensional ``traces`` dataset and a compound ``metadata``
    record holding the plaintext, key, and mask columns. The same layout is
    shared by ASCAD-derived datasets such as GE Wars.
    """

    _SPLITS = {
        "profiling": "Profiling_traces",
        "attack": "Attack_traces",
    }
    _FIELD_ALIASES = {
        "plaintexts": ("plaintext", "plaintexts"),
        "ciphertexts": ("ciphertext", "ciphertexts"),
        "keys": ("key", "keys"),
        "masks": ("masks", "mask"),
    }

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "ascadf",
        split: str = "profiling",
    ) -> None:
        super().__init__(
            path,
            name=name,
            splits=self._SPLITS,
            split=split,
            field_aliases=self._FIELD_ALIASES,
            labels_dataset=None,
            label="ASCAD",
        )


__all__ = ["ASCADDataset"]
