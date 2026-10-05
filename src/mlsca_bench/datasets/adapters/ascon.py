# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lazy adapter for the official Ascon side-channel HDF5 schema."""

from __future__ import annotations

from pathlib import Path

from .hdf5 import HDF5CompoundDataset


class AsconHDF5Dataset(HDF5CompoundDataset):
    """Read fixed- or random-key Ascon traces from the official HDF5 file.

    The Ascon release stores each key regime in a ``fixed_keys`` or
    ``random_keys`` group, with a top-level ``labels`` dataset alongside the
    ``traces`` and a compound ``metadata`` record. Nonce, associated-data, and
    tag columns that are not modelled as bulk fields surface as per-sample
    metadata.
    """

    _SPLITS = {
        "fixed": "fixed_keys",
        "random": "random_keys",
    }
    # The nonce is the public input of the initialisation phase, so it is
    # surfaced as ``plaintexts`` (as Spook does for its nonce); the AEAD message
    # plaintext, if present, is the fallback.
    _FIELD_ALIASES = {
        "plaintexts": ("nonce", "npub", "nonces", "plaintext", "plaintexts"),
        "ciphertexts": ("ciphertext", "ciphertexts"),
        "keys": ("key", "keys"),
    }

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "ascon-cw-unprotected",
        split: str = "fixed",
    ) -> None:
        super().__init__(
            path,
            name=name,
            splits=self._SPLITS,
            split=split,
            field_aliases=self._FIELD_ALIASES,
            labels_dataset="labels",
            require_labels=True,
            algorithm="Ascon",
            label="Ascon",
        )


__all__ = ["AsconHDF5Dataset"]
