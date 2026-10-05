# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Leakage models: map plaintext/key to the attacked intermediate and its label.

Standard profiled-SCA target: the AES round-1 S-box output ``Sbox[pt[b] ^ k[b]]``
for a chosen key byte ``b``. The label a model predicts is that intermediate
under a leakage transform: identity (256 classes) or Hamming weight (9 classes).

Everything here is pure NumPy. The two things downstream code needs:
* ``labels(...)`` — per-trace labels under the *true* key, to train a classifier.
* ``hypothesis_labels(...)`` — per-trace labels under *every* key guess (0..255),
  used by the ranking metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..datasets.base import SideChannelDataset

# AES S-box.
SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)

# Hamming weight of each byte value.
HAMMING_WEIGHT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

# Inverse AES S-box (SBOX_INV[SBOX[x]] == x).
SBOX_INV = np.zeros(256, dtype=np.uint8)
SBOX_INV[SBOX] = np.arange(256, dtype=np.uint8)

# ShiftRows byte permutation P (new[j] = old[P[j]], column-major state). The
# last-round Hamming-distance partner of ciphertext byte j is P[j] — e.g. byte
# 15 pairs with 11, matching the AES_HD dataset's HW(InvSbox[c15^k]^c11) model.
_SHIFTROWS = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.int64)

_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def aes128_round_keys(master_key: Any) -> np.ndarray:
    """Expand a 16-byte AES-128 master key into the 11 round keys ``(11, 16)``."""

    if isinstance(master_key, (bytes, bytearray, memoryview)):
        key = np.frombuffer(bytes(master_key), dtype=np.uint8)
    else:
        key = np.asarray(master_key, dtype=np.uint8).reshape(-1)
    key = key[:16]
    words = [[int(b) for b in key[4 * i : 4 * i + 4]] for i in range(4)]
    for rnd in range(10):
        temp = words[-1][1:] + words[-1][:1]            # RotWord
        temp = [int(SBOX[b]) for b in temp]             # SubWord
        temp[0] ^= _RCON[rnd]
        for i in range(4):
            base = words[-4]
            xor_with = temp if i == 0 else words[-1]
            words.append([base[j] ^ xor_with[j] for j in range(4)])
    return np.array([sum(words[4 * r : 4 * r + 4], []) for r in range(11)], dtype=np.uint8)


def _xtime(a: np.ndarray) -> np.ndarray:
    """GF(2^8) multiply-by-2 (xtime), vectorised over a uint8 array."""

    a = a.astype(np.uint16)
    return (((a << 1) ^ ((a >> 7) * 0x1B)) & 0xFF).astype(np.uint8)


def aes_round1_output(plaintext: Any, round_key0: Any) -> np.ndarray:
    """AES round-1 output state ``MixColumns(ShiftRows(Sbox[pt ⊕ RK0]))``.

    ``plaintext`` is ``(n, 16)`` and ``round_key0`` the 16-byte round key 0
    (column-major, the standard AES state order); returns ``(n, 16)``. This is
    the value XORed with round key 1 at the start of round 2 — the pivot for the
    AES-256 second-half key recovery.
    """

    pt = np.asarray(plaintext, dtype=np.uint8)[:, :16]
    rk0 = np.asarray(round_key0, dtype=np.uint8).reshape(-1)[:16]
    state = SBOX[pt ^ rk0]                              # AddRoundKey + SubBytes
    state = state[:, _SHIFTROWS]                        # ShiftRows
    out = np.empty_like(state)
    for c in range(4):
        s0, s1, s2, s3 = (state[:, 4 * c + r] for r in range(4))
        out[:, 4 * c + 0] = _xtime(s0) ^ (_xtime(s1) ^ s1) ^ s2 ^ s3
        out[:, 4 * c + 1] = s0 ^ _xtime(s1) ^ (_xtime(s2) ^ s2) ^ s3
        out[:, 4 * c + 2] = s0 ^ s1 ^ _xtime(s2) ^ (_xtime(s3) ^ s3)
        out[:, 4 * c + 3] = (_xtime(s0) ^ s0) ^ s1 ^ s2 ^ _xtime(s3)
    return out


@dataclass(frozen=True)
class LeakageModel:
    """AES leakage model for one key byte.

    ``target`` selects the intermediate:

    * ``sbox_out`` — round-1 S-box output ``Sbox[pt[b] ^ k[b]]`` (from the
      plaintext); the recovered key byte is the master-key byte ``b``.
    * ``sbox_in`` — round-1 S-box *input* ``pt[b] ^ k[b]`` (from the plaintext),
      i.e. the AddRoundKey output before SubBytes.
    * ``last_round_hd`` — the last-round register Hamming distance
      ``HW(InvSbox[ct[b] ^ rk[b]] ^ ct[P[b]])`` (from the ciphertext), the
      standard model for hardware AES (AES-HD, DPA Contest v2). The recovered
      key byte is the *last-round* subkey byte ``b``; ``true_key_byte`` derives
      it from the stored master key via the AES key schedule.
    * ``sbox_out_r2`` — the **round-2** S-box output ``Sbox[R1_out[b] ^ RK1[b]]``,
      for **AES-256 second-half key recovery**. ``R1_out`` is the round-1 output
      (needs the full plaintext), computed from the recovered round key 0 — pass
      it as ``round0_key`` (the 16-byte RK0 = AES-256 master bytes 0-15). The
      recovered byte is round-key-1 byte ``b`` = master byte ``16 + b``, so
      running all 8+8 stages yields the full 256-bit key with no key-schedule
      inversion (for AES-256, RK0‖RK1 *is* the master key).

    ``leakage`` maps the intermediate to a label: ``id`` (identity, 256 classes),
    ``hw`` (Hamming weight, 9 classes), or ``bit`` (a single bit ``self.bit``,
    2 classes). ``byte`` is the attacked key-byte index; ``bit`` selects the bit
    for the ``bit`` leakage (0 = LSB).
    """

    target: str = "sbox_out"
    leakage: str = "id"
    byte: int = 0
    bit: int = 0
    round0_key: Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.target not in ("sbox_out", "sbox_in", "last_round_hd", "sbox_out_r2"):
            raise ValueError(
                f"Unsupported target {self.target!r} "
                "(use 'sbox_out', 'sbox_in', 'last_round_hd', or 'sbox_out_r2')."
            )
        if self.leakage not in ("id", "hw", "bit"):
            raise ValueError(
                f"Unsupported leakage {self.leakage!r} (use 'id', 'hw', or 'bit')."
            )
        if not 0 <= self.bit <= 7:
            raise ValueError("bit must be in [0, 7].")
        if isinstance(self.byte, bool) or not isinstance(self.byte, (int, np.integer)) or not 0 <= self.byte <= 15:
            raise ValueError(f"byte must be an AES key-byte index from 0 to 15; got {self.byte!r}.")

    @property
    def n_classes(self) -> int:
        return {"id": 256, "hw": 9, "bit": 2}[self.leakage]

    def _transform(self, intermediate: np.ndarray) -> np.ndarray:
        if self.leakage == "id":
            return intermediate.astype(np.int64)
        if self.leakage == "hw":
            return HAMMING_WEIGHT[intermediate].astype(np.int64)
        return ((intermediate.astype(np.int64) >> self.bit) & 1)

    def _intermediate(self, value: np.ndarray) -> np.ndarray:
        """S-box input ``pt^k`` (sbox_in) or its S-box output (sbox_out)."""

        value = value.astype(np.uint8)
        return value if self.target == "sbox_in" else SBOX[value]

    def labels(self, plaintext_byte: Any, key_byte: Any) -> np.ndarray:
        """Labels under the given (true) key byte(s). Shapes broadcast."""

        data = np.asarray(plaintext_byte, dtype=np.uint8)
        key = np.asarray(key_byte, dtype=np.uint8)
        return self._transform(self._intermediate((data ^ key)))

    def hypothesis_labels(self, plaintext_byte: Any) -> np.ndarray:
        """Labels for every key guess 0..255 → shape ``(n_traces, 256)``."""

        data = np.asarray(plaintext_byte, dtype=np.uint8)
        guesses = np.arange(256, dtype=np.uint8)
        return self._transform(self._intermediate(data[:, None] ^ guesses[None, :]))

    # --- last-round Hamming-distance (hardware AES) ----------------------

    def hd_labels(self, ct_byte: Any, ct_partner: Any, subkey_byte: Any) -> np.ndarray:
        """Last-round HD label(s) ``InvSbox[ct[b]^rk] ^ ct[P[b]]`` (transformed)."""

        c = np.asarray(ct_byte, dtype=np.uint8)
        cp = np.asarray(ct_partner, dtype=np.uint8)
        rk = np.asarray(subkey_byte, dtype=np.uint8)
        return self._transform(SBOX_INV[(c ^ rk).astype(np.uint8)] ^ cp)

    def hd_hypothesis_labels(self, ct_byte: Any, ct_partner: Any) -> np.ndarray:
        """Last-round HD labels for every last-round subkey guess → ``(n, 256)``."""

        c = np.asarray(ct_byte, dtype=np.uint8)
        cp = np.asarray(ct_partner, dtype=np.uint8)
        guesses = np.arange(256, dtype=np.uint8)
        intermediate = SBOX_INV[(c[:, None] ^ guesses[None, :])] ^ cp[:, None]
        return self._transform(intermediate)

    # --- dataset convenience --------------------------------------------

    def _byte_column(self, field: Any, name: str, offset: int = 0) -> np.ndarray:
        if field is None:
            raise ValueError(f"Dataset has no {name}; cannot build labels.")
        column = np.asarray(field[:])
        if column.ndim != 2:
            raise ValueError(f"Expected 2-D {name}; got shape {column.shape}.")
        return column[:, offset + self.byte].astype(np.uint8)

    def _ct_pair(self, dataset: SideChannelDataset) -> tuple[np.ndarray, np.ndarray]:
        ct = dataset.ciphertexts
        if ct is None:
            raise ValueError("Dataset has no ciphertexts; needed for last_round_hd.")
        ct = np.asarray(ct[:]).astype(np.uint8)
        partner = int(_SHIFTROWS[self.byte])
        return ct[:, self.byte], ct[:, partner]

    def _last_round_key_byte(self, dataset: SideChannelDataset) -> int:
        if dataset.keys is None:
            # Some releases (e.g. AES-HD) ship no key but document the attacked
            # last-round key byte(s); the dataset declares them in its metadata.
            metadata = getattr(dataset, "metadata", None) or {}
            known = dict(metadata.get("last_round_key_bytes", {})) if hasattr(metadata, "get") else {}
            if self.byte in known:
                return int(known[self.byte])
            hint = (
                f" This dataset's known last-round key byte(s): {sorted(known)}; "
                f"use LeakageModel(target='last_round_hd', byte={sorted(known)[0]})."
                if known else ""
            )
            raise ValueError(
                f"Dataset has no keys; cannot derive last-round key byte {self.byte}.{hint}"
            )
        master = np.asarray(dataset.keys[0]).astype(np.uint8).reshape(-1)[:16]
        return int(aes128_round_keys(master)[10][self.byte])

    def _round1_state_byte(self, dataset: SideChannelDataset) -> np.ndarray:
        """Round-1 output byte ``self.byte`` from the full plaintext + RK0."""

        if self.round0_key is None:
            raise ValueError(
                "target='sbox_out_r2' needs round0_key: the recovered 16-byte "
                "round key 0 (AES-256 master bytes 0-15)."
            )
        if dataset.plaintexts is None:
            raise ValueError("Dataset has no plaintexts; needed for sbox_out_r2.")
        pt = np.asarray(dataset.plaintexts[:]).astype(np.uint8)
        if pt.ndim != 2 or pt.shape[1] < 16:
            raise ValueError("sbox_out_r2 needs full 16-byte plaintexts.")
        return aes_round1_output(pt, self.round0_key)[:, self.byte]

    def dataset_labels(
        self, dataset: SideChannelDataset, key: int | None = None
    ) -> np.ndarray:
        """Per-trace training labels from the dataset's public values and key."""

        if self.target == "last_round_hd":
            ct_b, ct_p = self._ct_pair(dataset)
            rk = key if key is not None else self._last_round_key_byte(dataset)
            return self.hd_labels(ct_b, ct_p, np.full(len(ct_b), int(rk), np.uint8))

        if self.target == "sbox_out_r2":
            r1b = self._round1_state_byte(dataset)
            if key is not None:
                rk1 = np.full(len(r1b), int(key), dtype=np.uint8)
            else:
                rk1 = self._byte_column(dataset.keys, "keys", offset=16)
            return self._transform(SBOX[(r1b ^ rk1).astype(np.uint8)])

        plaintext = self._byte_column(dataset.plaintexts, "plaintexts")
        if key is None:
            key_bytes = self._byte_column(dataset.keys, "keys")
        else:
            key_bytes = np.full(len(plaintext), int(key), dtype=np.uint8)
        return self.labels(plaintext, key_bytes)

    def dataset_hypotheses(self, dataset: SideChannelDataset) -> np.ndarray:
        """Per-trace labels for all 256 key guesses on an attack dataset."""

        if self.target == "last_round_hd":
            ct_b, ct_p = self._ct_pair(dataset)
            return self.hd_hypothesis_labels(ct_b, ct_p)
        if self.target == "sbox_out_r2":
            r1b = self._round1_state_byte(dataset)
            guesses = np.arange(256, dtype=np.uint8)
            return self._transform(SBOX[(r1b[:, None] ^ guesses[None, :]).astype(np.uint8)])
        return self.hypothesis_labels(self._byte_column(dataset.plaintexts, "plaintexts"))

    def true_key_byte(self, dataset: SideChannelDataset) -> int:
        """The attacked key byte (last-round subkey for ``last_round_hd``)."""

        if self.target == "last_round_hd":
            return self._last_round_key_byte(dataset)
        if self.target == "sbox_out_r2":
            return int(self._byte_column(dataset.keys, "keys", offset=16)[0])
        return int(self._byte_column(dataset.keys, "keys")[0])

__all__ = [
    "SBOX",
    "SBOX_INV",
    "HAMMING_WEIGHT",
    "LeakageModel",
    "aes128_round_keys",
    "aes_round1_output",
]
