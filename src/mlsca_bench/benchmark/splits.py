# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible train/validation/test splits over registered datasets.

Policy — "honor, then fill":

* **3-way** native (train/test/holdout, e.g. SCAAML): used directly, holdout->val.
* **2-way** native (profiling/attack, or variable/fixed key): the author's attack
  set becomes ``test`` untouched; ``val`` is carved from the profiling/train pool
  (``val_ratio``); ``train`` is the remainder.
* **no native split**: the loaded pool is shuffled into train/val/test by
  ``ratios``.

Splits are sets of row indices exposed as lazy :class:`Subset` views (no data is
copied) and are deterministic given a seed, so the same call reproduces the same
split anywhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np

from ..datasets.base import DatasetArray, SideChannelDataset, TraceSample
from ..datasets.loading import load_dataset
from ..datasets.registry import DatasetSpec, get_dataset


@dataclass(frozen=True)
class SplitPolicy:
    """How to build splits. Defaults define the canonical benchmark split."""

    seed: int = 0
    # Fraction of the profiling/train pool reserved for validation (2-way case).
    val_ratio: float = 0.1
    # train/val/test fractions when no native split exists.
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1).")
        if len(self.ratios) != 3 or abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three fractions summing to 1.")


_CONFIG_CACHE: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load the pinned split config (splits.json); tolerate its absence."""

    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        try:
            text = files("mlsca_bench.benchmark").joinpath("splits.json").read_text(
                encoding="utf-8"
            )
            _CONFIG_CACHE = json.loads(text)
        except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError):
            _CONFIG_CACHE = {"default": {}, "datasets": {}}
    return _CONFIG_CACHE


def resolve_policy(name: str, config: dict[str, Any] | None = None) -> SplitPolicy:
    """The canonical :class:`SplitPolicy` for a dataset (default + overrides)."""

    cfg = config if config is not None else _load_config()
    merged: dict[str, Any] = dict(cfg.get("default", {}))
    merged.update(cfg.get("datasets", {}).get(name, {}))
    kwargs: dict[str, Any] = {}
    if "seed" in merged:
        kwargs["seed"] = int(merged["seed"])
    if "val_ratio" in merged:
        kwargs["val_ratio"] = float(merged["val_ratio"])
    if "ratios" in merged:
        kwargs["ratios"] = tuple(merged["ratios"])
    return SplitPolicy(**kwargs)


def _index_checksum(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()[:16]


def describe_splits(splits: "DatasetSplits") -> dict[str, Any]:
    """A compact, archivable manifest of a split (policy, sizes, checksums)."""

    return {
        "dataset": splits.train.name,
        "policy": {
            "seed": splits.policy.seed,
            "val_ratio": splits.policy.val_ratio,
            "ratios": list(splits.policy.ratios),
        },
        "sizes": {name: int(len(idx)) for name, idx in splits.indices.items()},
        "checksums": {
            name: _index_checksum(idx) for name, idx in splits.indices.items()
        },
    }


def verify_splits(splits: "DatasetSplits", manifest: dict[str, Any]) -> bool:
    """Return whether ``splits`` matches a previously-saved manifest."""

    current = describe_splits(splits)
    return current["sizes"] == manifest.get("sizes") and current[
        "checksums"
    ] == manifest.get("checksums")


def _native_layout(spec: DatasetSpec) -> tuple[str, tuple[str, ...] | None]:
    """Classify a dataset's native split as ('3-way'|'2-way'|'none', roles)."""

    adapter = spec.adapter
    config = spec.adapter_config or {}
    if adapter == "scaaml":
        return "3-way", ("train", "test", "holdout")
    if adapter in ("ascad", "aes-hd-zaid"):
        return "2-way", ("profiling", "attack")
    if adapter == "ascon-hdf5":
        # Profile on variable (random) keys, attack on the fixed key.
        return "2-way", ("random", "fixed")
    if adapter == "two-class-npy":
        return "2-way", ("train", "test")
    if adapter == "npy-manifest" and config.get("split_dirs"):
        # e.g. SMAesH: vk0 = variable-key profiling, fk0 = fixed-key attack.
        return "2-way", ("profiling", "attack")
    if adapter == "npz" and {"profiling", "attack"} <= set(config.get("splits", {})):
        # e.g. Spook: random-key profiling glob + fixed-key attack glob.
        return "2-way", ("profiling", "attack")
    if adapter in ("hdf5", "flat-hdf5"):
        keys = set(config.get("splits", {}))
        if {"profiling", "attack"} & keys and "all" not in keys:
            return "2-way", ("profiling", "attack")
    return "none", None


class _IndexedArray:
    """Lazy row-subset view of a parent field array (any DatasetArray)."""

    def __init__(self, base: DatasetArray, indices: np.ndarray) -> None:
        self._base = base
        self._idx = indices
        self._trailing = tuple(base.shape[1:])

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._idx), *self._trailing)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._base.dtype)

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            chosen = self._idx[index]
            if len(chosen) == 0:
                return np.empty((0, *self._trailing), dtype=self.dtype)
            # Element-wise gather keeps this correct for every backend
            # (h5py needs monotonic fancy indices; splits are stored sorted).
            return np.stack([np.asarray(self._base[int(j)]) for j in chosen])
        return np.asarray(self._base[int(self._idx[index])])


class Subset(SideChannelDataset):
    """A row-subset of a loaded dataset, itself a ``SideChannelDataset``.

    Lifetime is owned by the enclosing :class:`DatasetSplits`; ``close()`` here is
    a no-op so that sibling subsets sharing a parent stay usable.
    """

    def __init__(
        self,
        parent: SideChannelDataset,
        indices: Sequence[int] | np.ndarray,
        *,
        split_label: str | None = None,
    ) -> None:
        self._parent = parent
        self._idx = np.asarray(indices, dtype=np.int64)
        self._split = split_label

    def _view(self, field_name: str) -> DatasetArray | None:
        value = getattr(self._parent, field_name)
        return None if value is None else _IndexedArray(value, self._idx)

    @property
    def name(self) -> str:
        return self._parent.name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> DatasetArray:
        return _IndexedArray(self._parent.traces, self._idx)

    @property
    def plaintexts(self) -> DatasetArray | None:
        return self._view("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        return self._view("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        return self._view("keys")

    @property
    def masks(self) -> DatasetArray | None:
        return self._view("masks")

    @property
    def labels(self) -> DatasetArray | None:
        return self._view("labels")

    @property
    def metadata(self) -> Any:
        return self._parent.metadata

    @property
    def closed(self) -> bool:
        return self._parent.closed

    def close(self) -> None:  # lifetime managed by DatasetSplits
        pass

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, index: int | slice) -> TraceSample | Sequence[TraceSample]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Split index out of range.")
        return self._parent[int(self._idx[index])]


@dataclass
class DatasetSplits:
    """The train/validation/test views plus the exact indices and policy used."""

    train: SideChannelDataset
    val: SideChannelDataset
    test: SideChannelDataset
    policy: SplitPolicy
    indices: dict[str, np.ndarray]
    _owned: tuple[SideChannelDataset, ...] = field(default_factory=tuple)

    def __enter__(self) -> "DatasetSplits":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        seen: set[int] = set()
        for dataset in self._owned:
            if id(dataset) not in seen:
                seen.add(id(dataset))
                dataset.close()


def _partition(n: int, policy: SplitPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.random.default_rng(policy.seed).permutation(n)
    train_frac, val_frac, _ = policy.ratios
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train : n_train + n_val])
    test = np.sort(perm[n_train + n_val :])
    return train, val, test


def _carve_val(n: int, policy: SplitPolicy) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(policy.seed).permutation(n)
    n_val = int(round(policy.val_ratio * n))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def make_splits(
    name: str,
    *,
    policy: SplitPolicy | None = None,
    path: str | None = None,
    **load_kwargs: Any,
) -> DatasetSplits:
    """Build reproducible train/validation/test splits for a registered dataset.

    ``path``/``load_kwargs`` are forwarded to :func:`load_dataset` (e.g. to point
    at a local copy). Use the result as a context manager to release files.
    """

    policy = policy or resolve_policy(name)
    spec = get_dataset(name)
    kind, roles = _native_layout(spec)

    if kind == "3-way":
        assert roles is not None
        train_role, test_role, val_role = roles
        d_train = load_dataset(name, split=train_role, path=path, **load_kwargs)
        d_val = load_dataset(name, split=val_role, path=path, **load_kwargs)
        d_test = load_dataset(name, split=test_role, path=path, **load_kwargs)
        idx = {
            "train": np.arange(len(d_train)),
            "val": np.arange(len(d_val)),
            "test": np.arange(len(d_test)),
        }
        return DatasetSplits(
            Subset(d_train, idx["train"], split_label="train"),
            Subset(d_val, idx["val"], split_label="val"),
            Subset(d_test, idx["test"], split_label="test"),
            policy,
            idx,
            (d_train, d_val, d_test),
        )

    if kind == "2-way":
        assert roles is not None
        train_role, test_role = roles
        pool = load_dataset(name, split=train_role, path=path, **load_kwargs)
        d_test = load_dataset(name, split=test_role, path=path, **load_kwargs)
        train_idx, val_idx = _carve_val(len(pool), policy)
        test_idx = np.arange(len(d_test))
        idx = {"train": train_idx, "val": val_idx, "test": test_idx}
        return DatasetSplits(
            Subset(pool, train_idx, split_label="train"),
            Subset(pool, val_idx, split_label="val"),
            Subset(d_test, test_idx, split_label="test"),
            policy,
            idx,
            (pool, d_test),
        )

    pool = load_dataset(name, path=path, **load_kwargs)
    train_idx, val_idx, test_idx = _partition(len(pool), policy)
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}
    return DatasetSplits(
        Subset(pool, train_idx, split_label="train"),
        Subset(pool, val_idx, split_label="val"),
        Subset(pool, test_idx, split_label="test"),
        policy,
        idx,
        (pool,),
    )


__all__ = [
    "DatasetSplits",
    "SplitPolicy",
    "Subset",
    "describe_splits",
    "make_splits",
    "resolve_policy",
    "verify_splits",
]
