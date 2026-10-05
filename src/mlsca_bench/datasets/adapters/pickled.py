# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in adapter for datasets distributed as Python pickle files.

.. danger::

    Loading a pickle **executes arbitrary Python code** embedded in the file
    (via ``__reduce__``). A malicious pickle can run any command the moment it
    is opened. This adapter therefore refuses to do anything unless the caller
    explicitly passes ``trust_pickle=True``, affirming they trust the source of
    the file. Never enable this for a file whose provenance you cannot vouch
    for. mlsca-bench never sets ``trust_pickle`` for you: the shipped registry
    leaves it off, so a pickle dataset stays inert until you opt in per call,
    e.g. ``load_dataset("galactics", trust_pickle=True)``.

Used by the GALACTICS/BLISS release, whose traces live in pandas-pickled
DataFrames. Because unpickling is required to read the traces at all, there is
no "safe" path that avoids it; the protection is the explicit opt-in plus the
download-time hash check on the archive.
"""

from __future__ import annotations

import fnmatch
import os
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..base import ArraySideChannelDataset
from ..errors import MissingDependencyError


class PickleTrustError(RuntimeError):
    """Raised when a pickle dataset is loaded without ``trust_pickle=True``."""


def _find_member(
    source: str | os.PathLike[str] | Iterable[Path], member: str
) -> Path:
    if isinstance(source, (str, os.PathLike)):
        roots: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        roots = tuple(Path(path).expanduser() for path in source)
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(p for p in root.rglob(member) if p.is_file())
        elif root.is_file() and fnmatch.fnmatch(root.name, member):
            matches.append(root)
    if not matches:
        raise FileNotFoundError(f"Pickle member {member!r} was not found.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files match {member!r}; make it more specific.")
    return matches[0]


class PickleDataset(ArraySideChannelDataset):
    """Load traces from a pandas-pickled DataFrame — only with explicit opt-in.

    SECURITY: unpickling runs arbitrary code. This constructor raises
    :class:`PickleTrustError` unless ``trust_pickle=True`` is passed.

    ``member`` selects the pickle file. ``traces_column`` names a DataFrame
    column whose cells are per-trace arrays; if omitted, the numeric columns of
    the DataFrame are taken as the ``(n_traces, n_samples)`` matrix.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        member: str,
        traces_column: str | None = None,
        trust_pickle: bool = False,
        split: str | None = None,
        algorithm: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not trust_pickle:
            raise PickleTrustError(
                f"Refusing to load {name!r}: it is distributed as Python pickle "
                "files, and unpickling EXECUTES ARBITRARY CODE embedded in the "
                "file. Only proceed if you trust the source, then pass "
                "trust_pickle=True, e.g. load_dataset(name, trust_pickle=True)."
            )
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - exercised without extra
            raise MissingDependencyError(
                "pandas", extra="pickle", feature="Loading pickled datasets"
            ) from error

        path = _find_member(source, member)
        warnings.warn(
            f"Unpickling {path.name!r} for dataset {name!r}: this executes code "
            "contained in the file. You enabled this with trust_pickle=True.",
            stacklevel=2,
        )
        frame = pd.read_pickle(path)

        if traces_column is not None:
            traces = np.stack(
                [np.asarray(item) for item in frame[traces_column].to_numpy()]
            )
        else:
            numeric = frame.select_dtypes(include="number")
            if numeric.shape[1] == 0:
                raise ValueError(
                    f"{path.name} has no numeric columns; set traces_column."
                )
            traces = np.asarray(numeric.to_numpy())
        if traces.ndim == 1:
            traces = traces.reshape(-1, 1)

        dataset_metadata = {"format": "pickle", "path": str(path)}
        if algorithm is not None:
            dataset_metadata["algorithm"] = algorithm
        dataset_metadata.update(metadata or {})
        super().__init__(
            name=name, split=split, traces=traces, metadata=dataset_metadata
        )


__all__ = ["PickleDataset", "PickleTrustError"]
