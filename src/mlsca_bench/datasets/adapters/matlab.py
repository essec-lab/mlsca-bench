# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for datasets distributed as MATLAB ``.mat`` files.

This covers classic MATLAB level-5 files (MATLAB v6/v7), which SciPy reads
directly. MATLAB v7.3 files are HDF5 containers and can be read with the
generic HDF5 adapter instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..base import ArraySideChannelDataset
from ..errors import MissingDependencyError
from ..errors import describe_paths

try:
    from scipy.io import loadmat
except ImportError as error:  # pragma: no cover - exercised without the extra.
    raise MissingDependencyError(
        "scipy", extra="matlab", feature="Loading MATLAB .mat datasets"
    ) from error


def _select_mat_file(
    source: str | Path | Iterable[Path], preferred_filename: str | None
) -> Path:
    if isinstance(source, (str, Path)):
        candidates: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        candidates = tuple(Path(path).expanduser() for path in source)

    mat_files: list[Path] = []
    for path in candidates:
        if path.is_dir():
            mat_files.extend(item for item in path.rglob("*.mat") if item.is_file())
        elif path.is_file() and path.suffix.casefold() == ".mat":
            mat_files.append(path)
    if not mat_files:
        raise FileNotFoundError(f"No MATLAB .mat file was found in {describe_paths(candidates)}.")

    if preferred_filename is not None:
        chosen = [
            path
            for path in mat_files
            if path.name.casefold() == preferred_filename.casefold()
        ]
        if not chosen:
            raise FileNotFoundError(
                f"Requested MATLAB file {preferred_filename!r} was not found."
            )
        if len(chosen) > 1:
            raise RuntimeError(f"Multiple files named {preferred_filename} were found.")
        return chosen[0]

    if len(mat_files) > 1:
        raise RuntimeError(
            "Multiple .mat files were found; set preferred_filename to choose one."
        )
    return mat_files[0]


class MatlabDataset(ArraySideChannelDataset):
    """Read aligned fields from the variables of a MATLAB ``.mat`` file.

    ``fields`` maps each public field to the name of the MATLAB variable that
    holds it. Trace matrices are commonly stored transposed (one column per
    trace); set ``transpose_traces`` to reorder them to ``(n_traces, samples)``.
    """

    def __init__(
        self,
        source: str | Path | Iterable[Path],
        *,
        name: str,
        fields: Mapping[str, str],
        split: str | None = None,
        transpose_traces: bool = False,
        preferred_filename: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if "traces" not in fields:
            raise ValueError("A MATLAB layout must map the traces variable.")

        path = _select_mat_file(source, preferred_filename)
        contents = loadmat(path)

        def variable(field_name: str) -> np.ndarray[Any, Any] | None:
            variable_name = fields.get(field_name)
            if variable_name is None:
                return None
            if variable_name not in contents:
                raise KeyError(
                    f"Variable {variable_name!r} was not found in {path.name}."
                )
            return np.asarray(contents[variable_name])

        traces = variable("traces")
        assert traces is not None
        if transpose_traces:
            traces = traces.T
        if traces.ndim == 1:
            traces = traces.reshape(-1, 1)

        dataset_metadata = {"format": "matlab", "path": str(path)}
        dataset_metadata.update(metadata or {})

        super().__init__(
            name=name,
            split=split,
            traces=traces,
            plaintexts=variable("plaintexts"),
            ciphertexts=variable("ciphertexts"),
            keys=variable("keys"),
            masks=variable("masks"),
            labels=variable("labels"),
            metadata=dataset_metadata,
        )


def _decode_hex(values: Any) -> np.ndarray[Any, Any]:
    """Decode hex strings ("1a2b...") into a (n, n_bytes) uint8 array.

    Handles both MATLAB cell arrays (loaded as an object array of strings) and
    char matrices (loaded as a 2-D array of single characters).
    """

    array = np.asarray(values)
    if array.dtype.kind in ("U", "S") and array.ndim == 2:
        strings = ["".join(str(char) for char in row) for row in array]
    else:
        strings = []
        for item in np.asarray(array, dtype=object).ravel():
            text = item.item() if hasattr(item, "item") else item
            if isinstance(text, bytes):
                text = text.decode("ascii")
            strings.append(str(text))
    return np.stack(
        [np.frombuffer(bytes.fromhex(text.strip()), dtype=np.uint8) for text in strings]
    )


def _index_mat_files(
    source: str | Path | Iterable[Path], base_dir: str | None
) -> dict[str, list[Path]]:
    if isinstance(source, (str, Path)):
        roots: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        roots = tuple(Path(path).expanduser() for path in source)
    by_name: dict[str, list[Path]] = {}
    for root in roots:
        candidates = (
            [item for item in root.rglob("*.mat") if item.is_file()]
            if root.is_dir()
            else ([root] if root.suffix.casefold() == ".mat" else [])
        )
        for path in candidates:
            if "__MACOSX" in path.parts:
                continue
            if base_dir is not None and path.parent.name.casefold() != base_dir.casefold():
                continue
            by_name.setdefault(path.name, []).append(path)
    return by_name


class MatlabMultiFileDataset(ArraySideChannelDataset):
    """Read each field from its own ``.mat`` file in a directory.

    ``field_files`` maps a public field to a spec ``{"filename", "var",
    "transpose"?, "hex"?}``. ``hex`` decodes an array of hex strings into bytes
    (used by datasets that store plaintext/ciphertext as 32-char hex). Pass
    ``base_dir`` to restrict the search to one subdirectory when the archive
    holds several same-named files (e.g. one folder per device).
    """

    def __init__(
        self,
        source: str | Path | Iterable[Path],
        *,
        name: str,
        field_files: Mapping[str, Mapping[str, Any]],
        split: str | None = None,
        base_dir: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if "traces" not in field_files:
            raise ValueError("A multi-file MATLAB layout must map the traces file.")
        by_name = _index_mat_files(source, base_dir)

        def read(field_name: str) -> np.ndarray[Any, Any] | None:
            spec = field_files.get(field_name)
            if spec is None:
                return None
            filename = str(spec["filename"])
            matches = by_name.get(filename, [])
            if not matches:
                raise FileNotFoundError(f"MATLAB file {filename!r} was not found.")
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple files named {filename}; set base_dir to disambiguate."
                )
            array = np.asarray(loadmat(matches[0])[spec["var"]])
            if spec.get("transpose"):
                array = array.T
            if spec.get("hex"):
                array = _decode_hex(array)
            if field_name != "traces" and array.ndim == 1:
                array = array.reshape(-1, 1)
            return array

        traces = read("traces")
        assert traces is not None

        dataset_metadata = {"format": "matlab"}
        dataset_metadata.update(metadata or {})
        super().__init__(
            name=name,
            split=split,
            traces=traces,
            plaintexts=read("plaintexts"),
            ciphertexts=read("ciphertexts"),
            keys=read("keys"),
            masks=read("masks"),
            labels=read("labels"),
            metadata=dataset_metadata,
        )


__all__ = ["MatlabDataset", "MatlabMultiFileDataset"]
