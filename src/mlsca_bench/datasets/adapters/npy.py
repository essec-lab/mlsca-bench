# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for datasets stored as NumPy ``.npy`` files or ``.npz`` archives."""

from __future__ import annotations

import os
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

from ..base import DatasetArray, SideChannelDataset, TraceSample, validate_dataset
from ..errors import describe_paths

_FIELDS = ("traces", "plaintexts", "ciphertexts", "keys", "masks", "labels")


class _NpzMember:
    """Lazy view of one array in a ``.npz`` archive.

    ``shape`` and ``dtype`` are read from the member's ``.npy`` header without
    decompressing the data, so a chunked dataset can report its length without
    loading anything. The array is materialized (and cached) only on first
    element/slice access.
    """

    def __init__(self, archive: Any, archive_path: Path, key: str) -> None:
        self._archive = archive
        self._key = key
        self._cache: np.ndarray[Any, Any] | None = None
        with zipfile.ZipFile(archive_path) as zip_file:
            with zip_file.open(f"{key}.npy") as handle:
                major, _minor = npy_format.read_magic(handle)
                if major == 1:
                    shape, _fortran, dtype = npy_format.read_array_header_1_0(handle)
                else:
                    shape, _fortran, dtype = npy_format.read_array_header_2_0(handle)
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __len__(self) -> int:
        return self._shape[0]

    def _materialize(self) -> np.ndarray[Any, Any]:
        if self._cache is None:
            self._cache = self._archive[self._key]
        return self._cache

    def __getitem__(self, index: Any) -> Any:
        return self._materialize()[index]


class _ArrayFieldsDataset(SideChannelDataset):
    """Common logic for datasets backed by one array per aligned field.

    Subclasses populate ``self._arrays`` (a ``field -> array-or-None`` mapping
    that must include a ``traces`` entry) and may override :meth:`_release` to
    free any backing resources. The arrays may be lazy (memory maps or archive
    members); this base never copies them.
    """

    _name: str
    _split: str | None
    _arrays: dict[str, DatasetArray | None]
    _metadata: Mapping[str, Any]
    _closed: bool

    def _finalize(self) -> None:
        """Validate the assembled fields once ``self._arrays`` is populated."""

        if self._arrays.get("traces") is None:
            raise ValueError("A NumPy dataset must define a traces array.")
        validate_dataset(self)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Dataset {self.name!r} is closed.")

    def _array(self, name: str) -> DatasetArray | None:
        self._ensure_open()
        return self._arrays[name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def traces(self) -> DatasetArray:
        value = self._array("traces")
        assert value is not None
        return value

    @property
    def plaintexts(self) -> DatasetArray | None:
        return self._array("plaintexts")

    @property
    def ciphertexts(self) -> DatasetArray | None:
        return self._array("ciphertexts")

    @property
    def keys(self) -> DatasetArray | None:
        return self._array("keys")

    @property
    def masks(self) -> DatasetArray | None:
        return self._array("masks")

    @property
    def labels(self) -> DatasetArray | None:
        return self._array("labels")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def _release(self) -> None:
        """Release backing resources. Overridden by subclasses as needed."""

    def close(self) -> None:
        if self._closed:
            return
        self._release()
        self._closed = True

    def __len__(self) -> int:
        traces = self._arrays["traces"]
        assert traces is not None
        return len(traces)

    def __getitem__(
        self, index: int | slice
    ) -> TraceSample | Sequence[TraceSample]:
        self._ensure_open()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset indexes must be integers or slices.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range.")

        def sample(field_name: str) -> np.ndarray[Any, Any] | None:
            value = self._arrays[field_name]
            return None if value is None else np.asarray(value[index])

        trace = sample("traces")
        assert trace is not None
        return TraceSample(
            trace=trace,
            plaintext=sample("plaintexts"),
            ciphertext=sample("ciphertexts"),
            key=sample("keys"),
            mask=sample("masks"),
            label=sample("labels"),
        )


class NpyDirectoryDataset(_ArrayFieldsDataset):
    """Load aligned fields from named ``.npy`` files using memory maps.

    ``layout`` maps each field to a filename template; ``{split}`` in a
    template is substituted with the active split, letting one layout describe
    both a profiling and an attack split.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        layout: Mapping[str, str],
        split: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if "traces" not in layout:
            raise ValueError("An NPY layout must define a traces file.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        files = self._resolve_files(source)
        by_name: dict[str, list[Path]] = {}
        for path in files:
            by_name.setdefault(path.name, []).append(path)

        self._arrays = {}
        for field_name in _FIELDS:
            template = layout.get(field_name)
            if template is None:
                self._arrays[field_name] = None
                continue
            filename = template.format(split=split or "")
            matches = by_name.get(filename, [])
            if not matches:
                raise FileNotFoundError(f"Required NumPy file was not found: {filename}")
            if len(matches) > 1:
                raise RuntimeError(f"Multiple NumPy files named {filename} were found.")
            self._arrays[field_name] = np.load(
                matches[0], mmap_mode="r", allow_pickle=False
            )

        dataset_metadata = {
            "format": "npy",
            "files": MappingProxyType(
                {field: layout[field].format(split=split or "") for field in layout}
            ),
        }
        dataset_metadata.update(metadata or {})
        self._metadata = MappingProxyType(dataset_metadata)
        self._finalize()

    @staticmethod
    def _resolve_files(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> tuple[Path, ...]:
        if isinstance(source, (str, os.PathLike)):
            supplied = (Path(source).expanduser(),)
        else:
            supplied = tuple(Path(path).expanduser() for path in source)
        files: list[Path] = []
        for path in supplied:
            if path.is_dir():
                files.extend(item for item in path.rglob("*.npy") if item.is_file())
            elif path.is_file() and path.suffix.casefold() == ".npy":
                files.append(path)
        return tuple(files)

    def _release(self) -> None:
        for value in self._arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


class NpzArchiveDataset(_ArrayFieldsDataset):
    """Load aligned fields from a single ``.npz`` archive.

    ``layout`` maps each field to the array name inside the archive. Only the
    members named in ``layout`` are read; any other arrays in the archive are
    left untouched.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        layout: Mapping[str, str],
        split: str | None = None,
        uint32_fields: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if "traces" not in layout:
            raise ValueError("An NPZ layout must define a traces array.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        as_uint32 = set(uint32_fields or ())
        archive_path = self._resolve_archive(source)
        self._archive = np.load(archive_path, allow_pickle=False)

        try:
            available = set(self._archive.files)
            self._arrays = {}
            for field_name in _FIELDS:
                member = layout.get(field_name)
                if member is None:
                    self._arrays[field_name] = None
                    continue
                key = member.format(split=split or "")
                if key not in available:
                    raise KeyError(
                        f"Array {key!r} was not found in the NPZ archive "
                        f"{archive_path.name}."
                    )
                if field_name in as_uint32:
                    # Fields stored as fixed-width byte strings (e.g. the Spook
                    # hardware nonce/key, |S16) are reinterpreted little-endian
                    # into (n, k) uint32 rows. These fields are small, so
                    # materializing here (traces stay lazy) is fine.
                    raw = np.asarray(self._archive[key])
                    matrix = np.frombuffer(raw.tobytes(), dtype="<u4").reshape(len(raw), -1)
                    self._arrays[field_name] = matrix
                else:
                    self._arrays[field_name] = _NpzMember(
                        self._archive, archive_path, key
                    )

            dataset_metadata = {
                "format": "npz",
                "path": str(archive_path),
                "arrays": MappingProxyType(
                    {field: layout[field].format(split=split or "") for field in layout}
                ),
            }
            dataset_metadata.update(metadata or {})
            self._metadata = MappingProxyType(dataset_metadata)
            self._finalize()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve_archive(
        source: str | os.PathLike[str] | Iterable[Path],
    ) -> Path:
        if isinstance(source, (str, os.PathLike)):
            candidates: tuple[Path, ...] = (Path(source).expanduser(),)
        else:
            candidates = tuple(Path(path).expanduser() for path in source)

        archives: list[Path] = []
        for path in candidates:
            if path.is_dir():
                archives.extend(
                    item for item in path.rglob("*.npz") if item.is_file()
                )
            elif path.is_file() and path.suffix.casefold() == ".npz":
                archives.append(path)
        if not archives:
            raise FileNotFoundError(f"No NPZ archive was found in {describe_paths(candidates)}.")
        if len(archives) > 1:
            raise RuntimeError(
                "Multiple NPZ archives were found; pass the intended file explicitly."
            )
        return archives[0]

    def _release(self) -> None:
        self._archive.close()


class AESHDZaidDataset(NpyDirectoryDataset):
    """Preprocessed AES-HD split published with the Zaid methodology."""

    _LAYOUT = {
        "traces": "{split}_traces_AES_HD.npy",
        "ciphertexts": "{split}_ciphertext_AES_HD.npy",
        "labels": "{split}_labels_AES_HD.npy",
    }

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str = "aes-hd-zaid",
        split: str = "profiling",
    ) -> None:
        if split not in {"profiling", "attack"}:
            raise ValueError("AES-HD Zaid split must be 'profiling' or 'attack'.")
        super().__init__(
            source,
            name=name,
            layout=self._LAYOUT,
            split=split,
            metadata={
                "algorithm": "AES",
                "variant": "Zaid preprocessed",
                # The release ships no key. Its labels equal
                # InvSbox[ct[11] ^ 0x00] ^ ct[7] for all 75,000 traces, so the
                # attacked last-round key byte is byte 11 = 0x00.
                "last_round_key_bytes": {11: 0x00},
            },
        )


__all__ = [
    "AESHDZaidDataset",
    "NpyDirectoryDataset",
    "NpzArchiveDataset",
]
