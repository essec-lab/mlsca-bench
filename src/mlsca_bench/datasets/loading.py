# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Load registered datasets through format-specific adapters."""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base import SideChannelDataset, TraceSample
from .download import (
    DEFAULT_DOWNLOAD_RETRIES,
    DEFAULT_DOWNLOAD_TIMEOUT,
    DEFAULT_RETRY_BACKOFF,
    download_dataset,
)
from .registry import get_dataset, is_user_dataset, local_dataset_path
from .errors import describe_paths


def _all_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for supplied_path in paths:
        path = Path(supplied_path).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    return tuple(files)


def _find_hdf5_file(
    paths: Iterable[Path], *, preferred_name: str, dataset_name: str
) -> Path:
    paths = tuple(paths)
    files = _all_files(paths)
    exact = [
        path for path in files if path.name.casefold() == preferred_name.casefold()
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(f"Multiple files named {preferred_name} were found.")

    hdf5_files = [
        path for path in files if path.suffix.casefold() in {".h5", ".hdf5"}
    ]
    if len(hdf5_files) == 1:
        return hdf5_files[0]
    if not hdf5_files:
        raise FileNotFoundError(
            f"No {dataset_name} HDF5 file was found in {describe_paths(paths)}."
        )
    raise RuntimeError(
        "Multiple HDF5 files were found; pass the intended file explicitly."
    )


def _natural_key(name: str) -> list[Any]:
    """Sort key that orders embedded numbers numerically (chunk_2 < chunk_10)."""

    return [
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", name)
    ]


def _chunk_files(
    paths: Iterable[Path], config: dict[str, Any]
) -> tuple[Path, ...]:
    glob = config.get("chunk_glob")
    suffix = config.get("chunk_suffix")
    selected = []
    for path in _all_files(paths):
        if glob is not None:
            if fnmatch.fnmatch(path.name, glob):
                selected.append(path)
        elif suffix is not None:
            if path.suffix.casefold() == str(suffix).casefold():
                selected.append(path)
        else:
            selected.append(path)
    return tuple(sorted(selected, key=lambda path: _natural_key(path.name)))


def _chunk_dirs(paths: Iterable[Path], traces_name: str) -> tuple[Path, ...]:
    """Directories that each hold one chunk's files (identified by traces_name)."""

    directories = {
        file.parent for file in _all_files(paths) if file.name == traces_name
    }
    return tuple(sorted(directories, key=lambda path: _natural_key(path.name)))


def _load_chunk(
    adapter: str,
    path: Path,
    config: dict[str, Any],
    split: str | None,
    name: str,
    label: str,
) -> SideChannelDataset:
    if adapter == "hdf5":
        from .adapters import HDF5CompoundDataset

        field_aliases = {
            field: tuple(aliases)
            for field, aliases in dict(config.get("field_aliases", {})).items()
        }
        return HDF5CompoundDataset(
            path,
            name=name,
            splits=dict(config.get("splits", {})),
            split=split,
            field_aliases=field_aliases,
            traces_dataset=config.get("traces_dataset", "traces"),
            labels_dataset=config.get("labels_dataset"),
            metadata_dataset=config.get("metadata_dataset", "metadata"),
            require_labels=bool(config.get("require_labels", False)),
            algorithm=config.get("algorithm"),
            label=label,
            extra_metadata=config.get("metadata"),
        )
    if adapter == "raw-binary":
        from .adapters import RawBinaryDataset

        record = config.get("record")
        if record is None:
            raise ValueError("Chunked raw-binary datasets require 'record' mode.")
        record = dict(record)
        record["filename"] = path.name
        return RawBinaryDataset(
            [path],
            name=name,
            record=record,
            split=split,
            metadata=config.get("metadata"),
        )
    if adapter == "npz":
        from .adapters import NpzArchiveDataset

        return NpzArchiveDataset(
            [path],
            name=name,
            layout=dict(config["layout"]),
            split=split,
            uint32_fields=config.get("uint32_fields"),
            metadata=config.get("metadata"),
        )
    if adapter == "npy":
        from .adapters import NpyDirectoryDataset

        # Some datasets store each field in a parallel per-chunk directory tree
        # (e.g. DTDS Dilithium3/5: traces under Traces/NNN, labels under
        # Median/NNN). ``field_subdirs`` names those sibling trees; the chunk
        # index (this dir's name) is shared across them.
        field_subdirs = config.get("field_subdirs")
        if field_subdirs:
            base = path.parent.parent
            sources: list[Path] = [
                base / subdir / path.name
                for subdir in dict.fromkeys(field_subdirs.values())
            ]
        else:
            sources = [path]
        return NpyDirectoryDataset(
            sources,
            name=name,
            layout=dict(config["layout"]),
            split=split,
            metadata=config.get("metadata"),
        )
    if adapter == "trs":
        from .adapters import TRSDataset

        return TRSDataset(
            path,
            name=name,
            split=split,
            data_fields=config.get("data_fields"),
            metadata=config.get("metadata"),
        )
    if adapter == "chameleon":
        from .adapters import ChameleonDataset

        return ChameleonDataset(
            path,
            name=name,
            split=split,
            algorithm=config.get("algorithm", "AES-128"),
        )
    if adapter == "matlab":
        from .adapters.matlab import MatlabMultiFileDataset

        # ``path`` is the anchor file (matched by chunk_glob); sibling field
        # files are derived from its name.
        field_files: dict[str, dict[str, Any]] = {}
        for field, spec in dict(config["field_files"]).items():
            spec = dict(spec)
            if spec.get("anchor"):
                spec["filename"] = path.name
            elif "from_anchor" in spec:
                old, new = spec.pop("from_anchor")
                spec["filename"] = path.name.replace(old, new)
            field_files[field] = spec
        return MatlabMultiFileDataset(
            [path.parent],
            name=name,
            field_files=field_files,
            split=split,
            metadata=config.get("metadata"),
        )
    raise ValueError(f"Chunked loading is not supported for adapter {adapter!r}.")


def _split_config(config: dict[str, Any], split: str | None) -> dict[str, Any]:
    """Overlay a per-split override (``config['splits'][split]``) onto ``config``.

    Lets one chunked dataset select different chunks/layouts per split — e.g. a
    random-key profiling glob vs a fixed-key attack glob, the latter supplying
    the published key via ``fixed_key``.
    """

    splits = config.get("splits")
    if not splits or split is None or split not in splits:
        return config
    override = splits[split]
    # HDF5 uses ``splits`` as a split->group-name (str) map; only npz-style
    # per-split config (a mapping of chunk_glob/layout/fixed_key) is overlaid.
    if not isinstance(override, Mapping):
        return config
    effective = dict(config)
    effective.update(override)
    return effective


class _ConstantKeyDataset(SideChannelDataset):
    """Wrap a dataset whose files carry no key, exposing a constant key.

    Used for fixed-key attack sets (e.g. Spook ``fkey_*`` files hold only traces
    and nonces) where the true key is published separately. ``fixed_key`` is a
    hex string decoded little-endian to ``uint32`` state rows (the Clyde/Spook
    convention), broadcast across all traces without copying.
    """

    def __init__(self, inner: SideChannelDataset, fixed_key: str) -> None:
        self._inner = inner
        self._key_row = np.frombuffer(bytes.fromhex(fixed_key), dtype="<u4")

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def split(self) -> str | None:
        return self._inner.split

    @property
    def traces(self):
        return self._inner.traces

    @property
    def plaintexts(self):
        return self._inner.plaintexts

    @property
    def ciphertexts(self):
        return self._inner.ciphertexts

    @property
    def keys(self):
        return np.broadcast_to(self._key_row, (len(self._inner), self._key_row.shape[0]))

    @property
    def masks(self):
        return self._inner.masks

    @property
    def labels(self):
        return self._inner.labels

    @property
    def metadata(self):
        return self._inner.metadata

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def close(self) -> None:
        self._inner.close()

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int | slice):
        sample = self._inner[index]
        if isinstance(sample, list):
            return [dataclasses.replace(s, key=self._key_row) for s in sample]
        return dataclasses.replace(sample, key=self._key_row)


_CONFIGURED_FIELDS = ("plaintexts", "ciphertexts", "keys", "masks", "labels")


def _hdf5_paths(dataset: SideChannelDataset, limit: int = 25) -> list[str]:
    """Dataset paths inside an open HDF5 file, to suggest what a typo meant."""

    handle = getattr(dataset, "_file", None) or getattr(dataset, "_h5", None)
    names: list[str] = []
    if handle is not None and hasattr(handle, "visit"):
        handle.visit(lambda n: names.append(n) if len(names) < limit and hasattr(handle[n], "shape") else None)
    return names


def _check_user_fields(name: str, dataset: SideChannelDataset) -> None:
    """For your own datasets, every configured field must be found in the files.

    Built-in datasets may lack a field in one split (e.g. SMAesH), so this
    strict check only applies to datasets added with register_dataset().
    """

    config = get_dataset(name).adapter_config or {}
    configured = set(dict(config.get("field_aliases", {}))) | set(dict(config.get("layout", {})))
    missing = [f for f in _CONFIGURED_FIELDS if f in configured and getattr(dataset, f, None) is None]
    if not missing:
        return
    aliases = dict(config.get("field_aliases", {}))
    detail = "; ".join(f"{f} -> {list(aliases.get(f, [])) or 'configured file'}" for f in missing)
    found = _hdf5_paths(dataset)
    dataset.close()
    hint = f" Paths found in the file: {', '.join(found)}." if found else ""
    raise ValueError(
        f"Dataset {name!r}: configured field(s) not found in the files ({detail}). "
        f"Check the spelling in adapter_config.{hint}"
    )


def load_dataset(
    name: str,
    *,
    path: str | os.PathLike[str] | None = None,
    split: str | None = None,
    destination: str | os.PathLike[str] | None = None,
    progress: bool = True,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    trust_pickle: bool = False,
    files: str | Sequence[str] | None = None,
) -> SideChannelDataset:
    """Load a registered dataset, downloading it when ``path`` is omitted.

    ``trust_pickle`` must be set to ``True`` to load datasets distributed as
    Python pickle files (currently GALACTICS). Unpickling executes arbitrary
    code, so this is off by default and never enabled by the registry; pass it
    only for a source you trust.

    ``files`` downloads only the matching files (see :func:`download_dataset`).
    For datasets whose splits are separate files (e.g. SMAesH), loading a split
    downloads only that split's files.
    """

    dataset = _open_dataset(
        name, path=path, split=split, destination=destination, progress=progress,
        timeout=timeout, retries=retries, retry_backoff=retry_backoff, trust_pickle=trust_pickle,
        files=files,
    )
    if is_user_dataset(name):
        _check_user_fields(name, dataset)
    return dataset


def _open_dataset(
    name: str,
    *,
    path: str | os.PathLike[str] | None = None,
    split: str | None = None,
    destination: str | os.PathLike[str] | None = None,
    progress: bool = True,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    trust_pickle: bool = False,
    files: str | Sequence[str] | None = None,
) -> SideChannelDataset:

    specification = get_dataset(name)
    if specification.adapter is None:
        raise ValueError(
            f"Dataset {specification.name!r} is downloadable but does not yet "
            "have a loading adapter."
        )

    config_splits = dict((specification.adapter_config or {}).get("splits") or {})
    if split is None and not (specification.adapter_config or {}).get("default_split") and len(config_splits) == 1:
        split = next(iter(config_splits))                 # the only split: no need to name it

    if path is None:
        path = local_dataset_path(specification.name)   # user dataset already on disk

    if path is None:
        config_all = specification.adapter_config or {}
        split_files = dict(config_all.get("split_files") or {})
        wanted_split = split or config_all.get("default_split")
        if files is None and wanted_split in split_files:
            files = list(split_files[wanted_split])      # only download what this split needs
        candidates = download_dataset(
            specification.name,
            destination,
            progress=progress,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            files=files,
        )
    else:
        candidates = (Path(path),)

    if specification.adapter == "ascad":
        from .adapters import ASCADDataset

        return ASCADDataset(
            _find_hdf5_file(
                candidates,
                preferred_name="ASCAD.h5",
                dataset_name="ASCAD",
            ),
            name=specification.name,
            split=split or "profiling",
        )

    if specification.adapter == "aes-hd-csv":
        from .adapters import AESHDCSVDataset

        if split is not None:
            raise ValueError("The AES-HD CSV release does not define splits.")
        return AESHDCSVDataset(candidates, name=specification.name)

    if specification.adapter == "aes-hd-zaid":
        from .adapters import AESHDZaidDataset

        return AESHDZaidDataset(
            candidates,
            name=specification.name,
            split=split or "profiling",
        )

    if specification.adapter == "ascon-hdf5":
        from .adapters import AsconHDF5Dataset

        return AsconHDF5Dataset(
            _find_hdf5_file(
                candidates,
                preferred_name="ascon_cw_unprotected.h5",
                dataset_name="Ascon",
            ),
            name=specification.name,
            split=split or "fixed",
        )

    config = dict(specification.adapter_config or {})

    if config.get("chunked"):
        from .adapters import ConcatDataset

        active_split = split or config.get("default_split")
        # A dataset may define per-split chunk selection (e.g. Spook: a random-key
        # profiling glob and a fixed-key attack glob, the latter with the published
        # fixed key supplied via `fixed_key`). Overlay that split's config, if any.
        chunk_config = _split_config(config, active_split)

        if specification.adapter == "npy":
            # Each chunk is a directory holding that chunk's per-field .npy files.
            chunk_paths = _chunk_dirs(candidates, dict(chunk_config["layout"])["traces"])
        else:
            chunk_paths = _chunk_files(candidates, chunk_config)
        if not chunk_paths:
            raise FileNotFoundError(
                "No chunk files matched the configured pattern for "
                f"{specification.name!r}."
            )
        chunks = [
            _load_chunk(
                specification.adapter,
                chunk_path,
                chunk_config,
                active_split,
                specification.name,
                config.get("label", specification.display_name),
            )
            for chunk_path in chunk_paths
        ]
        dataset = ConcatDataset(chunks, name=specification.name, split=active_split)
        fixed_key = chunk_config.get("fixed_key")
        if fixed_key is not None:
            dataset = _ConstantKeyDataset(dataset, fixed_key)
        return dataset

    if specification.adapter == "hdf5":
        from .adapters import HDF5CompoundDataset

        splits = dict(config.get("splits", {}))
        field_aliases = {
            field: tuple(aliases)
            for field, aliases in dict(config.get("field_aliases", {})).items()
        }
        hdf5_file = _find_hdf5_file(
            candidates,
            preferred_name=config.get("preferred_filename", ""),
            dataset_name=specification.display_name,
        )
        return HDF5CompoundDataset(
            hdf5_file,
            name=specification.name,
            splits=splits,
            split=split or config.get("default_split"),
            field_aliases=field_aliases,
            traces_dataset=config.get("traces_dataset", "traces"),
            labels_dataset=config.get("labels_dataset"),
            metadata_dataset=config.get("metadata_dataset", "metadata"),
            require_labels=bool(config.get("require_labels", False)),
            algorithm=config.get("algorithm"),
            label=config.get("label", specification.display_name),
            extra_metadata=config.get("metadata"),
        )

    if specification.adapter == "flat-hdf5":
        from .adapters import FlatHDF5Dataset

        return FlatHDF5Dataset(
            _find_hdf5_file(
                candidates,
                preferred_name=config.get("preferred_filename", ""),
                dataset_name=specification.display_name,
            ),
            name=specification.name,
            splits=dict(config.get("splits", {})),
            split=split or config.get("default_split"),
            labels=config.get("labels"),
            data=config.get("data"),
            data_columns=config.get("data_columns"),
            algorithm=config.get("algorithm"),
            label=config.get("label", specification.display_name),
            extra_metadata=config.get("metadata"),
        )

    if specification.adapter == "npy":
        from .adapters import NpyDirectoryDataset

        return NpyDirectoryDataset(
            candidates,
            name=specification.name,
            layout=dict(config["layout"]),
            split=split or config.get("default_split"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "npz":
        from .adapters import NpzArchiveDataset

        return NpzArchiveDataset(
            candidates,
            name=specification.name,
            layout=dict(config["layout"]),
            split=split or config.get("default_split"),
            uint32_fields=config.get("uint32_fields"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "matlab":
        if "field_files" in config:
            from .adapters.matlab import MatlabMultiFileDataset

            return MatlabMultiFileDataset(
                candidates,
                name=specification.name,
                field_files={
                    field: dict(spec)
                    for field, spec in dict(config["field_files"]).items()
                },
                split=split or config.get("default_split"),
                base_dir=config.get("base_dir"),
                metadata=config.get("metadata"),
            )

        from .adapters.matlab import MatlabDataset

        return MatlabDataset(
            candidates,
            name=specification.name,
            fields=dict(config["fields"]),
            split=split or config.get("default_split"),
            transpose_traces=bool(config.get("transpose_traces", False)),
            preferred_filename=config.get("preferred_filename"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "raw-binary":
        from .adapters import RawBinaryDataset

        return RawBinaryDataset(
            candidates,
            name=specification.name,
            layout=config.get("layout"),
            record=config.get("record"),
            split=split or config.get("default_split"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "lecroy-index":
        from .adapters import LeCroyIndexDataset

        return LeCroyIndexDataset(
            candidates,
            name=specification.name,
            index_filename=config["index_filename"],
            columns={key: int(value) for key, value in dict(config["columns"]).items()},
            split=split or config.get("default_split"),
            algorithm=config.get("algorithm"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "pickle":
        from .adapters import PickleDataset

        return PickleDataset(
            candidates,
            name=specification.name,
            member=config["member"],
            traces_column=config.get("traces_column"),
            trust_pickle=trust_pickle,
            split=split or config.get("default_split"),
            algorithm=config.get("algorithm"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "agilent-wave":
        from .adapters import AgilentWaveDataset

        return AgilentWaveDataset(
            candidates,
            name=specification.name,
            file_glob=config["file_glob"],
            filename_fields=config.get("filename_fields"),
            split=split or config.get("default_split"),
            algorithm=config.get("algorithm"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "two-class-npy":
        from .adapters import TwoClassNpyDataset

        return TwoClassNpyDataset(
            candidates,
            name=specification.name,
            class0_file=config.get("class0_file", "fixed.npy"),
            class1_file=config.get("class1_file", "random.npy"),
            base_dir=config.get("base_dir"),
            split=split or config.get("default_split"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "ascii-wave":
        from .adapters import AsciiWaveDataset

        return AsciiWaveDataset(
            candidates,
            name=specification.name,
            file_glob=config["file_glob"],
            comment_prefix=config.get("comment_prefix", "#"),
            dtype=config.get("dtype", "int16"),
            index_regex=config.get("index_regex"),
            filename_fields=config.get("filename_fields"),
            split=split or config.get("default_split"),
            algorithm=config.get("algorithm"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "scaaml":
        from .adapters.scaaml import open_scaaml

        return open_scaaml(
            candidates,
            name=specification.name,
            traces_feature=config["traces_feature"],
            field_map=dict(config["field_map"]),
            split=split or config.get("default_split"),
            metadata=config.get("metadata"),
        )

    if specification.adapter == "npy-manifest":
        from .adapters import ManifestNpyDataset

        active_split = split or config.get("default_split")
        split_dirs = config.get("split_dirs") or {}
        subdir = split_dirs.get(active_split) if active_split else None
        return ManifestNpyDataset(
            candidates,
            name=specification.name,
            field_map=dict(config["field_map"]),
            split=active_split,
            subdir=subdir,
            metadata=config.get("metadata"),
        )

    if specification.adapter == "trs":
        from .adapters import TRSDataset

        return TRSDataset(
            candidates,
            name=specification.name,
            split=split or config.get("default_split"),
            data_fields=config.get("data_fields"),
            preferred_filename=config.get("preferred_filename"),
            metadata=config.get("metadata"),
        )

    raise ValueError(f"Unsupported dataset adapter: {specification.adapter!r}")


__all__ = ["load_dataset"]
