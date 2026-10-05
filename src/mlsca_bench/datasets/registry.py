# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Load, validate, and query the mlsca-bench dataset registry."""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Literal, Mapping
from urllib.parse import urlparse


Availability = Literal["automatic", "manual", "custom", "unavailable"]

ArchiveType = Literal[
    "zip",
    "zip-deflate64",
    "tar",
    "tar.gz",
    "gz",
    "7z",
    "rar",
    "mat",
    "tar.zstd",
    "zip-parts",
    "tar.bz2-parts",
]

_ARCHIVE_VALUES = {
    "zip",
    "zip-deflate64",
    "tar",
    "tar.gz",
    "gz",
    "7z",
    "rar",
    "mat",
    "tar.zstd",
    "zip-parts",
    "tar.bz2-parts",
}
_AVAILABILITY_VALUES = {"automatic", "manual", "custom", "unavailable"}
_HASH_LENGTHS = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}
_REQUIRED_DATASET_FIELDS = {
    "display_name",
    "aliases",
    "availability",
    "files",
    "homepage",
    "paper",
    "license",
    "notes",
}

_OPTIONAL_DATASET_FIELDS = {
    "adapter",
    "adapter_config",
    "metadata",
    "size_bytes",
    "disk_bytes",
    "license_source",
    "fields",
}
_DATASET_FIELDS = (
    _REQUIRED_DATASET_FIELDS
    | _OPTIONAL_DATASET_FIELDS
)
_REQUIRED_METADATA_FIELDS = {
    "format",
    "algorithm",
    "measurement",
}
_OPTIONAL_METADATA_FIELDS = {
    "platform",
    "device",
    "countermeasures",
    "key",
    "n_samples",
    "year",
}
_METADATA_FIELDS = _REQUIRED_METADATA_FIELDS | _OPTIONAL_METADATA_FIELDS

# Controlled vocabularies for the diversity axes. Kept small and coarse so
# datasets stay comparable; use "other" when unsure.
_PLATFORM_VALUES = {
    "avr",
    "arm-cortex-m",
    "arm-cortex-a",
    "fpga",
    "asic",
    "risc-v",
    "gpu",
    "other",
}
_COUNTERMEASURE_VALUES = {
    "none",
    "masking",
    "affine-masking",
    "shuffling",
    "desynchronization",
    "random-delay",
    "clock-randomization",
    "hiding",
    "threshold-implementation",
    "constant-time",
    "other",
}
_KEY_VALUES = {"fixed", "variable", "partially-variable"}

_FORMAT_VALUES = {
    "hdf5",
    "numpy",
    "matlab",
    "trs",
    "binary",
    "csv",
    "mixed",
    "other",
}

_MEASUREMENT_VALUES = {
    "power",
    "electromagnetic",
    "timing",
    "simulated",
    "mixed",
    "other",
}
_FILE_FIELDS = {
    "backend",
    "url",
    "repo_id",
    "repo_type",
    "revision",
    "filename",
    "known_hash",
    "archive",
    "allow_patterns",
    "persistent_id",
    "path_prefix",
}
_BACKEND_VALUES = {
    "http",
    "gdrive",
    "huggingface",
    "gcs",
    "dataverse",
}



class DatasetRegistryError(ValueError):
    """Base exception for invalid registry data or registry lookups."""


class RegistryValidationError(DatasetRegistryError):
    """Raised when ``registry.json`` does not follow the registry schema."""


class UnknownDatasetError(DatasetRegistryError, KeyError):
    """Raised when no canonical dataset name or alias matches a lookup."""


@dataclass(frozen=True)
class DatasetFile:
    backend: str = "http"

    url: str | None = None

    repo_id: str | None = None
    repo_type: str | None = None
    revision: str | None = None

    persistent_id: str | None = None
    path_prefix: str | None = None

    filename: str | None = None
    known_hash: str | None = None
    archive: ArchiveType | None = None
    allow_patterns: str | list[str] | None = None

@dataclass(frozen=True)
class DatasetMetadata:
    """Scientific characteristics of a side-channel dataset.

    The first three fields are required; the rest describe the diversity axes
    used for coverage analysis (device platform, countermeasures, key regime,
    trace length, year) and are ``None``/empty when unknown.
    """

    format: str
    algorithm: str
    measurement: str
    platform: str | None = None
    device: str | None = None
    countermeasures: tuple[str, ...] = ()
    key: str | None = None
    n_samples: int | None = None
    year: int | None = None

@dataclass(frozen=True)
class DatasetSpec:
    """Validated metadata describing a dataset and how it can be obtained."""

    name: str
    display_name: str
    aliases: tuple[str, ...]
    availability: Availability
    files: tuple[DatasetFile, ...]
    homepage: str | None
    paper: str | None
    license: str | None
    notes: str | None
    metadata: DatasetMetadata | None = None
    adapter: str | None = None
    adapter_config: Mapping[str, Any] | None = None
    size_bytes: int | None = None
    disk_bytes: int | None = None         # space taken once downloaded and unpacked (measured)
    license_source: str | None = None     # where the license is stated (or where we looked)
    # Fields seen when the dataset was loaded from the real files, per split
    # ("all" when every split has the same); None = not checked on real data yet.
    fields: Mapping[str, tuple[str, ...]] | None = None

    @property
    def provides(self) -> tuple[str, ...]:
        """Every field any split provides (empty when not checked yet)."""

        if not self.fields:
            return ()
        return tuple(sorted({f for names in self.fields.values() for f in names}))

    @property
    def size(self) -> str:
        """Approximate download size, for example '4.4 GB', or 'unknown'."""

        return format_size(self.size_bytes)


def format_size(n_bytes: int | None) -> str:
    """Human-readable size in decimal units (as download tools report them)."""

    if n_bytes is None:
        return "unknown"
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def normalize_dataset_name(name: str) -> str:
    """Normalize user input so spaces, underscores, and hyphens match."""

    if not isinstance(name, str) or not name.strip():
        raise UnknownDatasetError("Dataset name must be a non-empty string.")
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def _validation_error(location: str, message: str) -> RegistryValidationError:
    return RegistryValidationError(f"Invalid dataset registry at {location}: {message}")


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _validation_error(location, "expected a JSON object")
    return value


def _freeze_config(value: Any) -> Any:
    """Recursively turn JSON containers into read-only, hashable-friendly views."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_config(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value


def _adapter_config(
    value: Any, location: str, *, adapter: str | None
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    config = _require_object(value, location)
    if adapter is None:
        raise _validation_error(
            location, "adapter_config requires an adapter to be set"
        )
    return _freeze_config(config)


def _require_string(value: Any, location: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        expected = "a non-empty string or null" if nullable else "a non-empty string"
        raise _validation_error(location, f"expected {expected}")
    return value.strip()


def _enum_or_none(
    value: Any, allowed: set[str], location: str
) -> str | None:
    text = _require_string(value, location, nullable=True)
    if text is None:
        return None
    if text not in allowed:
        choices = ", ".join(sorted(allowed))
        raise _validation_error(location, f"expected one of: {choices}")
    return text


def _optional_nonneg_int(value: Any, location: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _validation_error(location, "expected an integer or null")
    if value < 0:
        raise _validation_error(location, "expected a non-negative integer")
    return value


FIELD_NAMES = ("traces", "plaintexts", "ciphertexts", "keys", "masks", "labels")


def _observed_fields(value: Any, location: str) -> Mapping[str, tuple[str, ...]] | None:
    """``["traces", ...]`` for every split, or ``{"profiling": [...], "attack": [...]}``."""

    if value is None:
        return None
    per_split = {"all": value} if isinstance(value, list) else _require_object(value, location)
    result: dict[str, tuple[str, ...]] = {}
    for split, names in per_split.items():
        if not isinstance(names, list) or not names:
            raise _validation_error(f"{location}.{split}", "expected a non-empty list of field names")
        unknown = [n for n in names if n not in FIELD_NAMES]
        if unknown:
            raise _validation_error(f"{location}.{split}", f"unknown fields {unknown}; use {', '.join(FIELD_NAMES)}")
        result[str(split)] = tuple(n for n in FIELD_NAMES if n in names)
    return MappingProxyType(result)


def _optional_url(value: Any, location: str) -> str | None:
    url = _require_string(value, location, nullable=True)
    if url is None:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _validation_error(location, "expected an absolute HTTP(S) URL")
    return url


def _known_hash(value: Any, location: str) -> str | None:
    known_hash = _require_string(value, location, nullable=True)
    if known_hash is None:
        return None

    try:
        algorithm, digest = known_hash.lower().split(":", maxsplit=1)
    except ValueError as error:
        raise _validation_error(
            location, "expected '<algorithm>:<hex digest>'"
        ) from error

    expected_length = _HASH_LENGTHS.get(algorithm)
    if expected_length is None:
        supported = ", ".join(sorted(_HASH_LENGTHS))
        raise _validation_error(location, f"unsupported hash; use one of: {supported}")
    if len(digest) != expected_length or re.fullmatch(r"[0-9a-f]+", digest) is None:
        raise _validation_error(
            location, f"expected a {expected_length}-character {algorithm} hex digest"
        )
    return f"{algorithm}:{digest}"

def _dataset_file(value: Any, location: str) -> DatasetFile:
    url = None
    repo_id = None
    repo_type = None
    revision = None
    persistent_id = None
    path_prefix = None
    item = _require_object(value, location)
    unknown = set(item) - _FILE_FIELDS
    if unknown:
        raise _validation_error(
            location, f"unknown fields: {', '.join(sorted(unknown))}"
        )

    backend = item.get("backend", "http")
    if backend not in _BACKEND_VALUES:
        choices = ", ".join(sorted(_BACKEND_VALUES))
        raise _validation_error(
            f"{location}.backend",
            f"expected one of: {choices}",
        )

    url = None
    repo_id = None
    repo_type = None
    revision = None

    if backend in {"http", "gdrive"}:
        url = _optional_url(item.get("url"), f"{location}.url")
        if url is None:
            raise _validation_error(
                f"{location}.url",
                "download files require a URL",
            )
        if backend == "http" and "drive.google.com/file/" in url:
            # A Drive "view" link serves an HTML page, not the file.
            raise _validation_error(
                f"{location}.url",
                'a Google Drive "view" link returns a web page; use "backend": "gdrive", or '
                "https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t",
            )

    elif backend == "huggingface":
        repo_id = _require_string(
            item.get("repo_id"),
            f"{location}.repo_id",
        )
        assert repo_id is not None

        repo_type = item.get("repo_type", "dataset")
        if repo_type not in {"dataset", "model", "space"}:
            raise _validation_error(
                f"{location}.repo_type",
                "expected one of: dataset, model, space",
            )

        revision = _require_string(
            item.get("revision"),
            f"{location}.revision",
            nullable=True,
        )

    elif backend == "gcs":
        url = _require_string(
            item.get("url"),
            f"{location}.url",
        )

        if not url.startswith("gs://"):
            raise _validation_error(
                f"{location}.url",
                "GCS downloads require a gs:// URL",
            )
    elif backend == "dataverse":
        url = _optional_url(
            item.get("url"),
            f"{location}.url",
        )

        if url is None:
            raise _validation_error(
                f"{location}.url",
                "Dataverse downloads require a server URL",
            )

        persistent_id = _require_string(
            item.get("persistent_id"),
            f"{location}.persistent_id",
        )

        path_prefix = _require_string(
            item.get("path_prefix"),
            f"{location}.path_prefix",
        )
    filename = _require_string(
        item.get("filename"),
        f"{location}.filename",
        nullable=True,
    )

    if filename is not None:
        path = PurePath(filename)
        if path.name != filename or filename in {".", ".."}:
            raise _validation_error(
                f"{location}.filename",
                "expected a plain filename without directories",
            )

    archive = item.get("archive")
    if archive is not None and archive not in _ARCHIVE_VALUES:
        choices = ", ".join(sorted(_ARCHIVE_VALUES))
        raise _validation_error(
            f"{location}.archive",
            f"expected null or one of: {choices}",
        )
    if archive in ("zip-parts", "tar.bz2-parts"):
        if backend != "http" or filename is None or not re.search(r"\.part\d+$", filename):
            raise _validation_error(
                f"{location}.archive",
                f"{archive!r} is for HTTP files named like 'data.zip.part0', 'data.zip.part1', ...",
            )

    known_hash = None

    if backend in {"http", "gdrive"}:
        known_hash = _known_hash(
            item.get("known_hash"),
            f"{location}.known_hash",
        )
    allow_patterns = item.get("allow_patterns")
    return DatasetFile(
        backend=backend,
        allow_patterns=allow_patterns,
        url=url,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=filename,
        known_hash=known_hash,
        archive=archive,
        persistent_id=persistent_id,
        path_prefix=path_prefix,
    )

def _dataset_metadata(
    value: Any,
    location: str,
) -> DatasetMetadata | None:
    if value is None:
        return None

    item = _require_object(value, location)

    unknown = set(item) - _METADATA_FIELDS
    missing = _REQUIRED_METADATA_FIELDS - set(item)

    if unknown:
        raise _validation_error(
            location,
            f"unknown metadata fields: "
            f"{', '.join(sorted(unknown))}",
        )

    if missing:
        raise _validation_error(
            location,
            f"missing metadata fields: "
            f"{', '.join(sorted(missing))}",
        )

    data_format = _require_string(
        item["format"],
        f"{location}.format",
    )
    assert data_format is not None

    if data_format not in _FORMAT_VALUES:
        choices = ", ".join(sorted(_FORMAT_VALUES))
        raise _validation_error(
            f"{location}.format",
            f"expected one of: {choices}",
        )

    algorithm = _require_string(
        item["algorithm"],
        f"{location}.algorithm",
    )
    assert algorithm is not None

    measurement = _require_string(
        item["measurement"],
        f"{location}.measurement",
    )
    assert measurement is not None

    if measurement not in _MEASUREMENT_VALUES:
        choices = ", ".join(sorted(_MEASUREMENT_VALUES))
        raise _validation_error(
            f"{location}.measurement",
            f"expected one of: {choices}",
        )

    platform = _enum_or_none(
        item.get("platform"), _PLATFORM_VALUES, f"{location}.platform"
    )
    device = _require_string(item.get("device"), f"{location}.device", nullable=True)
    key = _enum_or_none(item.get("key"), _KEY_VALUES, f"{location}.key")
    n_samples = _optional_nonneg_int(item.get("n_samples"), f"{location}.n_samples")
    year = _optional_nonneg_int(item.get("year"), f"{location}.year")

    countermeasures_value = item.get("countermeasures", [])
    if not isinstance(countermeasures_value, list):
        raise _validation_error(
            f"{location}.countermeasures", "expected a JSON array"
        )
    countermeasures = []
    for index, entry in enumerate(countermeasures_value):
        cm = _require_string(entry, f"{location}.countermeasures[{index}]")
        if cm not in _COUNTERMEASURE_VALUES:
            choices = ", ".join(sorted(_COUNTERMEASURE_VALUES))
            raise _validation_error(
                f"{location}.countermeasures[{index}]",
                f"expected one of: {choices}",
            )
        countermeasures.append(cm)

    return DatasetMetadata(
        format=data_format,
        algorithm=algorithm,
        measurement=measurement,
        platform=platform,
        device=device,
        countermeasures=tuple(countermeasures),
        key=key,
        n_samples=n_samples,
        year=year,
    )

def _dataset_spec(name: str, value: Any, location: str) -> DatasetSpec:
    item = _require_object(value, location)
    unknown = set(item) - _DATASET_FIELDS
    missing = _REQUIRED_DATASET_FIELDS - set(item)
    if unknown:
        raise _validation_error(location, f"unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise _validation_error(location, f"missing fields: {', '.join(sorted(missing))}")

    normalized_name = normalize_dataset_name(name)
    if normalized_name != name:
        raise _validation_error(
            location, f"canonical name must be normalized as {normalized_name!r}"
        )

    aliases_value = item["aliases"]
    if not isinstance(aliases_value, list):
        raise _validation_error(f"{location}.aliases", "expected a JSON array")
    aliases: list[str] = []
    for index, alias_value in enumerate(aliases_value):
        alias = _require_string(alias_value, f"{location}.aliases[{index}]")
        assert alias is not None
        aliases.append(alias)

    availability = item["availability"]
    if availability not in _AVAILABILITY_VALUES:
        choices = ", ".join(sorted(_AVAILABILITY_VALUES))
        raise _validation_error(f"{location}.availability", f"expected one of: {choices}")

    files_value = item["files"]
    if not isinstance(files_value, list):
        raise _validation_error(f"{location}.files", "expected a JSON array")
    dataset_files = tuple(
        _dataset_file(file_value, f"{location}.files[{index}]")
        for index, file_value in enumerate(files_value)
    )
    if availability == "automatic" and not dataset_files:
        raise _validation_error(
            f"{location}.files", "automatic datasets require at least one file"
        )
    if availability == "automatic":
        for dataset_file in dataset_files:
            if (
                dataset_file.backend in {"http", "gdrive"}
                and dataset_file.known_hash is None
            ):
                raise _validation_error(
                    f"{location}.files",
                    "automatic HTTP/Google Drive downloads require a known hash",
                )
    adapter = _require_string(
        item.get("adapter"),
        f"{location}.adapter",
        nullable=True,
    )

    if (
        adapter is not None
        and normalize_dataset_name(adapter) != adapter
    ):
        raise _validation_error(
            f"{location}.adapter",
            "adapter name must contain only lowercase letters, "
            "numbers, and hyphens",
        )
    adapter_config = _adapter_config(
        item.get("adapter_config"),
        f"{location}.adapter_config",
        adapter=adapter,
    )
    return DatasetSpec(
        name=name,
        display_name=_require_string(item["display_name"], f"{location}.display_name"),
        aliases=tuple(aliases),
        availability=availability,
        files=dataset_files,
        homepage=_optional_url(item["homepage"], f"{location}.homepage"),
        paper=_optional_url(item["paper"], f"{location}.paper"),
        license=_require_string(item["license"], f"{location}.license", nullable=True),
        notes=_require_string(item["notes"], f"{location}.notes", nullable=True),
        metadata=_dataset_metadata(item.get("metadata"), f"{location}.metadata"),
        adapter=adapter,
        adapter_config=adapter_config,
        size_bytes=_optional_nonneg_int(item.get("size_bytes"), f"{location}.size_bytes"),
        disk_bytes=_optional_nonneg_int(item.get("disk_bytes"), f"{location}.disk_bytes"),
        license_source=_optional_url(item.get("license_source"), f"{location}.license_source"),
        fields=_observed_fields(item.get("fields"), f"{location}.fields"),
    )


def load_registry(path: str | Path) -> tuple[Mapping[str, DatasetSpec], Mapping[str, str]]:
    """Load and validate a registry file.

    The second returned mapping resolves normalized canonical names and aliases
    to canonical dataset names. Both mappings are read-only.
    """

    registry_path = Path(path)
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryValidationError(f"Dataset registry not found: {registry_path}") from error
    except json.JSONDecodeError as error:
        raise RegistryValidationError(
            f"Invalid JSON in dataset registry {registry_path}: "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    root = _require_object(raw, str(registry_path))
    if not root:
        raise _validation_error(str(registry_path), "registry must contain a dataset")

    registry: dict[str, DatasetSpec] = {}
    aliases: dict[str, str] = {}
    for name, value in root.items():
        if not isinstance(name, str):
            raise _validation_error(str(registry_path), "dataset names must be strings")
        spec = _dataset_spec(name, value, f"{registry_path}:{name}")
        registry[name] = spec

        for candidate in (name, *spec.aliases):
            normalized = normalize_dataset_name(candidate)
            existing = aliases.get(normalized)
            if existing is not None:
                raise _validation_error(
                    f"{registry_path}:{name}.aliases",
                    f"name or alias {candidate!r} conflicts with dataset {existing!r}",
                )
            aliases[normalized] = name

    return MappingProxyType(registry), MappingProxyType(aliases)


def _load_packaged_registry() -> tuple[Mapping[str, DatasetSpec], Mapping[str, str]]:
    resource = files("mlsca_bench.datasets").joinpath("registry.json")
    with resource.open("r", encoding="utf-8") as stream:
        try:
            raw = json.load(stream)
        except json.JSONDecodeError as error:
            raise RegistryValidationError(
                "Invalid JSON in packaged dataset registry: "
                f"line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error

    root = _require_object(raw, "registry.json")
    if not root:
        raise _validation_error("registry.json", "registry must contain a dataset")

    registry: dict[str, DatasetSpec] = {}
    aliases: dict[str, str] = {}
    for name, value in root.items():
        spec = _dataset_spec(name, value, f"registry.json:{name}")
        registry[name] = spec
        for candidate in (name, *spec.aliases):
            normalized = normalize_dataset_name(candidate)
            existing = aliases.get(normalized)
            if existing is not None:
                raise _validation_error(
                    f"registry.json:{name}.aliases",
                    f"name or alias {candidate!r} conflicts with dataset {existing!r}",
                )
            aliases[normalized] = name

    return MappingProxyType(registry), MappingProxyType(aliases)


_PACKAGED_REGISTRY, _PACKAGED_ALIASES = _load_packaged_registry()

# The live registry is a read-only view over a mutable store, so datasets added
# with register_dataset() are visible to every consumer of get_dataset().
_STORE: dict[str, DatasetSpec] = dict(_PACKAGED_REGISTRY)
_ALIAS_STORE: dict[str, str] = dict(_PACKAGED_ALIASES)
_REGISTRY: Mapping[str, DatasetSpec] = MappingProxyType(_STORE)
_ALIASES: Mapping[str, str] = MappingProxyType(_ALIAS_STORE)
_USER_DATASETS: set[str] = set()
_LOCAL_PATHS: dict[str, Path] = {}

REGISTRY_ENV = "MLSCA_BENCH_REGISTRY"


def _fill_user_defaults(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Fill the bookkeeping fields a user entry may omit."""

    item = dict(entry)
    item.setdefault("display_name", name)
    item.setdefault("aliases", [])
    item.setdefault("files", [])
    item.setdefault("availability", "automatic" if item["files"] else "manual")
    for field in ("homepage", "paper", "license", "notes"):
        item.setdefault(field, None)
    return item


def _prepare_user_dataset(
    name: str,
    entry: Mapping[str, Any],
    *,
    path: str | Path | None,
    base_dir: Path | None,
    location: str,
) -> tuple[DatasetSpec, Path | None]:
    if not isinstance(name, str) or not name.strip():
        raise _validation_error(location, "dataset name must be a non-empty string")
    canonical = normalize_dataset_name(name)
    item = dict(_require_object(entry, location))
    entry_path = item.pop("path", None)
    local = path if path is not None else entry_path
    local_path = None
    if local is not None:
        if not isinstance(local, (str, Path)) or not str(local).strip():
            raise _validation_error(f"{location}.path", "expected a file or directory path")
        local_path = Path(local).expanduser()
        if base_dir is not None and not local_path.is_absolute():
            # Resolve '..' on the path text, as the user wrote it, so a registry
            # file reached through a symlinked folder still points where intended.
            local_path = Path(os.path.normpath(base_dir / local_path))
    spec = _dataset_spec(canonical, _fill_user_defaults(canonical, item), location)
    if spec.adapter is None:
        raise _validation_error(
            f"{location}.adapter", "a dataset needs an adapter to be loadable"
        )
    known = sorted({s.adapter for s in _PACKAGED_REGISTRY.values() if s.adapter})
    if spec.adapter not in known:
        close = difflib.get_close_matches(spec.adapter, known, n=1)
        raise _validation_error(
            f"{location}.adapter",
            f"unknown adapter {spec.adapter!r}"
            + (f"; did you mean {close[0]!r}?" if close else "")
            + f" Available: {', '.join(known)}",
        )
    return spec, local_path


def _check_conflicts(
    spec: DatasetSpec, *, overwrite: bool, location: str, pending: Mapping[str, str]
) -> None:
    for candidate in (spec.name, *spec.aliases):
        normalized = normalize_dataset_name(candidate)
        existing = pending.get(normalized) or _ALIAS_STORE.get(normalized)
        if existing is None:
            continue
        if existing in _PACKAGED_REGISTRY:
            raise _validation_error(
                location,
                f"name or alias {candidate!r} is already used by the built-in "
                f"dataset {existing!r}; choose another name",
            )
        if existing == spec.name:
            if overwrite:
                continue
            raise _validation_error(
                location,
                f"dataset {spec.name!r} is already registered; pass overwrite=True to replace it",
            )
        raise _validation_error(
            location, f"name or alias {candidate!r} conflicts with dataset {existing!r}"
        )


def _remove_user_dataset(name: str) -> None:
    for alias, target in list(_ALIAS_STORE.items()):
        if target == name:
            del _ALIAS_STORE[alias]
    _STORE.pop(name, None)
    _LOCAL_PATHS.pop(name, None)
    _USER_DATASETS.discard(name)


def _commit_user_dataset(spec: DatasetSpec, local_path: Path | None) -> None:
    if spec.name in _USER_DATASETS:
        _remove_user_dataset(spec.name)
    _STORE[spec.name] = spec
    for candidate in (spec.name, *spec.aliases):
        _ALIAS_STORE[normalize_dataset_name(candidate)] = spec.name
    if local_path is not None:
        _LOCAL_PATHS[spec.name] = local_path
    _USER_DATASETS.add(spec.name)


def register_dataset(
    name: str,
    entry: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    overwrite: bool = False,
    save: bool = False,
) -> DatasetSpec:
    """Add your own dataset to the registry.

    By default the dataset is known for this Python session only. Pass
    ``save=True`` to also store it in your personal registry file
    (:func:`user_registry_path`), which is loaded automatically every time
    ``mlsca_bench`` is imported. Remove it again with
    ``unregister_dataset(name, forget=True)``.

    ``entry`` uses the same schema as an entry of the packaged ``registry.json``
    and is validated with the same rules. Only ``adapter`` (and usually
    ``adapter_config`` and ``metadata``) is required; ``display_name``,
    ``aliases``, ``files``, ``availability``, ``homepage``, ``paper``,
    ``license`` and ``notes`` get sensible defaults.

    Pass ``path`` (or a ``"path"`` field in ``entry``) for data already on
    disk: :func:`load_dataset` then reads it from there instead of
    downloading. Datasets downloaded over HTTP still need a ``known_hash`` for
    every file.

    Built-in datasets cannot be replaced. Re-registering one of your own
    datasets requires ``overwrite=True``.
    """

    location = f"register_dataset({name!r})"
    spec, local_path = _prepare_user_dataset(
        name, entry, path=path, base_dir=None, location=location
    )
    _check_conflicts(
        spec, overwrite=overwrite or _is_saved_replacement(name), location=location, pending={}
    )
    if save:
        _save_user_entries([(spec.name, entry, local_path)])   # validated above; write before committing
    _commit_user_dataset(spec, local_path)
    return spec


def unregister_dataset(name: str, *, forget: bool = False) -> None:
    """Remove a dataset added with :func:`register_dataset` or a registry file.

    With ``forget=True`` it is also deleted from your personal registry file,
    so it no longer comes back in new sessions.
    """

    canonical = _ALIAS_STORE.get(normalize_dataset_name(name))
    if canonical is None:
        raise UnknownDatasetError(f"Unknown dataset {name!r}.")
    if canonical not in _USER_DATASETS:
        raise DatasetRegistryError(
            f"Dataset {canonical!r} is built in and cannot be unregistered."
        )
    if forget:
        _forget_user_entry(canonical)
    _remove_user_dataset(canonical)


def load_registry_file(
    path: str | Path, *, overwrite: bool = False, save: bool = False
) -> tuple[str, ...]:
    """Register every dataset in a JSON file of your own.

    The file maps dataset names to entries, like the packaged
    ``registry.json``. A relative ``"path"`` in an entry is resolved against
    the file's directory. The file is validated completely before anything is
    registered, so a single bad entry leaves the registry unchanged. Returns
    the canonical names that were registered. ``save=True`` also stores every
    entry in your personal registry file, like ``register_dataset(save=True)``.
    """

    registry_path = Path(path).expanduser()
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryValidationError(f"Dataset registry not found: {registry_path}") from error
    except json.JSONDecodeError as error:
        raise RegistryValidationError(
            f"Invalid JSON in dataset registry {registry_path}: "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    root = _require_object(raw, str(registry_path))

    prepared: list[tuple[DatasetSpec, Path | None]] = []
    pending: dict[str, str] = {}
    to_save: list[tuple[str, Mapping[str, Any], Path | None]] = []
    for name, entry in root.items():
        location = f"{registry_path}:{name}"
        spec, local_path = _prepare_user_dataset(
            name, entry, path=None, base_dir=registry_path.parent, location=location
        )
        _check_conflicts(
            spec, overwrite=overwrite or _is_saved_replacement(name), location=location, pending=pending
        )
        for candidate in (spec.name, *spec.aliases):
            pending[normalize_dataset_name(candidate)] = spec.name
        prepared.append((spec, local_path))
        to_save.append((spec.name, entry, local_path))

    if save:
        _save_user_entries(to_save)
    for spec, local_path in prepared:
        _commit_user_dataset(spec, local_path)
    return tuple(spec.name for spec, _ in prepared)


def is_user_dataset(name: str) -> bool:
    """Return whether a dataset was added by the user rather than built in."""

    canonical = _ALIAS_STORE.get(normalize_dataset_name(name))
    return canonical is not None and canonical in _USER_DATASETS


def local_dataset_path(name: str) -> Path | None:
    """The on-disk location registered for a user dataset, if any."""

    canonical = _ALIAS_STORE.get(normalize_dataset_name(name))
    return _LOCAL_PATHS.get(canonical) if canonical is not None else None


def _load_registry_env() -> None:
    value = os.environ.get(REGISTRY_ENV, "")
    for entry in (part for part in value.split(os.pathsep) if part.strip()):
        try:
            load_registry_file(entry)
        except DatasetRegistryError as error:
            raise RegistryValidationError(
                f"Could not load the registry file listed in ${REGISTRY_ENV}: {error}"
            ) from error


USER_REGISTRY_ENV = "MLSCA_BENCH_USER_REGISTRY"
_SAVED_LOADED: set[str] = set()


def user_registry_path() -> Path:
    """Where datasets saved with ``register_dataset(..., save=True)`` are kept.

    ``$MLSCA_BENCH_USER_REGISTRY`` if set; otherwise the platform's config
    folder: ``~/Library/Application Support/mlsca-bench/datasets.json`` on
    macOS, ``%APPDATA%\\mlsca-bench\\datasets.json`` on Windows and
    ``$XDG_CONFIG_HOME`` (default ``~/.config``) ``/mlsca-bench/datasets.json``
    elsewhere.
    """

    configured = os.environ.get(USER_REGISTRY_ENV)
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "mlsca-bench" / "datasets.json"


def _read_user_registry() -> dict[str, Any]:
    path = user_registry_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryValidationError(f"Invalid JSON in your saved datasets file {path}: {error}") from error
    return _require_object(raw, str(path))


def _write_user_registry(entries: Mapping[str, Any]) -> None:
    path = user_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(entries), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)                         # atomic: never leaves a half-written file


def _save_user_entries(items: list[tuple[str, Mapping[str, Any], Path | None]]) -> None:
    entries = _read_user_registry()
    for name, entry, local_path in items:
        item = dict(entry)
        item.pop("path", None)
        if local_path is not None:
            item["path"] = str(Path(local_path).expanduser().resolve())
        try:
            json.dumps(item)
        except TypeError as error:
            raise RegistryValidationError(f"Dataset {name!r} cannot be saved as JSON: {error}") from error
        entries[name] = item
    _write_user_registry(entries)
    _SAVED_LOADED.update(name for name, _, _ in items)


def _is_saved_replacement(name: str) -> bool:
    """Registering a saved dataset again (e.g. an old script) replaces it instead of failing."""

    return _ALIAS_STORE.get(normalize_dataset_name(name)) in _SAVED_LOADED


def _forget_user_entry(name: str) -> None:
    _SAVED_LOADED.discard(name)
    entries = _read_user_registry()
    if name in entries:
        del entries[name]
        _write_user_registry(entries)


def saved_datasets() -> tuple[str, ...]:
    """Names of the datasets stored in your personal registry file."""

    return tuple(sorted(_read_user_registry()))


def _load_user_registry() -> None:
    path = user_registry_path()
    if not path.exists() or not _read_user_registry():
        return
    try:
        _SAVED_LOADED.update(load_registry_file(path))
    except DatasetRegistryError as error:
        raise RegistryValidationError(
            f"Could not load your saved datasets from {path}: {error} "
            "Fix or delete the entry, or set $MLSCA_BENCH_USER_REGISTRY to another file."
        ) from error


_load_user_registry()
_load_registry_env()


def list_datasets(*, builtin_only: bool = False) -> tuple[DatasetSpec, ...]:
    """Return all registered datasets sorted by canonical name.

    Includes your own datasets unless ``builtin_only=True``.
    """

    names = sorted(_PACKAGED_REGISTRY if builtin_only else _REGISTRY)
    return tuple(_REGISTRY[name] for name in names)


def get_dataset(name: str) -> DatasetSpec:
    """Resolve a canonical dataset name or alias to its specification."""

    normalized = normalize_dataset_name(name)
    canonical_name = _ALIASES.get(normalized)
    if canonical_name is not None:
        return _REGISTRY[canonical_name]

    suggestions = difflib.get_close_matches(normalized, _ALIASES, n=3, cutoff=0.55)
    hint = ""
    if suggestions:
        display = sorted({_ALIASES[suggestion] for suggestion in suggestions})
        hint = f" Did you mean: {', '.join(display)}?"
    hint += (
        " If this is your own dataset: register_dataset() without save=True only "
        "lasts for the Python session that called it. Use register_dataset(..., "
        "save=True) or `mlsca-bench add FILE.json` to keep it for every session. "
        "See all names with `mlsca-bench list`."
    )
    raise UnknownDatasetError(f"Unknown dataset {name!r}.{hint}")


def has_dataset(name: str) -> bool:
    """Return whether a canonical dataset name or alias is registered."""

    try:
        normalized = normalize_dataset_name(name)
    except UnknownDatasetError:
        return False
    return normalized in _ALIASES


__all__ = [
    "ArchiveType",
    "Availability",
    "DatasetFile",
    "DatasetMetadata",
    "DatasetRegistryError",
    "DatasetSpec",
    "REGISTRY_ENV",
    "RegistryValidationError",
    "UnknownDatasetError",
    "format_size",
    "get_dataset",
    "has_dataset",
    "is_user_dataset",
    "list_datasets",
    "load_registry",
    "load_registry_file",
    "local_dataset_path",
    "normalize_dataset_name",
    "register_dataset",
    "saved_datasets",
    "unregister_dataset",
    "user_registry_path",
    "USER_REGISTRY_ENV",
]
