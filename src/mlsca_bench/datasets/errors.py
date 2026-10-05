# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised while obtaining registered datasets."""

from __future__ import annotations


class DatasetDownloadError(RuntimeError):
    """Base exception for failures that occur before or during a download."""


class MissingDependencyError(ImportError):
    """Raised when an optional dataset feature needs another package."""

    def __init__(
        self,
        dependency: str,
        *,
        feature: str,
        extra: str | None = None,
        install_hint: str | None = None,
    ) -> None:
        self.dependency = dependency
        self.extra = extra
        self.feature = feature
        if install_hint is not None:          # e.g. a system tool such as 7z or unar
            hint = install_hint
        elif extra is not None:
            hint = f"Install it with 'python -m pip install mlsca-bench[{extra}]'."
        else:
            hint = f"Install it with 'python -m pip install {dependency}'."
        super().__init__(f"{feature} requires the optional dependency {dependency!r}. {hint}")


class LongPathsRequired(DatasetDownloadError):
    """Raised on Windows when extracting would exceed 260-character paths."""


class InsufficientDiskSpace(DatasetDownloadError):
    """Raised before a download that cannot fit in the free disk space."""


class ManualDownloadRequired(DatasetDownloadError):
    """Raised when a dataset cannot be downloaded without user interaction."""

    def __init__(self, dataset_name: str, instructions_url: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.instructions_url = instructions_url
        message = f"Dataset {dataset_name!r} requires a manual download."
        if instructions_url:
            message += f" Follow the instructions at {instructions_url}"
        super().__init__(message)


class DatasetUnavailable(DatasetDownloadError):
    """Raised when no usable public download is known for a dataset."""

    def __init__(self, dataset_name: str, information_url: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.information_url = information_url
        message = f"Dataset {dataset_name!r} is not currently available for download."
        if information_url:
            message += f" See {information_url} for more information."
        super().__init__(message)


class CustomDownloaderRequired(DatasetDownloadError):
    """Raised when a provider-specific downloader has not been implemented."""

    def __init__(self, dataset_name: str, provider_url: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.provider_url = provider_url
        message = f"Dataset {dataset_name!r} requires a custom downloader."
        if provider_url:
            message += f" Provider page: {provider_url}"
        super().__init__(message)


class ArchiveToolMissing(DatasetDownloadError):
    """Raised when a system tool needed to extract an archive is not installed."""

    def __init__(self, tool_names: str, feature: str, instructions_url: str | None = None) -> None:
        self.tool_names = tool_names
        message = (
            f"{feature} needs one of the system tools [{tool_names}], which was "
            "not found. Either install one and retry (Windows: 7-Zip from https://www.7-zip.org; "
            "macOS: 'brew install unar'; Debian/Ubuntu: 'apt-get install unar' or 'unrar'), or download and "
            "extract the archive manually and load it via "
            "load_dataset(name, path='/path/to/extracted.h5')."
        )
        if instructions_url:
            message += f" Dataset page: {instructions_url}"
        super().__init__(message)


__all__ = [
    "ArchiveToolMissing",
    "CustomDownloaderRequired",
    "DatasetDownloadError",
    "InsufficientDiskSpace",
    "LongPathsRequired",
    "DatasetUnavailable",
    "ManualDownloadRequired",
    "MissingDependencyError",
]


def describe_paths(paths) -> str:
    """The searched location(s) for a not-found error, flagging paths that do not exist."""

    from pathlib import Path

    items = [Path(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    parts = [f"'{p}'" + ("" if p.exists() else " (does not exist)") for p in items[:5]]
    more = f" and {len(items) - 5} more" if len(items) > 5 else ""
    return ", ".join(parts) + more if parts else "the supplied path"
