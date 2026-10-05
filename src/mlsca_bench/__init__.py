# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Public side-channel datasets, baselines and metrics for machine-learning side-channel analysis."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("mlsca-bench")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

from .datasets import (
    ArraySideChannelDataset,
    CustomDownloaderRequired,
    DatasetDownloadError,
    DatasetFile,
    DatasetMetadata,
    DatasetSpec,
    DatasetUnavailable,
    DatasetArray,
    ManualDownloadRequired,
    MissingDependencyError,
    SideChannelDataset,
    TraceSample,
    broadcast_field,
    cache_root,
    cache_usage,
    cached_datasets,
    validate_dataset,
    download_dataset,
    get_dataset,
    list_datasets,
    load_dataset,
    load_registry_file,
    register_dataset,
    remove_cached_dataset,
    saved_datasets,
    unregister_dataset,
    user_registry_path,
)

__all__ = [
    "__version__",
    "ArraySideChannelDataset",
    "CustomDownloaderRequired",
    "DatasetDownloadError",
    "DatasetFile",
    "DatasetMetadata",
    "DatasetSpec",
    "DatasetUnavailable",
    "DatasetArray",
    "ManualDownloadRequired",
    "MissingDependencyError",
    "SideChannelDataset",
    "TraceSample",
    "broadcast_field",
    "cache_root",
    "cache_usage",
    "cached_datasets",
    "download_dataset",
    "get_dataset",
    "list_datasets",
    "load_dataset",
    "load_registry_file",
    "register_dataset",
    "remove_cached_dataset",
    "saved_datasets",
    "unregister_dataset",
    "user_registry_path",
    "validate_dataset",
]
