# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Dataset discovery, downloads, and common data interfaces."""

from .base import (
    ArraySideChannelDataset,
    DatasetArray,
    SideChannelDataset,
    TraceSample,
    broadcast_field,
    validate_dataset,
)

from .cache import CachedDataset, cache_usage, cached_datasets, remove_cached_dataset
from .download import cache_root, download_dataset
from .errors import (
    CustomDownloaderRequired,
    DatasetDownloadError,
    DatasetUnavailable,
    ManualDownloadRequired,
    MissingDependencyError,
)
from .loading import load_dataset
from .registry import (
    DatasetFile,
    DatasetMetadata,
    DatasetSpec,
    RegistryValidationError,
    get_dataset,
    list_datasets,
    load_registry_file,
    register_dataset,
    saved_datasets,
    unregister_dataset,
    user_registry_path,
)

__all__ = [
    "ArraySideChannelDataset",
    "CachedDataset",
    "CustomDownloaderRequired",
    "DatasetArray",
    "DatasetDownloadError",
    "DatasetFile",
    "DatasetMetadata",
    "DatasetSpec",
    "DatasetUnavailable",
    "ManualDownloadRequired",
    "MissingDependencyError",
    "RegistryValidationError",
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
