# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""A missing optional package must give an actionable message, never a TypeError."""

from __future__ import annotations

import sys

import pytest

from mlsca_bench import download_dataset
from mlsca_bench.datasets.errors import MissingDependencyError


@pytest.mark.parametrize(
    ("dataset", "module", "extra"),
    [
        ("present-2021-randomized-clock", "gdown", "gdrive"),        # Google Drive
        ("chameleon-base", "huggingface_hub", "huggingface"),        # Hugging Face
        ("scaaml-ecc-gpam-cm0", "google.cloud", "gcs"),              # Google Cloud Storage
    ],
)
def test_missing_download_backend_names_the_extra(tmp_path, monkeypatch, dataset, module, extra):
    monkeypatch.setitem(sys.modules, module, None)                 # makes the import fail
    with pytest.raises(MissingDependencyError) as info:
        download_dataset(dataset, tmp_path, progress=False)
    assert f"mlsca-bench[{extra}]" in str(info.value)
    assert info.value.extra == extra


def test_message_with_extra():
    error = MissingDependencyError("gdown", extra="gdrive", feature="Google Drive downloads")
    assert str(error) == ("Google Drive downloads requires the optional dependency 'gdown'. "
                          "Install it with 'python -m pip install mlsca-bench[gdrive]'.")


def test_message_with_system_tool_hint():
    error = MissingDependencyError("7-Zip", feature="extracting archives", install_hint="Install the '7z' command.")
    assert str(error).endswith("Install the '7z' command.")
    assert error.extra is None


def test_message_without_extra_falls_back_to_pip():
    assert "pip install pooch" in str(MissingDependencyError("pooch", feature="dataset downloading"))
