# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Keep the test suite away from the user's real saved datasets and cache."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Set before mlsca_bench is imported, so the personal registry loaded at import is an empty test file.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="mlsca-bench-tests-"))
os.environ["MLSCA_BENCH_USER_REGISTRY"] = str(_TEST_HOME / "saved-datasets.json")
os.environ.pop("MLSCA_BENCH_REGISTRY", None)


import shutil  # noqa: E402

import pytest  # noqa: E402

_REAL_DISK_USAGE = shutil.disk_usage


@pytest.fixture(autouse=True)
def _plenty_of_disk_space(monkeypatch):
    """CI machines have little free disk; tests that fake their downloads should not
    trip the free-space check. Tests of the check set their own value."""

    plenty = type("Usage", (), {"total": 10**15, "used": 0, "free": 10**15})()
    monkeypatch.setattr(shutil, "disk_usage", lambda path: plenty)
