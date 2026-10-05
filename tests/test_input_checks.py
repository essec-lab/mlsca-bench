# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Clear errors for common input mistakes found in the user trials."""

from __future__ import annotations

import numpy as np
import pytest

import mlsca_bench
from mlsca_bench import load_dataset
from mlsca_bench.benchmark import LeakageModel


def test_version_is_exposed():
    assert isinstance(mlsca_bench.__version__, str) and mlsca_bench.__version__


@pytest.mark.parametrize("byte", [-1, 16, 2.0, True])
def test_key_byte_must_be_0_to_15(byte):
    with pytest.raises(ValueError, match="from 0 to 15"):
        LeakageModel(byte=byte)


def test_not_found_error_names_the_path(tmp_path):
    missing = tmp_path / "nope.h5"
    with pytest.raises(FileNotFoundError, match=r"nope\.h5' \(does not exist\)"):
        load_dataset("ascadf", path=missing)
