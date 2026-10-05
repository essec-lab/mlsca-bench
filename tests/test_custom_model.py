# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Feature 'add your own model': register a builder by name and use it like a baseline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from mlsca_bench import register_dataset, unregister_dataset  # noqa: E402
from mlsca_bench.benchmark import HAMMING_WEIGHT, LeakageModel  # noqa: E402
from mlsca_bench.models import run_attack  # noqa: E402
from mlsca_bench.models.registry import build_model, list_models, register_model  # noqa: E402

from test_custom_datasets import ENTRY, _write_leaky_hdf5  # noqa: E402

BUILT = []


@register_model("test_tiny_mlp")
def _tiny(*, input_length: int, n_classes: int, hidden: int = 32):
    BUILT.append((input_length, n_classes, hidden))
    return torch.nn.Sequential(torch.nn.Linear(input_length, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, n_classes))


@pytest.fixture
def leaky_dataset(tmp_path):
    path = tmp_path / "leaky.h5"
    _write_leaky_hdf5(path, n=3000)
    register_dataset("custom-model-test", ENTRY, path=path)
    yield "custom-model-test"
    unregister_dataset("custom-model-test")


def test_registered_model_is_listed_and_built():
    assert "test_tiny_mlp" in list_models()
    net = build_model("test_tiny_mlp", input_length=20, n_classes=9, hidden=8)
    assert net(torch.zeros(2, 20)).shape == (2, 9)


def test_registered_model_runs_an_attack(leaky_dataset):
    BUILT.clear()
    result = run_attack(leaky_dataset, model="test_tiny_mlp", model_kwargs={"hidden": 16},
                        leakage=LeakageModel(byte=0, leakage="hw"), epochs=5, n_experiments=10, device="cpu")
    assert BUILT == [(20, 9, 16)]                                 # our builder, with our options
    assert result.guessing_entropy[-1] < 50                       # learns the planted leak
    assert result.success_rate.shape == result.guessing_entropy.shape


def test_builtin_model_names_are_protected():
    with pytest.raises(ValueError, match="built-in baseline"):
        register_model("mlp")(lambda **k: None)
    register_model("test_tiny_mlp")(_tiny)                        # own name again: replaced, fine
    assert "test_tiny_mlp" in list_models()


def test_train_traces_limits_the_training_set(leaky_dataset):
    result = run_attack(leaky_dataset, model="test_tiny_mlp", leakage=LeakageModel(byte=0, leakage="hw"),
                        epochs=1, n_experiments=5, device="cpu", train_traces=500)
    assert result.n_train == 500
    with pytest.raises(ValueError, match="positive"):
        run_attack(leaky_dataset, model="mlp", train_traces=0)
