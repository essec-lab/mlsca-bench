# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from mlsca_bench.benchmark.leakage import SBOX, LeakageModel
from mlsca_bench.benchmark.metrics import evaluate_key_rank
from mlsca_bench.benchmark.splits import DatasetSplits, SplitPolicy, Subset
from mlsca_bench.datasets.base import ArraySideChannelDataset
from mlsca_bench.models import (
    build_model,
    list_models,
    predict_proba,
    run_attack,
    train_model,
)
from mlsca_bench.models import attack as attack_mod


def test_registry_has_baselines() -> None:
    import torch

    for name in ("mlp", "ascad_mlp", "ascad_cnn", "zaid_cnn"):
        assert name in list_models()
    net = build_model("mlp", input_length=50, n_classes=9)
    assert sum(p.numel() for p in net.parameters()) > 0
    with pytest.raises(KeyError):
        build_model("nope", input_length=10, n_classes=2)

    # CNNs accept (batch, length) trace batches and return per-class logits.
    for name, length in (("ascad_cnn", 128), ("zaid_cnn", 64)):
        model = build_model(name, input_length=length, n_classes=256)
        out = model(torch.zeros(8, length))
        assert out.shape == (8, 256)


def test_pipeline_recovers_key_on_leaky_toy() -> None:
    # Traces = one-hot of the Sbox label -> an MLP learns the mapping quickly,
    # so the correct key must rank first.
    key, n_train, n_test = 42, 1500, 400
    rng = np.random.default_rng(0)
    model = LeakageModel(leakage="id", byte=0)

    def onehot(pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels = model.labels(pt[:, 0], key)
        x = np.eye(256, dtype=np.float32)[labels]
        return x, labels

    pt_tr = rng.integers(0, 256, size=(n_train, 16), dtype=np.uint8)
    pt_te = rng.integers(0, 256, size=(n_test, 16), dtype=np.uint8)
    x_tr, y_tr = onehot(pt_tr)
    x_te, _ = onehot(pt_te)

    net = build_model("mlp", input_length=256, n_classes=256, hidden=(128,))
    net = train_model(net, x_tr, y_tr, epochs=15, batch_size=128, seed=0)
    probs = predict_proba(net, x_te)
    assert probs.shape == (n_test, 256)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    hyps = model.hypothesis_labels(pt_te[:, 0])
    res = evaluate_key_rank(probs, hyps, key, n_experiments=30, seed=0)
    assert res.guessing_entropy[-1] < 5.0            # correct key at/near the top
    assert res.traces_to_disclosure(threshold=0.0) is not None


def _synthetic_splits(monkeypatch: pytest.MonkeyPatch, key: int = 7) -> None:
    def fake_make_splits(name, *, policy=None, path=None, **kw):
        rng = np.random.default_rng(1)

        def ds(n: int) -> ArraySideChannelDataset:
            pt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
            keys = np.full((n, 16), key, dtype=np.uint8)
            traces = rng.standard_normal((n, 24)).astype(np.float32)
            return ArraySideChannelDataset(name="syn", traces=traces, plaintexts=pt, keys=keys)

        train, test = ds(120), ds(60)
        return DatasetSplits(
            Subset(train, np.arange(len(train)), split_label="train"),
            Subset(train, np.arange(10), split_label="val"),
            Subset(test, np.arange(len(test)), split_label="test"),
            policy or SplitPolicy(),
            {"train": np.arange(120), "val": np.arange(10), "test": np.arange(60)},
            (train, test),
        )

    monkeypatch.setattr(attack_mod, "make_splits", fake_make_splits)


@pytest.mark.parametrize("model", ["mlp", "zaid_cnn"])
def test_run_attack_end_to_end(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    _synthetic_splits(monkeypatch, key=7)
    result = run_attack(
        "syn", model=model, epochs=2, batch_size=32, n_experiments=10, seed=0
    )
    assert result.n_train == 120 and result.n_test == 60
    assert result.guessing_entropy.shape == (60,)
    assert result.success_rate.shape == (60,)
    assert np.all(result.guessing_entropy >= 0)


def test_baselines_carry_their_reference_settings() -> None:
    from mlsca_bench.models import DEFAULT_TRAINING, OneCycle, get_training_settings

    assert get_training_settings("mlp") == DEFAULT_TRAINING
    mlp = get_training_settings("ascad_mlp")
    assert (mlp.optimizer, mlp.lr, mlp.epochs, mlp.batch_size, mlp.scaling) == ("rmsprop", 1e-5, 200, 100, "none")
    cnn = get_training_settings("ascad_cnn")
    assert (cnn.optimizer, cnn.lr, cnn.epochs, cnn.batch_size, cnn.scaling) == ("rmsprop", 1e-5, 75, 200, "none")
    zaid = get_training_settings("zaid_cnn")
    assert (zaid.optimizer, zaid.lr, zaid.epochs, zaid.batch_size) == ("adam", 5e-3, 50, 50)
    assert zaid.scaling == "standard+minmax"
    assert zaid.schedule == OneCycle(end_percentage=0.2, scale_percentage=0.1)
    assert all(get_training_settings(name).source for name in list_models())


def test_one_cycle_matches_the_reference_schedule() -> None:
    from mlsca_bench.models import OneCycle
    from mlsca_bench.models.attack import one_cycle_lr

    schedule, total, top = OneCycle(end_percentage=0.2, scale_percentage=0.1), 1000, 5e-3
    mid = 400                                            # int(1000 * 0.8 / 2)
    assert one_cycle_lr(0, total, top, schedule) == pytest.approx(5e-4)     # starts at a tenth
    assert one_cycle_lr(mid, total, top, schedule) == pytest.approx(5e-3)   # peak
    assert one_cycle_lr(2 * mid, total, top, schedule) == pytest.approx(5e-4)
    assert one_cycle_lr(total, total, top, schedule) == pytest.approx(5e-6)  # a hundredth of the start
    rates = [one_cycle_lr(i, total, top, schedule) for i in range(total + 1)]
    assert max(rates) == pytest.approx(5e-3)


def test_run_attack_uses_and_reports_the_model_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    real_train = attack_mod.train_model

    def spy(net, x, y, **kw):
        seen.update(kw, x_max=float(x.max()), x_min=float(x.min()))
        return real_train(net, x, y, **{**kw, "epochs": 1})

    monkeypatch.setattr(attack_mod, "train_model", spy)
    _synthetic_splits(monkeypatch, key=7)
    result = run_attack("syn", model="zaid_cnn", n_experiments=5)
    assert seen["optimizer"] == "adam" and seen["batch_size"] == 50 and seen["schedule"] is not None
    assert seen["x_min"] >= 0.0 and seen["x_max"] <= 1.0 + 1e-6          # standard+minmax
    assert result.training.source.startswith("Zaid")

    result = run_attack("syn", model="ascad_mlp", lr=1e-3, standardize=True, epochs=1, n_experiments=5)
    assert seen["optimizer"] == "rmsprop" and seen["lr"] == 1e-3
    assert result.training.scaling == "standard" and result.training.epochs == 1

    with pytest.raises(ValueError, match="optimizer"):
        run_attack("syn", model="mlp", optimizer="sgd", n_experiments=5)
    with pytest.raises(ValueError, match="not both"):
        run_attack("syn", model="mlp", scaling="none", standardize=True)


def test_own_model_can_bring_its_settings() -> None:
    from mlsca_bench.models import TrainingSettings, get_training_settings, register_model

    mine = TrainingSettings(optimizer="rmsprop", lr=1e-4, epochs=3, batch_size=16, source="my paper")

    @register_model("settings_test_model", training=mine)
    def _build(*, input_length: int, n_classes: int):
        return build_model("mlp", input_length=input_length, n_classes=n_classes, hidden=(8,))

    assert get_training_settings("settings_test_model") == mine
