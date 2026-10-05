# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end profiled attack: split -> train -> predict -> Guessing Entropy.

Ties the framework-agnostic pieces (splits, leakage model, metrics) to a PyTorch
baseline and returns the standard attack metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from ..benchmark.leakage import LeakageModel
from ..benchmark.metrics import RankResult, evaluate_key_rank
from ..benchmark.splits import SplitPolicy, make_splits
from ..datasets.base import SideChannelDataset
from ..datasets.errors import MissingDependencyError
from .registry import OneCycle, TrainingSettings, build_model, get_training_settings

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise MissingDependencyError(
        "torch", extra="eval", feature="DL-SCA training and attacks"
    ) from error


@dataclass
class AttackResult:
    """Outcome of a profiled attack."""

    ranks: RankResult
    model: "nn.Module"
    n_train: int
    n_test: int
    training: TrainingSettings | None = None     # the settings the model was trained with

    @property
    def guessing_entropy(self) -> np.ndarray:
        return self.ranks.guessing_entropy

    @property
    def success_rate(self) -> np.ndarray:
        return self.ranks.success_rate

    @property
    def traces_to_disclosure(self) -> int | None:
        return self.ranks.traces_to_disclosure()


def _materialize(dataset: SideChannelDataset) -> np.ndarray:
    return np.asarray(dataset.traces[:], dtype=np.float32)


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample z-score with the training mean and standard deviation."""

    x_train = np.asarray(x_train, dtype=np.float64)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-12
    scale = lambda x: ((np.asarray(x, dtype=np.float64) - mean) / std).astype(np.float32)  # noqa: E731
    return scale(x_train), scale(x_test)


def _min_max(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample scaling to [0, 1] with the training minimum and maximum."""

    low, high = x_train.min(axis=0), x_train.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    scale = lambda x: ((x - low) / span).astype(np.float32)  # noqa: E731
    return scale(x_train), scale(x_test)


_SCALINGS = ("none", "standard", "standard+minmax")


def scale_traces(x_train: np.ndarray, x_test: np.ndarray, scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``scaling`` (fitted on the training traces) to both sets."""

    if scaling not in _SCALINGS:
        raise ValueError(f"Unknown scaling {scaling!r}; choose one of {', '.join(_SCALINGS)}.")
    if scaling == "none":
        return x_train, x_test
    x_train, x_test = _standardize(x_train, x_test)
    if scaling == "standard+minmax":
        x_train, x_test = _min_max(x_train, x_test)
    return x_train, x_test


def one_cycle_lr(iteration: int, total: int, max_lr: float, schedule: OneCycle) -> float:
    """The learning rate after ``iteration`` batches (Zaid et al.'s ``compute_lr``)."""

    mid = int(total * (1 - schedule.end_percentage) / 2)
    scale = schedule.scale_percentage
    if mid <= 0:
        return max_lr * scale
    if iteration > 2 * mid:
        done = (iteration - 2 * mid) / max(total - 2 * mid, 1)
        return max_lr * (1 + done * (1 - 100) / 100) * scale
    if iteration > mid:
        done = 1 - (iteration - mid) / mid
    else:
        done = iteration / mid
    return max_lr * (1 + done * (scale * 100 - 1)) * scale


def seed_everything(seed: int) -> None:
    """Seed every random source a training run uses, for repeatable results.

    Called before the model is built, so the initial weights are seeded too,
    and again before training, so the batch order is. On GPUs, cuDNN is set
    to its deterministic algorithms.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    model: "nn.Module",
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    optimizer: str = "adam",
    schedule: OneCycle | None = None,
    device: str | None = None,
    seed: int = 0,
) -> "nn.Module":
    """Train a classifier on (traces, labels) with cross-entropy.

    ``optimizer`` is ``"adam"`` or ``"rmsprop"``, with the Keras default constants
    (the reference code of the baselines is Keras). With a ``schedule`` the rate
    changes after every batch and ``lr`` is its maximum. The order of the training
    traces is shuffled with its own generator, derived from ``seed``, so a run
    repeats exactly.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    shuffle_seed = int(np.random.SeedSequence([seed, 0x5EED]).generate_state(1)[0])
    shuffle_rng = torch.Generator().manual_seed(shuffle_seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_rng,
    )
    if optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    elif optimizer == "rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-7)
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}; choose 'adam' or 'rmsprop'.")
    total = epochs * len(loader)
    step = 0

    def set_lr() -> None:
        if schedule is not None:
            for group in optim.param_groups:
                group["lr"] = one_cycle_lr(step, total, lr, schedule)

    set_lr()
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
            step += 1
            set_lr()
    return model


def predict_proba(
    model: "nn.Module",
    x: np.ndarray,
    *,
    device: str | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """Class probabilities ``(n, n_classes)`` for the given traces."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            logits = model(batch.to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def run_attack(
    name: str,
    *,
    model: str = "mlp",
    model_kwargs: dict[str, Any] | None = None,
    leakage: LeakageModel | None = None,
    policy: SplitPolicy | None = None,
    path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    optimizer: str | None = None,
    scaling: str | None = None,
    device: str | None = None,
    seed: int = 0,
    n_experiments: int = 100,
    standardize: bool | None = None,
    train_traces: int | None = None,
    **load_kwargs: Any,
) -> AttackResult:
    """Run a full profiled attack on a registered dataset and score it.

    Splits the dataset (honoring the author split), trains ``model`` on the
    training traces to predict the leakage label, predicts on the attack/test
    set, and returns Guessing Entropy / Success Rate.

    ``leakage`` defaults to ``LeakageModel()``: AES first-round S-box output,
    identity labels (256 classes), key byte 0. For ASCAD v1 pass
    ``LeakageModel(byte=2)``: the ASCAD databases only contain the window around
    key byte 2 (the standard, first masked target), so other bytes give random
    results.

    ``model_kwargs`` is forwarded to the model builder, so its options (e.g.
    ``hidden`` for ``mlp``) can be changed without a custom builder.

    Training follows the model's own settings (``get_training_settings(model)``):
    for the baselines these are the settings of the authors' reference code, for
    ``mlp`` and your own models Adam, learning rate 1e-3, 50 epochs, batch size 128
    and z-scored traces. ``epochs``, ``batch_size``, ``lr``, ``optimizer`` and
    ``scaling`` (``"none"``, ``"standard"``, ``"standard+minmax"``) override them;
    ``standardize=True`` / ``False`` is a shortcut for ``scaling="standard"`` /
    ``"none"``. The settings used are returned in ``result.training``.

    ``train_traces`` trains on only the first N training traces (the attack set
    is unchanged), e.g. to try a large model on a CPU.
    """

    if train_traces is not None and (isinstance(train_traces, bool) or not isinstance(train_traces, int)
                                     or train_traces < 1):
        raise ValueError("train_traces must be a positive number of traces.")
    if standardize is not None and scaling is not None:
        raise ValueError("Pass either scaling= or standardize=, not both.")
    if standardize is not None:
        scaling = "standard" if standardize else "none"
    overrides = {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                 "optimizer": optimizer, "scaling": scaling}
    training = replace(get_training_settings(model),
                       **{key: value for key, value in overrides.items() if value is not None})
    if training.optimizer not in ("adam", "rmsprop"):
        raise ValueError(f"Unknown optimizer {training.optimizer!r}; choose 'adam' or 'rmsprop'.")
    if training.scaling not in _SCALINGS:
        raise ValueError(f"Unknown scaling {training.scaling!r}; choose one of {', '.join(_SCALINGS)}.")
    leakage = leakage or LeakageModel()
    with make_splits(name, policy=policy, path=path, **load_kwargs) as splits:
        x_train = _materialize(splits.train)
        y_train = leakage.dataset_labels(splits.train)
        if train_traces is not None:
            x_train, y_train = x_train[:train_traces], y_train[:train_traces]
        x_test = _materialize(splits.test)
        hypotheses = leakage.dataset_hypotheses(splits.test)
        correct_key = leakage.true_key_byte(splits.test)

    x_train, x_test = scale_traces(x_train, x_test, training.scaling)

    seed_everything(seed)              # seeds the initial weights
    net = build_model(
        model,
        input_length=x_train.shape[1],
        n_classes=leakage.n_classes,
        **(model_kwargs or {}),
    )
    net = train_model(
        net, x_train, y_train,
        epochs=training.epochs, batch_size=training.batch_size, lr=training.lr,
        optimizer=training.optimizer, schedule=training.schedule, device=device, seed=seed,
    )
    probabilities = predict_proba(net, x_test, device=device)
    ranks = evaluate_key_rank(
        probabilities, hypotheses, correct_key,
        n_experiments=n_experiments, seed=seed,
    )
    return AttackResult(ranks, net, len(x_train), len(x_test), training)

__all__ = [
    "AttackResult",
    "predict_proba",
    "run_attack",
    "train_model",
]
