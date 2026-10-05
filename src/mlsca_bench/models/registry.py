# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Registry of DL-SCA model architectures, keyed by name.

Each builder takes ``input_length`` (number of trace samples) and ``n_classes``
(from the leakage model) and returns a ``torch.nn.Module``. Register new
architectures with :func:`register_model` (or the ``@register_model(name)``
decorator) — the attack runner looks them up by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..datasets.errors import MissingDependencyError

try:
    import torch.nn as nn
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise MissingDependencyError(
        "torch", extra="eval", feature="DL-SCA baseline models"
    ) from error

ModelBuilder = Callable[..., "nn.Module"]
_REGISTRY: dict[str, ModelBuilder] = {}
_TRAINING: dict[str, "TrainingSettings"] = {}
_BUILTIN: set[str] = set()          # filled once the package's own baselines are registered


@dataclass(frozen=True)
class OneCycle:
    """The one-cycle learning-rate schedule of Zaid et al. (their ``clr.py``).

    Updated after every batch: the rate rises linearly from ``lr * scale_percentage``
    to ``lr``, falls back, and in the last ``end_percentage`` of the training drops
    to a hundredth of the starting rate.
    """

    end_percentage: float = 0.1
    scale_percentage: float = 0.1


@dataclass(frozen=True)
class TrainingSettings:
    """How a model is trained when ``run_attack`` is not told otherwise.

    ``optimizer`` is ``"adam"`` or ``"rmsprop"``; ``scaling`` is ``"none"``,
    ``"standard"`` (z-score with the training mean and standard deviation) or
    ``"standard+minmax"`` (z-score, then scaled to [0, 1]); ``source`` says where
    the values come from.
    """

    optimizer: str = "adam"
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    scaling: str = "standard"
    schedule: OneCycle | None = None
    source: str = "MLSCA-Bench default"


DEFAULT_TRAINING = TrainingSettings()


def register_model(
    name: str, *, overwrite: bool = False, training: TrainingSettings | None = None
) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder under ``name`` (use as ``@register_model("my_cnn")``).

    The builder is called as ``builder(input_length=..., n_classes=..., **model_kwargs)``
    and returns a ``torch.nn.Module``. Registering your own name again replaces
    it (e.g. re-running a notebook cell); built-in baseline names are protected
    unless ``overwrite=True``. ``training`` sets how ``run_attack`` trains the
    model by default (``DEFAULT_TRAINING`` if not given).
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("A model name must be a non-empty string.")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if not callable(builder):
            raise TypeError(f"register_model({name!r}) needs a function that builds the model.")
        if name in _BUILTIN and not overwrite:
            raise ValueError(
                f"{name!r} is a built-in baseline; choose another name for your model "
                "(or pass overwrite=True to replace the built-in one)."
            )
        _REGISTRY[name] = builder
        _TRAINING[name] = training or DEFAULT_TRAINING
        return builder

    return decorator


def _protect_builtins() -> None:
    _BUILTIN.update(_REGISTRY)


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model(name: str) -> ModelBuilder:
    if name not in _REGISTRY:
        choices = ", ".join(list_models())
        raise KeyError(f"Unknown model {name!r}. Available: {choices}.")
    return _REGISTRY[name]


def get_training_settings(name: str) -> TrainingSettings:
    """The default training settings of a registered model."""

    get_model(name)
    return _TRAINING[name]


def build_model(name: str, *, input_length: int, n_classes: int, **kwargs: Any) -> "nn.Module":
    return get_model(name)(input_length=input_length, n_classes=n_classes, **kwargs)


def _mlp(input_length: int, n_classes: int, hidden: tuple[int, ...]) -> "nn.Module":
    layers: list[nn.Module] = []
    width = input_length
    for units in hidden:
        layers += [nn.Linear(width, units), nn.ReLU()]
        width = units
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


def keras_default_init(module: "nn.Module") -> "nn.Module":
    """Glorot-uniform weights and zero biases, the Keras defaults the reference code used."""

    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return module


# ANSSI's reference code for the ASCAD paper (github.com/ANSSI-FR/ASCAD,
# ASCAD_train_models.py): RMSprop, learning rate 1e-5, raw traces (no scaling);
# MLP_best trained for 200 epochs with batch size 100, CNN_best for 75 epochs with
# batch size 200 (the settings of the published trained models).
ASCAD_MLP_TRAINING = TrainingSettings(
    optimizer="rmsprop", lr=1e-5, epochs=200, batch_size=100, scaling="none",
    source="ANSSI ASCAD code (ASCAD_train_models.py, mlp_best)",
)
ASCAD_CNN_TRAINING = TrainingSettings(
    optimizer="rmsprop", lr=1e-5, epochs=75, batch_size=200, scaling="none",
    source="ANSSI ASCAD code (ASCAD_train_models.py, cnn_best)",
)


@register_model("mlp")
def _mlp_default(*, input_length: int, n_classes: int, hidden: tuple[int, ...] = (200,) * 4) -> "nn.Module":
    """A small configurable MLP baseline."""

    return _mlp(input_length, n_classes, hidden)


@register_model("ascad_mlp", training=ASCAD_MLP_TRAINING)
def _ascad_mlp(*, input_length: int, n_classes: int) -> "nn.Module":
    """MLP_best from the ASCAD paper (6 hidden layers of 200 ReLU units)."""

    return keras_default_init(_mlp(input_length, n_classes, (200,) * 6))


__all__ = [
    "DEFAULT_TRAINING",
    "OneCycle",
    "TrainingSettings",
    "build_model",
    "get_model",
    "get_training_settings",
    "list_models",
    "register_model",
]
