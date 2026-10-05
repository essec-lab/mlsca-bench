# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Baseline DL-SCA models and the end-to-end attack runner (PyTorch).

This subpackage requires the optional ``[eval]`` extra (PyTorch). Models are
registered by name so more architectures can be added without touching the
runner (see ``register_model``):

    from mlsca_bench.models import get_model, list_models, run_attack
"""

from __future__ import annotations

from . import cnn as _cnn  # noqa: F401 -- registers ascad_cnn / zaid_cnn
from .attack import AttackResult, predict_proba, run_attack, train_model
from .registry import (
    DEFAULT_TRAINING,
    OneCycle,
    TrainingSettings,
    _protect_builtins,
    build_model,
    get_model,
    get_training_settings,
    list_models,
    register_model,
)

_protect_builtins()                 # the baselines above can no longer be replaced by accident

__all__ = [
    "AttackResult",
    "DEFAULT_TRAINING",
    "OneCycle",
    "TrainingSettings",
    "build_model",
    "get_model",
    "get_training_settings",
    "list_models",
    "predict_proba",
    "register_model",
    "run_attack",
    "train_model",
]
