# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Convolutional DL-SCA baselines, reimplemented from their papers.

* ``ascad_cnn`` — CNN_best from Prouff et al., "Study of Deep Learning
  Techniques for SCA and Introduction to the ASCAD Database" (2018): a VGG-like
  stack of five Conv/AvgPool blocks then two 4096-unit dense layers.
* ``zaid_cnn`` — the ASCAD (desync 0) model from Zaid et al., "Methodology for
  Efficient CNN Architectures in Profiling Attacks" (TCHES 2020): a single tiny
  Conv(4, k=1)+BN+pool block then two 10-unit dense layers, SELU activations.

Each is registered with the training settings of its reference code (see
``get_training_settings``). Both accept trace batches shaped ``(batch, length)`` (a channel axis is added
internally) and return logits; softmax is applied by ``predict_proba``.
"""

from __future__ import annotations

from ..datasets.errors import MissingDependencyError
from .registry import ASCAD_CNN_TRAINING, OneCycle, TrainingSettings, keras_default_init, register_model

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise MissingDependencyError(
        "torch", extra="eval", feature="CNN baselines"
    ) from error


def _pooled_length(length: int, n_pools: int, pool: int = 2) -> int:
    for _ in range(n_pools):
        length //= pool
    return length


class _TraceNet(nn.Module):
    """Adds a channel axis to (batch, length) inputs before the conv stack."""

    features: nn.Module
    classifier: nn.Module

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)
        return self.classifier(self.features(x))


class AscadCNN(_TraceNet):
    def __init__(self, input_length: int, n_classes: int) -> None:
        super().__init__()
        widths = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 512)]
        blocks: list[nn.Module] = []
        for in_ch, out_ch in widths:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=11, padding=5),
                nn.ReLU(),
                nn.AvgPool1d(2),
            ]
        self.features = nn.Sequential(*blocks)
        flat = 512 * _pooled_length(input_length, len(widths))
        if flat <= 0:
            raise ValueError("ascad_cnn needs at least 32 samples per trace.")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes),
        )


class ZaidCNN(_TraceNet):
    def __init__(self, input_length: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=1, padding=0),
            nn.SELU(),
            nn.BatchNorm1d(4, eps=1e-3, momentum=0.01),   # the Keras defaults
            nn.AvgPool1d(2),
        )
        flat = 4 * _pooled_length(input_length, 1)
        if flat <= 0:
            raise ValueError("zaid_cnn needs at least 2 samples per trace.")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, n_classes),
        )
        # As in the reference code: He-uniform weights for the hidden layers,
        # Glorot-uniform for the output layer, zero biases.
        keras_default_init(self)
        output = self.classifier[-1]
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d)) and layer is not output:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")   # Keras he_uniform


# Zaid et al.'s reference code (github.com/gabzai/Methodology-for-efficient-CNN-
# architectures-in-SCA, ASCAD/N0=0/cnn_architecture.py): Adam with their one-cycle
# schedule (maximum rate 5e-3, end_percentage 0.2, scale_percentage 0.1), 50 epochs,
# batch size 50, traces z-scored and then scaled to [0, 1].
ZAID_CNN_TRAINING = TrainingSettings(
    optimizer="adam", lr=5e-3, epochs=50, batch_size=50, scaling="standard+minmax",
    schedule=OneCycle(end_percentage=0.2, scale_percentage=0.1),
    source="Zaid et al. code (ASCAD/N0=0/cnn_architecture.py)",
)


@register_model("ascad_cnn", training=ASCAD_CNN_TRAINING)
def _ascad_cnn(*, input_length: int, n_classes: int) -> "nn.Module":
    return keras_default_init(AscadCNN(input_length, n_classes))


@register_model("zaid_cnn", training=ZAID_CNN_TRAINING)
def _zaid_cnn(*, input_length: int, n_classes: int) -> "nn.Module":
    return ZaidCNN(input_length, n_classes)


__all__ = ["AscadCNN", "ZaidCNN"]
