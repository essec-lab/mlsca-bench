# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Standardized SCA attack metrics: Guessing Entropy, Success Rate, Mean Rank.

Given a model's per-trace class probabilities on the attack set, the leakage
model's per-hypothesis labels, and the true key byte, we rank the 256 key
candidates by their accumulated log-likelihood as more traces are used, and
report:

* **Guessing Entropy (GE)** — the *mean rank* of the correct key vs the number
  of attack traces (0 = broken). "Mean Rank" is the same quantity.
* **Success Rate (SR, order o)** — the fraction of attack experiments in which
  the correct key is within the top ``o`` after N traces.

A single, documented definition (log-likelihood accumulation, averaged over
random attack orderings) — pure NumPy, deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class RankResult:
    """Attack-metric curves over the number of traces used."""

    n_traces: np.ndarray          # (N,) 1..N
    guessing_entropy: np.ndarray  # (N,) mean rank of the correct key
    success_rate: np.ndarray      # (N,) fraction with rank < order

    def traces_to_disclosure(self, threshold: float = 0.0) -> int | None:
        """Smallest trace count at which GE stays <= threshold (else None)."""

        for start in range(len(self.guessing_entropy)):
            if np.all(self.guessing_entropy[start:] <= threshold):
                return int(self.n_traces[start])
        return None


def evaluate_key_rank(
    probabilities: np.ndarray,
    hypothesis_labels: np.ndarray,
    correct_key: int,
    *,
    n_experiments: int = 100,
    max_traces: int | None = None,
    order: int = 1,
    seed: int = 0,
) -> RankResult:
    """Compute GE and Success-Rate curves for a profiled attack.

    ``probabilities``: ``(n_traces, n_classes)`` model outputs on the attack set.
    ``hypothesis_labels``: ``(n_traces, 256)`` class label for each key guess
    (from :meth:`LeakageModel.hypothesis_labels`); values must be < n_classes.
    ``correct_key``: the true key byte (0..255).
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    hypothesis_labels = np.asarray(hypothesis_labels)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2-D (n_traces, n_classes).")
    n_traces, n_classes = probabilities.shape
    if hypothesis_labels.shape[0] != n_traces:
        raise ValueError("probabilities and hypothesis_labels disagree on n_traces.")
    if hypothesis_labels.max(initial=0) >= n_classes:
        raise ValueError("hypothesis_labels index beyond the probability classes.")
    if not 0 <= int(correct_key) < hypothesis_labels.shape[1]:
        raise ValueError("correct_key out of range.")

    log_probs = np.log(probabilities + _EPS)
    # per_hypothesis[i, k] = log P(class predicted for trace i under key guess k)
    per_hypothesis = np.take_along_axis(log_probs, hypothesis_labels, axis=1)

    limit = n_traces if max_traces is None else min(max_traces, n_traces)
    rng = np.random.default_rng(seed)
    key = int(correct_key)

    ge = np.zeros(limit)
    sr = np.zeros(limit)
    n_keys = per_hypothesis.shape[1]
    for _ in range(n_experiments):
        order_idx = rng.permutation(n_traces)[:limit]
        running = np.cumsum(per_hypothesis[order_idx], axis=0)  # (limit, 256)
        # Tiny per-experiment offset breaks exact ties randomly, so an
        # uninformative model averages to a random rank instead of rank 0.
        running = running + rng.standard_normal(n_keys) * 1e-9
        true_score = running[:, key][:, None]
        ranks = (running > true_score).sum(axis=1)  # 0 = correct key is top
        ge += ranks
        sr += ranks < order
    ge /= n_experiments
    sr /= n_experiments
    return RankResult(np.arange(1, limit + 1), ge, sr)


def guessing_entropy(
    probabilities: np.ndarray,
    hypothesis_labels: np.ndarray,
    correct_key: int,
    **kwargs: object,
) -> np.ndarray:
    """The Guessing Entropy (mean-rank) curve. See :func:`evaluate_key_rank`."""

    return evaluate_key_rank(
        probabilities, hypothesis_labels, correct_key, **kwargs  # type: ignore[arg-type]
    ).guessing_entropy


def success_rate(
    probabilities: np.ndarray,
    hypothesis_labels: np.ndarray,
    correct_key: int,
    **kwargs: object,
) -> np.ndarray:
    """The Success-Rate curve. See :func:`evaluate_key_rank`."""

    return evaluate_key_rank(
        probabilities, hypothesis_labels, correct_key, **kwargs  # type: ignore[arg-type]
    ).success_rate


__all__ = ["RankResult", "evaluate_key_rank", "guessing_entropy", "success_rate"]
