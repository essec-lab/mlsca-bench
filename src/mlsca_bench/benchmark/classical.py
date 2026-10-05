# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Classical (non-deep-learning) SCA baselines: CPA, DPA, Gaussian templates.

These give the "classical" half of the paper's Objective 5 and slot into the
same evaluation as the DL models — they return an attack result exposing the
Guessing-Entropy / Success-Rate curves, so the benchmark runner treats them as
just more models (``"cpa"``, ``"dpa"``, ``"template"``).

* **CPA** (Brier et al., 2004) — *non-profiled* Correlation Power Analysis:
  rank key guesses by the peak Pearson correlation between a Hamming-weight
  power model of the S-box output and the traces.
* **DPA** (Kocher et al., 1999) — *non-profiled* single-bit Differential Power
  Analysis: rank key guesses by the peak difference-of-means between the two
  groups a selection bit induces.
* **Template attack** (Chari et al., 2002) — *profiled* Gaussian templates:
  fit a per-class Gaussian on points of interest from the profiling set, then
  score attack traces by likelihood (reuses the log-likelihood key ranking).

Everything here is pure NumPy — no PyTorch needed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from ..datasets.base import SideChannelDataset
from .leakage import LeakageModel
from .metrics import RankResult, evaluate_key_rank
from .splits import SplitPolicy, make_splits

CLASSICAL_ATTACKS = ("cpa", "dpa", "template")

_EPS = 1e-12


@dataclass
class ClassicalAttackResult:
    """Outcome of a classical attack (mirrors models.AttackResult, no net)."""

    ranks: RankResult
    n_train: int
    n_test: int

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


def _checkpoints(limit: int, n_points: int) -> np.ndarray:
    """Trace-count checkpoints: denser early, log-spaced up to ``limit``."""

    if limit <= n_points:
        return np.arange(1, limit + 1)
    pts = np.unique(np.geomspace(1, limit, n_points).round().astype(int))
    if pts[-1] != limit:
        pts = np.append(pts, limit)
    return pts


# ---------------------------------------------------------------------------
# Non-profiled distinguishers (CPA / DPA)
# ---------------------------------------------------------------------------
def _nonprofiled_curves(
    traces: np.ndarray,
    hypotheses: np.ndarray,     # (N, 256): HW values for CPA, S-box out for DPA
    correct_key: int,
    *,
    kind: str,                  # "cpa" | "dpa"
    dpa_bit: int = 7,
    n_experiments: int = 100,
    order: int = 1,
    max_traces: int | None = None,
    n_points: int = 40,
    seed: int = 0,
) -> RankResult:
    """GE / SR curves for a non-profiled distinguisher.

    For each of ``n_experiments`` random trace orderings, the distinguisher is
    accumulated block-by-block up to each checkpoint (so the total matmul cost
    per experiment is one pass over the traces), the correct key is ranked, and
    the ranks are averaged.
    """

    x = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = x.shape
    n_keys = hypotheses.shape[1]
    limit = n_traces if max_traces is None else min(max_traces, n_traces)
    checkpoints = _checkpoints(limit, n_points)
    rng = np.random.default_rng(seed)
    key = int(correct_key)

    if kind == "cpa":
        model = np.asarray(hypotheses, dtype=np.float32)             # HW values
    elif kind == "dpa":
        model = ((np.asarray(hypotheses).astype(np.int64) >> dpa_bit) & 1).astype(np.float32)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown non-profiled kind {kind!r}")

    ge = np.zeros(len(checkpoints))
    sr = np.zeros(len(checkpoints))
    for _ in range(n_experiments):
        idx = rng.permutation(n_traces)[:limit]
        xo = x[idx]
        mo = model[idx]

        sx = np.zeros(n_samples, dtype=np.float64)
        if kind == "cpa":
            sxx = np.zeros(n_samples)
            sh = np.zeros(n_keys)
            shh = np.zeros(n_keys)
            sxh = np.zeros((n_keys, n_samples))
        else:
            s1 = np.zeros((n_keys, n_samples))
            c1 = np.zeros(n_keys)

        prev = 0
        for ci, cp in enumerate(checkpoints):
            xb = xo[prev:cp].astype(np.float64)
            mb = mo[prev:cp].astype(np.float64)
            sx += xb.sum(axis=0)
            n = cp
            if kind == "cpa":
                sxx += (xb * xb).sum(axis=0)
                sh += mb.sum(axis=0)
                shh += (mb * mb).sum(axis=0)
                sxh += mb.T @ xb
                num = n * sxh - sh[:, None] * sx[None, :]
                den = np.sqrt(
                    np.maximum(n * shh - sh * sh, _EPS)[:, None]
                    * np.maximum(n * sxx - sx * sx, _EPS)[None, :]
                )
                score = np.abs(num / den).max(axis=1)
            else:  # dpa
                s1 += mb.T @ xb
                c1 += mb.sum(axis=0)
                c0 = n - c1
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean1 = s1 / np.where(c1[:, None] > 0, c1[:, None], np.nan)
                    mean0 = (sx[None, :] - s1) / np.where(c0[:, None] > 0, c0[:, None], np.nan)
                dom = np.abs(np.nan_to_num(mean1 - mean0, nan=0.0))
                score = dom.max(axis=1)
            prev = cp

            score = score + rng.standard_normal(n_keys) * 1e-12   # break ties
            rank = int((score > score[key]).sum())
            ge[ci] += rank
            sr[ci] += rank < order

    ge /= n_experiments
    sr /= n_experiments
    return RankResult(np.asarray(checkpoints), ge, sr)


def _relabel(leakage, label: str):
    """The user's leakage model with only its label type changed (any cipher)."""

    if not any(f.name == "leakage" for f in fields(leakage)):
        raise TypeError(f"{type(leakage).__name__} has no label type and cannot be used here.")
    return replace(leakage, leakage=label)


def run_cpa(
    name: str, *, leakage: LeakageModel | None = None, path: str | None = None,
    policy: SplitPolicy | None = None, n_experiments: int = 100, order: int = 1,
    seed: int = 0, n_points: int = 40, max_traces: int | None = None,
    **load_kwargs,
) -> ClassicalAttackResult:
    """Correlation Power Analysis (non-profiled, Hamming-weight power model)."""

    leakage = leakage or LeakageModel()
    hw = _relabel(leakage, "hw")
    with make_splits(name, policy=policy, path=path, **load_kwargs) as splits:
        x = _materialize(splits.test)
        hyp = hw.dataset_hypotheses(splits.test)
        key = hw.true_key_byte(splits.test)
        n_train = len(splits.train)
    ranks = _nonprofiled_curves(
        x, hyp, key, kind="cpa", n_experiments=n_experiments, order=order,
        max_traces=max_traces, n_points=n_points, seed=seed,
    )
    return ClassicalAttackResult(ranks, n_train, len(x))


def run_dpa(
    name: str, *, leakage: LeakageModel | None = None, path: str | None = None,
    policy: SplitPolicy | None = None, dpa_bit: int | None = None, n_experiments: int = 100,
    order: int = 1, seed: int = 0, n_points: int = 40, max_traces: int | None = None,
    **load_kwargs,
) -> ClassicalAttackResult:
    """Single-bit Differential Power Analysis (non-profiled, difference of means).

    ``dpa_bit`` selects the bit of the attacked intermediate; by default its most
    significant bit (bit 7 of the AES S-box output).
    """

    leakage = leakage or LeakageModel()
    idm = _relabel(leakage, "id")
    width = int(idm.n_classes).bit_length() - 1          # bits of the intermediate
    if dpa_bit is None:
        dpa_bit = width - 1
    if not 0 <= dpa_bit < width:
        raise ValueError(f"dpa_bit must be between 0 and {width - 1} for {type(leakage).__name__}.")
    with make_splits(name, policy=policy, path=path, **load_kwargs) as splits:
        x = _materialize(splits.test)
        hyp = idm.dataset_hypotheses(splits.test)
        key = idm.true_key_byte(splits.test)
        n_train = len(splits.train)
    ranks = _nonprofiled_curves(
        x, hyp, key, kind="dpa", dpa_bit=dpa_bit, n_experiments=n_experiments,
        order=order, max_traces=max_traces, n_points=n_points, seed=seed,
    )
    return ClassicalAttackResult(ranks, n_train, len(x))


# ---------------------------------------------------------------------------
# Profiled Gaussian template attack
# ---------------------------------------------------------------------------
def _select_poi(traces: np.ndarray, labels: np.ndarray, n_poi: int, n_classes: int) -> np.ndarray:
    """Top-``n_poi`` samples by SNR = var(class means) / mean(class variances)."""

    n_samples = traces.shape[1]
    class_means = np.zeros((n_classes, n_samples))
    class_vars = np.zeros((n_classes, n_samples))
    present = np.zeros(n_classes, dtype=bool)
    for c in range(n_classes):
        rows = traces[labels == c]
        if len(rows) == 0:
            continue
        present[c] = True
        class_means[c] = rows.mean(axis=0)
        class_vars[c] = rows.var(axis=0)
    signal = class_means[present].var(axis=0)
    noise = class_vars[present].mean(axis=0) + _EPS
    snr = signal / noise
    return np.argsort(snr)[::-1][:n_poi]


def _fit_templates(traces: np.ndarray, labels: np.ndarray, n_classes: int):
    """Per-class means + a shared (pooled) covariance on the POI subspace."""

    n_poi = traces.shape[1]
    means = np.full((n_classes, n_poi), np.nan)
    pooled = np.zeros((n_poi, n_poi))
    total = 0
    for c in range(n_classes):
        rows = traces[labels == c]
        if len(rows) == 0:
            continue
        means[c] = rows.mean(axis=0)
        centered = rows - means[c]
        pooled += centered.T @ centered
        total += len(rows)
    pooled /= max(total - n_classes, 1)
    pooled += np.eye(n_poi) * (np.trace(pooled) / n_poi * 1e-3 + _EPS)  # regularize
    inv_cov = np.linalg.inv(pooled)
    return means, inv_cov


def _template_logliks(traces: np.ndarray, means: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Per-trace, per-class Gaussian log-likelihood (pooled cov ⇒ const drops)."""

    n, _ = traces.shape
    n_classes = means.shape[0]
    ll = np.full((n, n_classes), -np.inf)
    for c in range(n_classes):
        if np.isnan(means[c]).any():
            continue
        d = traces - means[c]
        ll[:, c] = -0.5 * np.einsum("ij,jk,ik->i", d, inv_cov, d)
    return ll


def run_template_attack(
    name: str, *, leakage: LeakageModel | None = None, path: str | None = None,
    policy: SplitPolicy | None = None, n_poi: int = 20, n_experiments: int = 100,
    order: int = 1, seed: int = 0, max_traces: int | None = None,
    **load_kwargs,
) -> ClassicalAttackResult:
    """Gaussian template attack (profiled): fit on the profiling set, attack the
    test set via likelihood, and rank keys with the shared log-likelihood metric.
    The ``n_poi`` points of interest are the samples with the highest SNR on the
    profiling traces."""

    leakage = leakage or LeakageModel(leakage="hw")
    with make_splits(name, policy=policy, path=path, **load_kwargs) as splits:
        x_train = _materialize(splits.train)
        y_train = leakage.dataset_labels(splits.train)
        x_test = _materialize(splits.test)
        hyp = leakage.dataset_hypotheses(splits.test)
        key = leakage.true_key_byte(splits.test)

    n_poi = min(n_poi, x_train.shape[1])
    poi = _select_poi(x_train, y_train, n_poi, leakage.n_classes)
    means, inv_cov = _fit_templates(x_train[:, poi], y_train, leakage.n_classes)
    ll = _template_logliks(x_test[:, poi], means, inv_cov)

    # log-likelihoods -> probabilities (stable softmax over classes)
    ll = ll - ll.max(axis=1, keepdims=True)
    probs = np.exp(ll)
    probs /= probs.sum(axis=1, keepdims=True) + _EPS

    ranks = evaluate_key_rank(
        probs, hyp, key, n_experiments=n_experiments, order=order,
        max_traces=max_traces, seed=seed,
    )
    return ClassicalAttackResult(ranks, len(x_train), len(x_test))


def run_classical(
    name: str, dataset: str, *, leakage: LeakageModel | None = None,
    path: str | None = None, seed: int = 0, n_experiments: int = 100,
    **kwargs,
) -> ClassicalAttackResult:
    """Dispatch to a classical attack by name (``"cpa"``/``"dpa"``/``"template"``)."""

    common = dict(leakage=leakage, path=path, seed=seed, n_experiments=n_experiments)
    if name == "cpa":
        return run_cpa(dataset, **common, **kwargs)
    if name == "dpa":
        return run_dpa(dataset, **common, **kwargs)
    if name == "template":
        return run_template_attack(dataset, **common, **kwargs)
    raise ValueError(f"unknown classical attack {name!r}; choose from {CLASSICAL_ATTACKS}")


__all__ = [
    "CLASSICAL_ATTACKS",
    "ClassicalAttackResult",
    "run_classical",
    "run_cpa",
    "run_dpa",
    "run_template_attack",
]
