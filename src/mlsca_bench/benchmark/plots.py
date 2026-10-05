# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Guessing entropy and success rate plots.

Plot the result of ``run_attack`` / ``run_classical`` (or a dict of several
results, to compare them) against the number of attack traces.

Requires the optional ``[plot]`` extra (matplotlib).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.errors import MissingDependencyError

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; callers can override before importing
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover
    raise MissingDependencyError("matplotlib", extra="plot", feature="GE and SR plots") from error


def _curves(results: Any) -> list[tuple[str, Any]]:
    """``{label: result}`` or one result -> [(label, RankResult)]."""

    def ranks_of(item: Any) -> Any:
        ranks = getattr(item, "ranks", item)
        if not hasattr(ranks, "guessing_entropy"):
            raise TypeError(f"Cannot plot {type(item).__name__}: pass a result of run_attack or run_classical.")
        return ranks

    if isinstance(results, Mapping):
        return [(str(label), ranks_of(item)) for label, item in results.items()]
    return [("attack", ranks_of(results))]


def _plot(results: Any, attr: str, ylabel: str, title: str, *, ax: Any, max_traces: int | None,
          logy: bool, save: str | Path | None) -> Any:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for label, ranks in _curves(results):
        x, y = np.asarray(ranks.n_traces), np.asarray(getattr(ranks, attr))
        if max_traces is not None:
            x, y = x[x <= max_traces], y[x <= max_traces]
        ax.plot(x, y, label=label)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    return ax


def _save(ax: Any, save: str | Path | None, own_figure: bool) -> Any:
    if save is not None:
        ax.figure.savefig(save, dpi=130, bbox_inches="tight")
        if own_figure:                 # only close figures this function created
            plt.close(ax.figure)
    return ax


def plot_guessing_entropy(results: Any, *, ax: Any = None, max_traces: int | None = None,
                          logy: bool = False, save: str | Path | None = None, title: str = "Guessing Entropy") -> Any:
    """Guessing entropy (mean rank of the correct key) against the number of attack traces.

    ``results`` is one result of ``run_attack`` / ``run_classical``, or a dict
    ``{"label": result, ...}`` to compare several in one plot. ``save`` writes
    the figure to a file.
    """

    own_figure = ax is None
    ax = _plot(results, "guessing_entropy", "Guessing entropy (mean rank)", title,
               ax=ax, max_traces=max_traces, logy=logy, save=None)
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    return _save(ax, save, own_figure)


def plot_success_rate(results: Any, *, ax: Any = None, max_traces: int | None = None,
                      save: str | Path | None = None, title: str = "Success Rate") -> Any:
    """Success rate against the number of attack traces (same inputs as plot_guessing_entropy)."""

    own_figure = ax is None
    ax = _plot(results, "success_rate", "Success rate", title,
               ax=ax, max_traces=max_traces, logy=False, save=None)
    ax.set_ylim(-0.02, 1.02)
    return _save(ax, save, own_figure)


__all__ = ["plot_guessing_entropy", "plot_success_rate"]
