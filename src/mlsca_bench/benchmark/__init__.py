# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark layer: leakage models, reproducible splits, classical attacks and metrics.

The metrics, splits and classical attacks are pure NumPy. The deep-learning
baselines live in :mod:`mlsca_bench.models` behind the optional ``eval`` extra,
and the plots in :mod:`mlsca_bench.benchmark.plots` behind the ``plot`` extra.
"""

from .classical import (
    CLASSICAL_ATTACKS,
    ClassicalAttackResult,
    run_classical,
    run_cpa,
    run_dpa,
    run_template_attack,
)
from .leakage import (
    HAMMING_WEIGHT,
    SBOX,
    SBOX_INV,
    LeakageModel,
    aes128_round_keys,
    aes_round1_output,
)
from .metrics import (
    RankResult,
    evaluate_key_rank,
    guessing_entropy,
    success_rate,
)
from .splits import (
    DatasetSplits,
    SplitPolicy,
    Subset,
    describe_splits,
    make_splits,
    resolve_policy,
    verify_splits,
)

__all__ = [
    "CLASSICAL_ATTACKS",
    "ClassicalAttackResult",
    "DatasetSplits",
    "HAMMING_WEIGHT",
    "LeakageModel",
    "RankResult",
    "SBOX",
    "SBOX_INV",
    "SplitPolicy",
    "Subset",
    "aes128_round_keys",
    "aes_round1_output",
    "describe_splits",
    "evaluate_key_rank",
    "guessing_entropy",
    "make_splits",
    "resolve_policy",
    "run_classical",
    "run_cpa",
    "run_dpa",
    "run_template_attack",
    "success_rate",
    "verify_splits",
]
