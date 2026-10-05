# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""The same seed must give the same result (the benchmark's reproducibility promise)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - exercised without the extras.
    h5py = None

from mlsca_bench import register_dataset, unregister_dataset
from mlsca_bench.benchmark import LeakageModel


@unittest.skipIf(h5py is None, "requires the hdf5 and eval extras")
class SeedReproducibilityTests(unittest.TestCase):
    NAME = "repro-synthetic"

    @classmethod
    def setUpClass(cls) -> None:
        from mlsca_bench.benchmark import HAMMING_WEIGHT, SBOX

        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "traces.h5"
        rng = np.random.default_rng(0)
        key = rng.integers(0, 256, 16, dtype=np.uint8)
        pt = rng.integers(0, 256, (600, 16), dtype=np.uint8)
        traces = rng.normal(0, 0.5, (600, 60)).astype(np.float32)
        for b in range(2):
            traces[:, 10 + 20 * b] += HAMMING_WEIGHT[SBOX[pt[:, b] ^ key[b]]]
        with h5py.File(path, "w") as f:
            f["traces"] = traces
            f["metadata/plaintext"] = pt
            f["metadata/key"] = np.tile(key, (600, 1))
        register_dataset(cls.NAME, {
            "adapter": "hdf5",
            "adapter_config": {
                "splits": {"all": "/"}, "default_split": "all", "traces_dataset": "traces",
                "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]},
            },
            "metadata": {"format": "hdf5", "algorithm": "AES-128", "measurement": "simulated"},
        }, path=path)

    @classmethod
    def tearDownClass(cls) -> None:
        unregister_dataset(cls.NAME)
        cls._tmp.cleanup()

    @staticmethod
    def _same_weights(a, b) -> bool:
        return all(bool((x == y).all()) for x, y in zip(a.parameters(), b.parameters()))

    def _attack(self, model: str, seed: int):
        from mlsca_bench.models import run_attack

        return run_attack(self.NAME, model=model, leakage=LeakageModel(byte=0),
                          epochs=1, n_experiments=5, seed=seed)

    def test_same_seed_same_result(self) -> None:
        for model in ("mlp", "zaid_cnn"):
            with self.subTest(model=model):
                r1, r2 = self._attack(model, 0), self._attack(model, 0)
                self.assertTrue(self._same_weights(r1.model, r2.model))
                np.testing.assert_array_equal(r1.guessing_entropy, r2.guessing_entropy)

    def test_different_seed_different_model(self) -> None:
        r1, r2 = self._attack("mlp", 0), self._attack("mlp", 1)
        self.assertFalse(self._same_weights(r1.model, r2.model))


if __name__ == "__main__":
    unittest.main()
