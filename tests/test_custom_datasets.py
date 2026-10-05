# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mlsca_bench
from mlsca_bench import (
    get_dataset,
    list_datasets,
    load_dataset,
    load_registry_file,
    register_dataset,
    unregister_dataset,
    validate_dataset,
)
from mlsca_bench.datasets.registry import (
    DatasetRegistryError,
    RegistryValidationError,
    UnknownDatasetError,
    has_dataset,
    is_user_dataset,
)

try:
    import h5py
except ImportError:  # pragma: no cover - exercised without the extra.
    h5py = None

KEY = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
                0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], dtype=np.uint8)

ENTRY = {
    "adapter": "hdf5",
    "adapter_config": {
        "splits": {"all": "/"},
        "default_split": "all",
        "traces_dataset": "traces",
        "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]},
    },
    "metadata": {
        "format": "hdf5",
        "algorithm": "AES-128",
        "measurement": "simulated",
        "platform": "other",
        "countermeasures": ["none"],
        "key": "fixed",
    },
}


def _write_leaky_hdf5(path: Path, n: int = 2000, n_samples: int = 20) -> None:
    """Synthetic AES traces: sample 5 leaks HW(Sbox[pt[0] ^ k[0]]) plus noise."""

    from mlsca_bench.benchmark import HAMMING_WEIGHT, SBOX

    rng = np.random.default_rng(0)
    plaintext = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    traces = rng.normal(0.0, 0.5, size=(n, n_samples)).astype(np.float32)
    traces[:, 5] += HAMMING_WEIGHT[SBOX[plaintext[:, 0] ^ KEY[0]]]
    with h5py.File(path, "w") as f:
        f["traces"] = traces
        f["metadata/plaintext"] = plaintext
        f["metadata/key"] = np.tile(KEY, (n, 1))


class _UserRegistryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        for spec in list_datasets():
            if is_user_dataset(spec.name):
                unregister_dataset(spec.name)
        self._tmp.cleanup()


class RegisterDatasetTests(_UserRegistryCase):
    def test_minimal_entry_gets_defaults(self) -> None:
        spec = register_dataset("my-lab-aes", ENTRY, path=self.tmp / "x.h5")
        self.assertEqual(spec.name, "my-lab-aes")
        self.assertEqual(spec.display_name, "my-lab-aes")
        self.assertEqual(spec.availability, "manual")
        self.assertEqual(spec.files, ())
        self.assertEqual(spec.metadata.algorithm, "AES-128")

    def test_name_is_normalized_and_aliases_resolve(self) -> None:
        register_dataset("My Lab AES", {**ENTRY, "aliases": ["lab aes v1"]})
        self.assertEqual(get_dataset("my_lab_aes").name, "my-lab-aes")
        self.assertEqual(get_dataset("Lab AES v1").name, "my-lab-aes")
        self.assertIn("my-lab-aes", [s.name for s in list_datasets()])

    def test_unregister_removes_name_and_aliases(self) -> None:
        register_dataset("my-lab-aes", {**ENTRY, "aliases": ["lab-aes"]})
        unregister_dataset("lab-aes")
        self.assertFalse(has_dataset("my-lab-aes"))
        self.assertFalse(has_dataset("lab-aes"))

    def test_builtin_datasets_are_protected(self) -> None:
        with self.assertRaisesRegex(RegistryValidationError, "built-in dataset 'ascadf'"):
            register_dataset("ascadf", ENTRY)
        with self.assertRaisesRegex(RegistryValidationError, "built-in"):
            register_dataset("mine", {**ENTRY, "aliases": ["ascad-fixed"]})
        with self.assertRaises(DatasetRegistryError):
            unregister_dataset("ascadf")
        self.assertEqual(get_dataset("ascadf").display_name, "ASCAD fixed-key")

    def test_reregistering_needs_overwrite(self) -> None:
        register_dataset("my-lab-aes", ENTRY)
        with self.assertRaisesRegex(RegistryValidationError, "overwrite=True"):
            register_dataset("my-lab-aes", ENTRY)
        spec = register_dataset("my-lab-aes", {**ENTRY, "display_name": "v2"}, overwrite=True)
        self.assertEqual(get_dataset("my-lab-aes").display_name, "v2")
        self.assertIs(get_dataset("my-lab-aes"), spec)

    def test_invalid_metadata_is_rejected_with_location(self) -> None:
        bad = {**ENTRY, "metadata": {**ENTRY["metadata"], "platform": "raspberry"}}
        with self.assertRaisesRegex(RegistryValidationError, r"register_dataset\('bad'\)\.metadata\.platform"):
            register_dataset("bad", bad)
        self.assertFalse(has_dataset("bad"))

    def test_adapter_is_required(self) -> None:
        with self.assertRaisesRegex(RegistryValidationError, "needs an adapter"):
            register_dataset("no-adapter", {"metadata": ENTRY["metadata"]})

    def test_http_downloads_still_need_a_hash(self) -> None:
        entry = {**ENTRY, "files": [{"url": "https://example.org/traces.h5"}]}
        with self.assertRaisesRegex(RegistryValidationError, "known hash"):
            register_dataset("remote", entry)

    def test_unknown_dataset_after_unregister(self) -> None:
        register_dataset("temp-ds", ENTRY)
        unregister_dataset("temp-ds")
        with self.assertRaises(UnknownDatasetError):
            get_dataset("temp-ds")


@unittest.skipIf(h5py is None, "requires the hdf5 extra")
class LoadAndAttackUserDatasetTests(_UserRegistryCase):
    def setUp(self) -> None:
        super().setUp()
        self.h5 = self.tmp / "my_traces.h5"
        _write_leaky_hdf5(self.h5)

    def test_load_registered_local_file(self) -> None:
        register_dataset("my-lab-aes", ENTRY, path=self.h5)
        with load_dataset("my-lab-aes") as ds:
            validate_dataset(ds)
            self.assertEqual(ds.shape, (2000, 20))
            np.testing.assert_array_equal(ds[0].key, KEY)

    def test_path_can_live_in_the_entry(self) -> None:
        register_dataset("my-lab-aes", {**ENTRY, "path": str(self.h5)})
        with load_dataset("my-lab-aes") as ds:
            self.assertEqual(len(ds), 2000)

    def test_benchmark_runs_on_your_own_traces(self) -> None:
        from mlsca_bench.benchmark import LeakageModel, run_cpa

        register_dataset("my-lab-aes", ENTRY, path=self.h5)
        result = run_cpa("my-lab-aes", leakage=LeakageModel(byte=0), n_experiments=5)
        self.assertEqual(result.ranks.guessing_entropy[-1], 0.0)


@unittest.skipIf(h5py is None, "requires the hdf5 extra")
class RegistryFileTests(_UserRegistryCase):
    def test_relative_paths_resolve_against_the_file(self) -> None:
        _write_leaky_hdf5(self.tmp / "data.h5", n=50)
        registry = self.tmp / "my_datasets.json"
        registry.write_text(json.dumps({"lab-a": {**ENTRY, "path": "data.h5"}}))
        self.assertEqual(load_registry_file(registry), ("lab-a",))
        with load_dataset("lab-a") as ds:
            self.assertEqual(len(ds), 50)

    def test_one_bad_entry_registers_nothing(self) -> None:
        bad = {**ENTRY, "metadata": {**ENTRY["metadata"], "measurement": "vibes"}}
        registry = self.tmp / "my_datasets.json"
        registry.write_text(json.dumps({"good-one": ENTRY, "bad-one": bad}))
        with self.assertRaisesRegex(RegistryValidationError, "bad-one.metadata.measurement"):
            load_registry_file(registry)
        self.assertFalse(has_dataset("good-one"))

    def test_environment_variable_loads_at_import(self) -> None:
        registry = self.tmp / "env_datasets.json"
        registry.write_text(json.dumps({"env-lab": ENTRY}))
        env = {**os.environ, "MLSCA_BENCH_REGISTRY": str(registry)}
        out = subprocess.run(
            [sys.executable, "-c",
             "import mlsca_bench as m; print(m.get_dataset('env-lab').metadata.algorithm)"],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(out.stdout.strip(), "AES-128")

    def test_broken_environment_registry_fails_loudly(self) -> None:
        env = {**os.environ, "MLSCA_BENCH_REGISTRY": str(self.tmp / "missing.json")}
        out = subprocess.run(
            [sys.executable, "-c", "import mlsca_bench"],
            env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("MLSCA_BENCH_REGISTRY", out.stderr)


class PublicApiTests(unittest.TestCase):
    def test_functions_are_exported_at_top_level(self) -> None:
        for name in ("register_dataset", "unregister_dataset", "load_registry_file"):
            self.assertIn(name, mlsca_bench.__all__)
            self.assertTrue(callable(getattr(mlsca_bench, name)))


if __name__ == "__main__":
    unittest.main()


class UnknownDatasetMessageTests(unittest.TestCase):
    def test_unknown_name_explains_session_scope(self) -> None:
        with self.assertRaises(UnknownDatasetError) as ctx:
            get_dataset("my-dummy-traces")
        message = str(ctx.exception)
        for fragment in ("register_dataset()", "Python session", "save=True", "mlsca-bench add"):
            self.assertIn(fragment, message)


@unittest.skipIf(h5py is None, "requires the hdf5 extra")
class FieldTypoTests(_UserRegistryCase):
    def test_typo_in_field_path_is_reported_with_the_real_paths(self) -> None:
        h5 = self.tmp / "t.h5"
        _write_leaky_hdf5(h5, n=20)
        bad = {**ENTRY, "adapter_config": {**ENTRY["adapter_config"],
               "field_aliases": {"plaintexts": ["metadata/plain"], "keys": ["metadata/key"]}}}
        register_dataset("typo-ds", bad, path=h5)
        with self.assertRaises(ValueError) as ctx:
            load_dataset("typo-ds")
        message = str(ctx.exception)
        self.assertIn("plaintexts -> ['metadata/plain']", message)
        self.assertIn("metadata/plaintext", message)          # the path that does exist


@unittest.skipIf(h5py is None, "requires the hdf5 extra")
class SymlinkedRegistryTests(_UserRegistryCase):
    def test_parent_path_through_a_symlinked_folder(self) -> None:
        real = self.tmp / "real" / "deep"
        real.mkdir(parents=True)
        _write_leaky_hdf5(self.tmp / "data.h5", n=10)
        (self.tmp / "link").symlink_to(real, target_is_directory=True)
        registry = self.tmp / "link" / "reg.json"
        registry.write_text(json.dumps({"sym-ds": {**ENTRY, "path": "../data.h5"}}))
        load_registry_file(registry)                      # '../data.h5' as written: next to 'link'
        with load_dataset("sym-ds") as ds:
            self.assertEqual(len(ds), 10)


class SingleSplitDefaultTests(_UserRegistryCase):
    def test_the_only_split_is_used_without_default_split(self) -> None:
        if h5py is None:
            self.skipTest("needs h5py")
        path = self.tmp / "one.h5"
        _write_leaky_hdf5(path, n=50)
        entry = {**ENTRY, "adapter_config": {k: v for k, v in ENTRY["adapter_config"].items() if k != "default_split"}}
        register_dataset("one-split", entry, path=path)
        with load_dataset("one-split") as ds:
            self.assertEqual(ds.split, "all")
            self.assertEqual(len(ds), 50)
