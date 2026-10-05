# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from mlsca_bench.datasets.registry import RegistryValidationError, load_registry


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "display_name": "Example",
        "aliases": [],
        "availability": "manual",
        "files": [],
        "homepage": None,
        "paper": None,
        "license": None,
        "notes": None,
    }
    entry.update(overrides)
    return entry


class AdapterConfigTests(unittest.TestCase):
    def _write(self, data: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_adapter_config_is_parsed_and_frozen(self) -> None:
        path = self._write(
            {
                "example": _entry(
                    adapter="hdf5",
                    adapter_config={
                        "splits": {"profiling": "train"},
                        "field_aliases": {"keys": ["k", "key"]},
                    },
                )
            }
        )
        registry, _ = load_registry(path)
        config = registry["example"].adapter_config
        assert config is not None
        # Nested containers are read-only views.
        self.assertIsInstance(config, Mapping)
        self.assertEqual(config["splits"]["profiling"], "train")
        self.assertEqual(config["field_aliases"]["keys"], ("k", "key"))
        with self.assertRaises(TypeError):
            config["splits"]["profiling"] = "other"  # type: ignore[index]

    def test_adapter_config_requires_adapter(self) -> None:
        path = self._write({"example": _entry(adapter_config={"splits": {}})})
        with self.assertRaisesRegex(RegistryValidationError, "requires an adapter"):
            load_registry(path)

    def test_adapter_config_must_be_object(self) -> None:
        path = self._write(
            {"example": _entry(adapter="hdf5", adapter_config=["not", "an", "object"])}
        )
        with self.assertRaisesRegex(RegistryValidationError, "expected a JSON object"):
            load_registry(path)

    def test_adapter_config_is_optional(self) -> None:
        path = self._write({"example": _entry(adapter="hdf5")})
        registry, _ = load_registry(path)
        self.assertIsNone(registry["example"].adapter_config)


if __name__ == "__main__":
    unittest.main()
