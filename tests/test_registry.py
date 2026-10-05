# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlsca_bench.datasets.registry import (
    RegistryValidationError,
    UnknownDatasetError,
    get_dataset,
    has_dataset,
    list_datasets,
    load_registry,
)


class PackagedRegistryTests(unittest.TestCase):
    def test_get_dataset_by_canonical_name(self) -> None:
        dataset = get_dataset("ascon-cw-unprotected")
        self.assertEqual(dataset.display_name, "Ascon ChipWhisperer unprotected")
        self.assertEqual(dataset.availability, "automatic")
        self.assertEqual(len(dataset.files), 1)

    def test_get_dataset_by_normalized_alias(self) -> None:
        dataset = get_dataset("ASCON software_unprotected")
        self.assertEqual(dataset.name, "ascon-cw-unprotected")

    def test_list_datasets_is_sorted(self) -> None:
        names = [dataset.name for dataset in list_datasets()]
        self.assertEqual(names, sorted(names))
        self.assertGreaterEqual(len(names), 50)
        self.assertTrue(
            {
                "aes-hd-git",
                "aes-hd-zaid",
                "ascadf",
                "ascon-cw-unprotected",
                "eshard-aes-masked-non-shuffled",
                "eshard-aes-masked-shuffled",
                "ge-wars",
                "kyber-reference-ppm",
                "dpacontest-v1-secmatv1-asic",
                "dpacontest-v2",
                "dpacontest-v3-secmatv3-des",
                "dpacontest-v4-1-rsm",
                "dpacontest-v4-2",
                "chameleon-base",
                "ches-ctf-2018",
                "dfs-desynch",
                "dtds-dilithium5",
                "scaaml-ecc-gpam-cm3",
                "x-deepsca",
            }.issubset(names)
        )

    def test_aes_hd_csv_alias_resolves_to_canonical_name(self) -> None:
        dataset = get_dataset("AES HD CSV")
        self.assertEqual(dataset.name, "aes-hd-git")

    def test_aes_hd_datasets_are_automatic_and_fully_hashed(self) -> None:
        for name, expected_file_count in (("aes-hd-git", 6), ("aes-hd-zaid", 1)):
            with self.subTest(name=name):
                dataset = get_dataset(name)
                self.assertEqual(dataset.availability, "automatic")
                self.assertEqual(len(dataset.files), expected_file_count)
                self.assertTrue(
                    all(
                        dataset_file.known_hash
                        and dataset_file.known_hash.startswith("sha256:")
                        for dataset_file in dataset.files
                    )
                )

    def test_eshard_datasets_are_automatic_and_fully_hashed(self) -> None:
        for name in (
            "eshard-aes-masked-non-shuffled",
            "eshard-aes-masked-shuffled",
        ):
            with self.subTest(name=name):
                dataset = get_dataset(name)
                self.assertEqual(dataset.availability, "automatic")
                self.assertEqual(len(dataset.files), 1)
                self.assertTrue(dataset.files[0].known_hash.startswith("sha256:"))

    def test_eshard_aliases_resolve(self) -> None:
        self.assertEqual(
            get_dataset("eShard non shuffled").name,
            "eshard-aes-masked-non-shuffled",
        )
        self.assertEqual(
            get_dataset("eShard shuffled").name,
            "eshard-aes-masked-shuffled",
        )

    def test_ge_wars_dataset_is_automatic_and_hashed(self) -> None:
        dataset = get_dataset("ge_wars")

        self.assertEqual(dataset.name, "ge-wars")
        self.assertEqual(dataset.display_name, "GE Wars 2025")
        self.assertEqual(dataset.availability, "automatic")
        self.assertEqual(len(dataset.files), 1)
        self.assertEqual(dataset.files[0].filename, "CHES_Challenge.h5")
        self.assertEqual(
            dataset.files[0].known_hash,
            "sha256:132ae2e9a8213c983bf3b63449e9572d5d71d3b376a75b236415d4a728b9379f",
        )
        self.assertIsNone(dataset.files[0].archive)

    def test_ge_wars_aliases_resolve(self) -> None:
        for alias in ("GEWars", "GE Wars 2025", "CHES Challenge 2025"):
            with self.subTest(alias=alias):
                self.assertEqual(get_dataset(alias).name, "ge-wars")

    def test_kyber_dataset_is_automatic_and_fully_hashed(self) -> None:
        dataset = get_dataset("kyber")

        self.assertEqual(dataset.name, "kyber-reference-ppm")
        self.assertEqual(dataset.availability, "automatic")
        self.assertEqual(dataset.license, "CC-BY-4.0")
        self.assertEqual(
            [dataset_file.filename for dataset_file in dataset.files],
            ["Reference-PPM.zip", "load-reference-ppm.zip"],
        )
        self.assertTrue(
            all(
                dataset_file.archive == "zip"
                and dataset_file.known_hash
                and dataset_file.known_hash.startswith("md5:")
                for dataset_file in dataset.files
            )
        )

    def test_kyber_aliases_resolve(self) -> None:
        for alias in ("Kyber PPM", "CRYSTALS_Kyber", "reference ppm"):
            with self.subTest(alias=alias):
                self.assertEqual(get_dataset(alias).name, "kyber-reference-ppm")

    def test_dpa_contest_campaign_availability(self) -> None:
        available_campaigns = (
            "dpacontest-v1-secmatv1-asic",
            "dpacontest-v2",
            "dpacontest-v3-secmatv3-des",
            "dpacontest-v3-secmatv3-des-20071219",
            "dpacontest-v4-1-rsm",
            "dpacontest-v4-2",
        )
        for name in available_campaigns:
            with self.subTest(name=name):
                self.assertEqual(get_dataset(name).availability, "automatic")

    def test_dpa_contest_version_aliases_resolve(self) -> None:
        self.assertEqual(get_dataset("DPA v1").name, "dpacontest-v1-secmatv1-asic")
        self.assertEqual(get_dataset("DPA v2").name, "dpacontest-v2")
        self.assertEqual(get_dataset("DPA v4").name, "dpacontest-v4-1-rsm")
        self.assertEqual(get_dataset("DPA v4.2").name, "dpacontest-v4-2")

    def test_has_dataset(self) -> None:
        self.assertTrue(has_dataset("ascad fixed"))
        self.assertFalse(has_dataset("not-a-real-dataset"))

    def test_unknown_dataset_has_suggestion(self) -> None:
        with self.assertRaisesRegex(UnknownDatasetError, r"Did you mean:.*\bascadf\b"):
            get_dataset("ascad-fixd")

    def test_ascad_has_loading_adapter(self) -> None:
        self.assertEqual(get_dataset("ascadf").adapter, "ascad")

    def test_adapter_is_optional(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "example": {
                        "display_name": "Example",
                        "aliases": [],
                        "availability": "manual",
                        "files": [],
                        "homepage": None,
                        "paper": None,
                        "license": None,
                        "notes": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        registry, _ = load_registry(path)
        self.assertIsNone(registry["example"].adapter)

    def test_every_dataset_has_an_adapter(self) -> None:
        # After wiring the yellow gaps, no registered dataset should lack one.
        missing = [s.name for s in list_datasets(builtin_only=True) if s.adapter is None]
        self.assertEqual(missing, [])

    def test_newly_wired_gap_datasets(self) -> None:
        # Group names verified against the real (RAR-extracted) HDF5: the
        # masked schemes are "MS1"/"MS2" (not the long names), nested under a
        # per-device top-level group the adapter auto-detects.
        checks = {
            "aes-ptv2-pinata-ms1": ("hdf5", "MS1/Profiling"),
            "aes-ptv2-pinata-ms2": ("hdf5", "MS2/Profiling"),
            "aes-ptv2-stm32f4-ms1": ("hdf5", "MS1/Profiling"),
            "aes-ptv2-stm32f4-ms2": ("hdf5", "MS2/Profiling"),
        }
        for name, (adapter, prof_group) in checks.items():
            with self.subTest(name=name):
                spec = get_dataset(name)
                self.assertEqual(spec.adapter, adapter)
                self.assertEqual(
                    spec.adapter_config["splits"]["profiling"], prof_group
                )
                self.assertIn("masking", spec.metadata.countermeasures)

        self.assertEqual(get_dataset("ecc-cswap-arith").adapter, "flat-hdf5")
        self.assertEqual(
            get_dataset("ecc-cswap-arith").adapter_config["preferred_filename"],
            "cswap_arith.h5",
        )
        prot = get_dataset("ascon-cw-protected")
        self.assertEqual(prot.adapter, "ascon-hdf5")
        self.assertTrue(prot.files[0].known_hash.startswith("md5:"))
        self.assertEqual(get_dataset("ascon-cw-unprotected-raw").adapter, "trs")
        # ASCAD raw (fixed) rides the same ASCAD_data.zip as ascadf -> automatic.
        raw = get_dataset("ascad-raw-fixed")
        self.assertEqual(raw.availability, "automatic")
        self.assertEqual(
            raw.files[0].known_hash, get_dataset("ascadf").files[0].known_hash
        )
        self.assertEqual(
            raw.adapter_config["preferred_filename"], "ATMega8515_raw_traces.h5"
        )
        # ascadv2r: sha1-pinned from the ANSSI sha1.txt -> automatic (8 files).
        v2r = get_dataset("ascadv2r")
        self.assertEqual(v2r.availability, "automatic")
        self.assertEqual(len(v2r.files), 8)
        self.assertTrue(all(f.known_hash.startswith("sha1:") for f in v2r.files))

    def test_packaged_metadata_is_populated(self) -> None:
        # Every registered dataset should carry the diversity-axis metadata.
        from mlsca_bench.datasets.registry import (
            _COUNTERMEASURE_VALUES,
            _KEY_VALUES,
            _PLATFORM_VALUES,
        )

        for spec in list_datasets(builtin_only=True):
            with self.subTest(name=spec.name):
                m = spec.metadata
                self.assertIsNotNone(m, f"{spec.name} has no metadata")
                self.assertIn(m.platform, _PLATFORM_VALUES)
                self.assertTrue(m.countermeasures, f"{spec.name} has no countermeasures")
                for cm in m.countermeasures:
                    self.assertIn(cm, _COUNTERMEASURE_VALUES)
                if m.key is not None:
                    self.assertIn(m.key, _KEY_VALUES)

    def test_contrasting_datasets_have_loading_adapters(self) -> None:
        self.assertEqual(get_dataset("aes-hd-git").adapter, "aes-hd-csv")
        self.assertEqual(
            get_dataset("ascon-cw-unprotected").adapter,
            "ascon-hdf5",
        )
        self.assertEqual(get_dataset("aes-hd-zaid").adapter, "aes-hd-zaid")
        self.assertEqual(get_dataset("ge-wars").adapter, "ascad")


class RegistryValidationTests(unittest.TestCase):
    def _write_registry(self, data: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_automatic_dataset_requires_a_file(self) -> None:
        path = self._write_registry(
            {
                "example": {
                    "display_name": "Example",
                    "aliases": [],
                    "availability": "automatic",
                    "files": [],
                    "homepage": None,
                    "paper": None,
                    "license": None,
                    "notes": None,
                }
            }
        )
        with self.assertRaisesRegex(RegistryValidationError, "at least one file"):
            load_registry(path)

    def test_duplicate_alias_is_rejected(self) -> None:
        entry = {
            "display_name": "Example",
            "aliases": ["shared"],
            "availability": "manual",
            "files": [],
            "homepage": None,
            "paper": None,
            "license": None,
            "notes": None,
        }
        second = dict(entry)
        second["display_name"] = "Other"
        path = self._write_registry({"example": entry, "other": second})
        with self.assertRaisesRegex(RegistryValidationError, "conflicts"):
            load_registry(path)

    def _metadata_entry(self, metadata: object) -> Path:
        return self._write_registry(
            {
                "example": {
                    "display_name": "Example",
                    "aliases": [],
                    "availability": "manual",
                    "files": [],
                    "homepage": None,
                    "paper": None,
                    "license": None,
                    "notes": None,
                    "metadata": metadata,
                }
            }
        )

    def test_metadata_diversity_fields_parse(self) -> None:
        path = self._metadata_entry(
            {
                "format": "hdf5",
                "algorithm": "AES-128",
                "measurement": "power",
                "platform": "avr",
                "device": "ATMega8515",
                "countermeasures": ["masking", "desynchronization"],
                "key": "variable",
                "n_samples": 1400,
                "year": 2018,
            }
        )
        registry, _ = load_registry(path)
        m = registry["example"].metadata
        assert m.platform == "avr"
        assert m.device == "ATMega8515"
        assert m.countermeasures == ("masking", "desynchronization")
        assert m.key == "variable"
        assert m.n_samples == 1400
        assert m.year == 2018

    def test_metadata_optional_fields_default(self) -> None:
        path = self._metadata_entry(
            {"format": "hdf5", "algorithm": "AES", "measurement": "power"}
        )
        registry, _ = load_registry(path)
        m = registry["example"].metadata
        assert m.platform is None
        assert m.countermeasures == ()
        assert m.n_samples is None

    def test_metadata_rejects_bad_platform(self) -> None:
        path = self._metadata_entry(
            {"format": "hdf5", "algorithm": "AES", "measurement": "power", "platform": "quantum"}
        )
        with self.assertRaisesRegex(RegistryValidationError, "expected one of"):
            load_registry(path)

    def test_metadata_rejects_bad_countermeasure(self) -> None:
        path = self._metadata_entry(
            {
                "format": "hdf5",
                "algorithm": "AES",
                "measurement": "power",
                "countermeasures": ["wizardry"],
            }
        )
        with self.assertRaisesRegex(RegistryValidationError, "expected one of"):
            load_registry(path)

    def test_metadata_rejects_negative_samples(self) -> None:
        path = self._metadata_entry(
            {"format": "hdf5", "algorithm": "AES", "measurement": "power", "n_samples": -5}
        )
        with self.assertRaisesRegex(RegistryValidationError, "non-negative"):
            load_registry(path)

    def test_bad_hash_is_rejected(self) -> None:
        path = self._write_registry(
            {
                "example": {
                    "display_name": "Example",
                    "aliases": [],
                    "availability": "automatic",
                    "files": [
                        {
                            "url": "https://example.test/data.h5",
                            "filename": "data.h5",
                            "known_hash": "sha256:not-a-hash",
                            "archive": None,
                        }
                    ],
                    "homepage": None,
                    "paper": None,
                    "license": None,
                    "notes": None,
                }
            }
        )
        with self.assertRaisesRegex(RegistryValidationError, "hex digest"):
            load_registry(path)


if __name__ == "__main__":
    unittest.main()


class VerifiedMetadataTests(unittest.TestCase):
    """Values checked against each dataset's paper or README (Sep 2026)."""

    def test_measurement_corrections(self) -> None:
        # DPA Contest v4 docs: Langer RF-U 5-2 EM near-field probe.
        for name in ("dpacontest-v4-1-rsm", "dpacontest-v4-2"):
            self.assertEqual(get_dataset(name).metadata.measurement, "electromagnetic")
        # ASCAD v1 is power despite the 2018 paper (ANSSI-FR/ASCAD issue 13).
        self.assertEqual(get_dataset("ascadf").metadata.measurement, "power")

    def test_countermeasure_corrections(self) -> None:
        expected = {
            "ascadv-raw": ("masking",),
            "ascadv2": ("affine-masking", "shuffling"),
            "dpacontest-v4-2": ("masking", "shuffling"),
            "present-2021-ti-misaligned": ("threshold-implementation", "desynchronization"),
            "galactics-attack-data": ("constant-time",),
            "wolfssl-ed25519": ("constant-time",),
            "reassure-c25519-arithm": ("constant-time", "masking"),
            "ecc-cswap-arith": ("constant-time", "masking"),
        }
        for name, cms in expected.items():
            self.assertEqual(get_dataset(name).metadata.countermeasures, cms, name)

    def test_re_encryption_targets_aes(self) -> None:
        # ePrint 2021/849: the leaking primitive is AES used as the FO re-encryption PRF.
        self.assertEqual(get_dataset("re-encryption").metadata.algorithm, "AES-128")

    def test_dpa_contest_v1_des_tables(self) -> None:
        # The SecMatV3 DES sets are DPA Contest v1 trace tables; old names stay as aliases.
        self.assertEqual(get_dataset("dpacontest-v1-secmatv3-des").name, "dpacontest-v3-secmatv3-des")
        self.assertEqual(get_dataset("dpa-v3").name, "dpacontest-v3-secmatv3-des")
        self.assertIn("DPA Contest v1", get_dataset("dpacontest-v3-secmatv3-des").display_name)

    def test_poster_totals(self) -> None:
        specs = list_datasets(builtin_only=True)
        self.assertEqual(len(specs), 64)
        self.assertEqual(len({s.metadata.algorithm for s in specs}), 12)
        self.assertEqual(sum(s.metadata.measurement == "electromagnetic" for s in specs), 13)
        self.assertEqual(sum(s.metadata.countermeasures == ("none",) for s in specs), 19)
