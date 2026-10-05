# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np

from mlsca_bench.datasets import (
    ArraySideChannelDataset,
    SideChannelDataset,
    TraceSample,
    broadcast_field,
    validate_dataset,
)


class DatasetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traces = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.plaintexts = np.arange(6, dtype=np.uint8).reshape(3, 2)
        self.key = np.array([10, 11], dtype=np.uint8)
        self.keys = broadcast_field(self.key, len(self.traces))
        self.labels = np.array([1, 2, 3], dtype=np.uint8)
        self.dataset = ArraySideChannelDataset(
            name="synthetic",
            split="profiling",
            traces=self.traces,
            plaintexts=self.plaintexts,
            keys=self.keys,
            labels=self.labels,
            metadata={"algorithm": "test-cipher"},
            sample_metadata=[{"source_index": i} for i in range(3)],
        )

    def test_bulk_fields_have_a_shared_trace_axis(self) -> None:
        validate_dataset(self.dataset)
        self.assertEqual(self.dataset.shape, (3, 4))
        self.assertEqual(self.dataset.dtype, np.dtype(np.float32))
        assert self.dataset.plaintexts is not None
        assert self.dataset.keys is not None
        self.assertEqual(self.dataset.plaintexts.shape[0], len(self.dataset))
        self.assertEqual(self.dataset.keys.shape[0], len(self.dataset))
        self.assertEqual(
            self.dataset.available_fields,
            ("traces", "plaintexts", "keys", "labels"),
        )

    def test_missing_fields_are_none(self) -> None:
        self.assertIsNone(self.dataset.ciphertexts)
        self.assertIsNone(self.dataset.masks)
        sample = self.dataset[0]
        self.assertIsNone(sample.ciphertext)
        self.assertIsNone(sample.mask)

    def test_integer_access_returns_an_aligned_sample(self) -> None:
        sample = self.dataset[1]
        self.assertIsInstance(sample, TraceSample)
        np.testing.assert_array_equal(sample.trace, self.traces[1])
        np.testing.assert_array_equal(sample.plaintext, self.plaintexts[1])
        np.testing.assert_array_equal(sample.key, self.key)
        self.assertEqual(sample.label.item(), self.labels[1])
        self.assertEqual(sample.metadata, {"source_index": 1})

    def test_dataset_and_sample_metadata_are_separate_and_read_only(self) -> None:
        self.assertEqual(self.dataset.metadata, {"algorithm": "test-cipher"})
        with self.assertRaises(TypeError):
            self.dataset.metadata["algorithm"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.dataset[0].metadata["source_index"] = 10  # type: ignore[index]

    def test_fixed_field_is_a_zero_copy_broadcast_view(self) -> None:
        self.assertFalse(self.keys.flags.writeable)
        self.assertTrue(np.shares_memory(self.keys, self.key))
        np.testing.assert_array_equal(self.keys[2], self.key)

    def test_rejects_misaligned_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "one entry per trace"):
            ArraySideChannelDataset(
                name="invalid",
                traces=self.traces,
                plaintexts=np.zeros((2, 2), dtype=np.uint8),
            )

    def test_validator_rejects_closed_dataset(self) -> None:
        self.dataset.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            validate_dataset(self.dataset)

    def test_close_is_idempotent_and_prevents_reads(self) -> None:
        self.dataset.close()
        self.dataset.close()
        self.assertTrue(self.dataset.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = self.dataset.traces
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = self.dataset[0]

    def test_context_manager_closes_dataset(self) -> None:
        with self.dataset as opened:
            self.assertIs(opened, self.dataset)
            self.assertFalse(opened.closed)
        self.assertTrue(self.dataset.closed)

    def test_base_contract_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            SideChannelDataset()  # type: ignore[abstract]

    def test_contract_is_available_from_top_level_package(self) -> None:
        import mlsca_bench

        self.assertIs(mlsca_bench.SideChannelDataset, SideChannelDataset)
        self.assertIs(
            mlsca_bench.ArraySideChannelDataset,
            ArraySideChannelDataset,
        )


if __name__ == "__main__":
    unittest.main()
