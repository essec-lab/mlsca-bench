# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mlsca_bench.datasets.errors import DatasetDownloadError
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from requests.exceptions import ReadTimeout

from mlsca_bench.datasets.download import cache_root, download_dataset
from mlsca_bench.datasets.errors import (
    CustomDownloaderRequired,
    DatasetUnavailable,
    ManualDownloadRequired,
)
from mlsca_bench.datasets.registry import DatasetFile, DatasetSpec


class FakePooch:
    def __init__(self) -> None:
        self.retrieve_calls: list[dict[str, object]] = []
        self.downloader_calls: list[dict[str, object]] = []
        self.retrieve_effects: list[BaseException] = []

    def os_cache(self, name: str) -> str:
        return f"/cache/{name}"

    def Unzip(self) -> str:
        return "unzip-processor"

    def Untar(self) -> str:
        return "untar-processor"

    def Decompress(self) -> str:
        return "decompress-processor"

    def HTTPDownloader(self, **kwargs: object) -> str:
        self.downloader_calls.append(kwargs)
        return "http-downloader"

    def retrieve(self, **kwargs: object) -> str | list[str]:
        self.retrieve_calls.append(kwargs)
        if self.retrieve_effects:
            raise self.retrieve_effects.pop(0)
        path = Path(str(kwargs["path"]))
        filename = str(kwargs["fname"])
        if kwargs["processor"] is None:
            return str(path / filename)
        return [str(path / "extracted" / "traces.h5")]


def make_dataset(
    *,
    availability: str,
    archive: str | None = None,
) -> DatasetSpec:
    files = ()
    if availability in {"automatic", "custom"}:
        files = (
            DatasetFile(
                url="https://example.test/traces.zip",
                filename="traces.zip",
                known_hash="sha256:" + "a" * 64,
                archive=archive,
            ),
        )
    return DatasetSpec(
        name="example",
        display_name="Example",
        aliases=(),
        availability=availability,
        files=files,
        homepage="https://example.test",
        paper=None,
        license=None,
        notes=None,
    )


class CacheRootTests(unittest.TestCase):
    def test_explicit_destination_wins(self) -> None:
        with patch.dict(os.environ, {"MLSCA_BENCH_DATA": "/ignored"}):
            self.assertEqual(cache_root("~/datasets"), Path("~/datasets").expanduser())

    def test_environment_destination(self) -> None:
        with patch.dict(os.environ, {"MLSCA_BENCH_DATA": "/data/mlsca"}):
            self.assertEqual(cache_root(), Path("/data/mlsca"))

    def test_platform_cache_fallback(self) -> None:
        fake = FakePooch()
        with patch.dict(os.environ, {}, clear=True), patch(
            "mlsca_bench.datasets.download._load_pooch", return_value=fake
        ):
            self.assertEqual(cache_root(), Path("/cache/mlsca-bench"))


class AvailabilityTests(unittest.TestCase):
    def test_manual_download_is_rejected_before_loading_pooch(self) -> None:
        dataset = make_dataset(availability="manual")
        with patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch") as load_pooch:
            with self.assertRaisesRegex(ManualDownloadRequired, "manual download"):
                download_dataset("example")
            load_pooch.assert_not_called()

    def test_unavailable_download_is_rejected(self) -> None:
        dataset = make_dataset(availability="unavailable")
        with patch("mlsca_bench.datasets.download.get_dataset", return_value=dataset):
            with self.assertRaises(DatasetUnavailable):
                download_dataset("example")

    def test_custom_download_is_rejected(self) -> None:
        dataset = make_dataset(availability="custom")
        with patch("mlsca_bench.datasets.download.get_dataset", return_value=dataset):
            with self.assertRaises(CustomDownloaderRequired):
                download_dataset("example")


class DownloadTests(unittest.TestCase):
    def test_direct_file_is_downloaded_with_registered_hash(self) -> None:
        fake = FakePooch()
        dataset = make_dataset(availability="automatic")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch", return_value=fake):
            paths = download_dataset("example", directory, progress=False)

        expected = Path(directory) / "example" / "traces.zip"
        self.assertEqual(paths, (expected,))
        self.assertEqual(len(fake.retrieve_calls), 1)
        call = fake.retrieve_calls[0]
        self.assertEqual(call["known_hash"], "sha256:" + "a" * 64)
        self.assertEqual(call["path"], Path(directory) / "example")
        self.assertEqual(call["downloader"], "http-downloader")
        self.assertFalse(call["progressbar"])
        self.assertEqual(
            fake.downloader_calls,
            [{"progressbar": False, "timeout": 120.0}],
        )

    def test_custom_http_timeout_is_forwarded_to_pooch(self) -> None:
        fake = FakePooch()
        dataset = make_dataset(availability="automatic")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch", return_value=fake):
            download_dataset("example", directory, timeout=300)

        self.assertEqual(fake.downloader_calls[0]["timeout"], 300)

    def test_non_positive_timeout_is_rejected_before_loading_pooch(self) -> None:
        dataset = make_dataset(availability="automatic")
        with patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch") as load_pooch:
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                download_dataset("example", timeout=0)
            load_pooch.assert_not_called()

    def test_negative_retries_are_rejected_before_loading_pooch(self) -> None:
        dataset = make_dataset(availability="automatic")
        with patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch") as load_pooch:
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                download_dataset("example", retries=-1)
            load_pooch.assert_not_called()

    def test_zip_archive_returns_extracted_paths(self) -> None:
        fake = FakePooch()
        dataset = make_dataset(availability="automatic", archive="zip")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch("mlsca_bench.datasets.download._load_pooch", return_value=fake):
            paths = download_dataset("example", directory)

        self.assertEqual(fake.retrieve_calls[0]["processor"], "unzip-processor")
        self.assertEqual(
            paths,
            (Path(directory) / "example" / "extracted" / "traces.h5",),
        )

    def test_transient_timeout_is_retried_with_backoff(self) -> None:
        fake = FakePooch()
        fake.retrieve_effects.append(ReadTimeout("TLS handshake timed out"))
        dataset = make_dataset(availability="automatic")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch(
            "mlsca_bench.datasets.download._load_pooch", return_value=fake
        ), patch("mlsca_bench.datasets.download.time.sleep") as sleep:
            paths = download_dataset(
                "example", directory, retries=2, retry_backoff=0.5
            )

        self.assertEqual(paths, (Path(directory) / "example" / "traces.zip",))
        self.assertEqual(len(fake.retrieve_calls), 2)
        sleep.assert_called_once_with(0.5)

    def test_transient_timeout_stops_after_retry_limit(self) -> None:
        fake = FakePooch()
        fake.retrieve_effects.extend(
            [ReadTimeout("first timeout"), ReadTimeout("second timeout")]
        )
        dataset = make_dataset(availability="automatic")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download.get_dataset", return_value=dataset
        ), patch(
            "mlsca_bench.datasets.download._load_pooch", return_value=fake
        ), patch("mlsca_bench.datasets.download.time.sleep") as sleep:
            with self.assertRaises(DatasetDownloadError) as caught:
                download_dataset(
                    "example", directory, retries=1, retry_backoff=0.25
                )

        self.assertIsInstance(caught.exception.__cause__, ReadTimeout)
        self.assertIn("stopped responding", str(caught.exception))
        self.assertEqual(len(fake.retrieve_calls), 2)
        sleep.assert_called_once_with(0.25)

    def test_kyber_download_uses_both_verified_archives(self) -> None:
        fake = FakePooch()
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download._load_pooch", return_value=fake
        ):
            download_dataset("kyber", directory, progress=False)

        self.assertEqual(
            [call["fname"] for call in fake.retrieve_calls],
            ["Reference-PPM.zip", "load-reference-ppm.zip"],
        )
        self.assertEqual(
            [call["known_hash"] for call in fake.retrieve_calls],
            [
                "md5:083158acd950e5d467a35bc4375e54b3",
                "md5:1cf6ecb6d3bdbfdeffb6e4ea6b925a58",
            ],
        )
        self.assertTrue(
            all(call["processor"] == "unzip-processor" for call in fake.retrieve_calls)
        )

    def test_ge_wars_download_uses_verified_hdf5_file(self) -> None:
        fake = FakePooch()
        with tempfile.TemporaryDirectory() as directory, patch(
            "mlsca_bench.datasets.download._load_pooch", return_value=fake
        ):
            paths = download_dataset("GE Wars 2025", directory, progress=False)

        expected = Path(directory) / "ge-wars" / "CHES_Challenge.h5"
        self.assertEqual(paths, (expected,))
        self.assertEqual(len(fake.retrieve_calls), 1)
        call = fake.retrieve_calls[0]
        self.assertEqual(call["fname"], "CHES_Challenge.h5")
        self.assertEqual(
            call["known_hash"],
            "sha256:132ae2e9a8213c983bf3b63449e9572d5d71d3b376a75b236415d4a728b9379f",
        )
        self.assertIsNone(call["processor"])


class GcsMd5VerificationTests(unittest.TestCase):
    """The GCS backend must verify the server-side MD5 (base64) when present."""

    @staticmethod
    def _blob(md5_b64):
        return type("Blob", (), {"md5_hash": md5_b64})()

    def test_matching_md5_passes(self):
        import base64, hashlib
        from mlsca_bench.datasets.download import _gcs_md5_ok
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obj"
            p.write_bytes(b"scaaml shard bytes")
            md5 = base64.b64encode(hashlib.md5(b"scaaml shard bytes").digest()).decode()
            self.assertTrue(_gcs_md5_ok(p, self._blob(md5)))

    def test_mismatching_md5_fails(self):
        from mlsca_bench.datasets.download import _gcs_md5_ok
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obj"
            p.write_bytes(b"corrupted")
            self.assertFalse(_gcs_md5_ok(p, self._blob("AAAAAAAAAAAAAAAAAAAAAA==")))

    def test_composite_object_without_md5_is_ok(self):
        from mlsca_bench.datasets.download import _gcs_md5_ok
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obj"
            p.write_bytes(b"x")
            self.assertTrue(_gcs_md5_ok(p, self._blob(None)))


if __name__ == "__main__":
    unittest.main()
