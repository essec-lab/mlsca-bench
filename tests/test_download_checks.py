# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Fixes from the download check of 2026-09-30 (no network needed)."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import requests

from mlsca_bench import download_dataset, get_dataset, list_datasets, register_dataset, unregister_dataset
from mlsca_bench.datasets import download
from mlsca_bench.datasets.errors import DatasetDownloadError
from mlsca_bench.datasets.registry import RegistryValidationError

h5py = pytest.importorskip("h5py")

META = {"format": "hdf5", "algorithm": "AES-128", "measurement": "power"}


@pytest.fixture
def cleanup():
    yield
    for name in ("t-drive", "t-space", "t-dv", "t-flat"):
        try:
            unregister_dataset(name)
        except Exception:
            pass


def test_no_builtin_http_file_uses_a_drive_view_page():
    for spec in list_datasets(builtin_only=True):
        for f in spec.files:
            if f.backend == "http":
                assert "drive.google.com/file/" not in (f.url or ""), spec.name
                assert f.url == f.url.strip(), spec.name


def test_drive_view_link_on_http_is_rejected(cleanup):
    entry = {"adapter": "hdf5", "metadata": META, "files": [{
        "url": "https://drive.google.com/file/d/abc/view", "filename": "a.h5",
        "known_hash": "sha256:" + "0" * 64}]}
    with pytest.raises(RegistryValidationError, match="gdrive"):
        register_dataset("t-drive", entry)


def test_url_spaces_are_trimmed(cleanup):
    entry = {"adapter": "hdf5", "metadata": META, "files": [{
        "url": " https://example.org/a.h5 ", "filename": "a.h5", "known_hash": "sha256:" + "0" * 64}]}
    assert register_dataset("t-space", entry).files[0].url == "https://example.org/a.h5"


def _raising(error):
    def fake(*_args, **_kwargs):
        raise error
    return fake


@pytest.mark.parametrize("error, text", [
    (requests.exceptions.ConnectionError("down"), "Check your internet connection"),
    (requests.exceptions.ReadTimeout("slow"), "stopped responding"),
    (ValueError("SHA256 hash of downloaded file (x) does not match the known hash"), "checksum does not match"),
])
def test_network_errors_are_explained(tmp_path, monkeypatch, error, text):
    monkeypatch.setattr(download, "_retrieve_with_retries", _raising(error))
    with pytest.raises(DatasetDownloadError, match=text) as caught:
        download_dataset("wolfssl-ed25519", tmp_path, progress=False)
    assert "wolfssl-ed25519" in str(caught.value)
    assert caught.value.__cause__ is error        # the original stays available


def test_http_status_is_explained(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 404
    monkeypatch.setattr(download, "_retrieve_with_retries",
                        _raising(requests.exceptions.HTTPError("404", response=response)))
    with pytest.raises(DatasetDownloadError, match="HTTP 404.*report the broken link"):
        download_dataset("wolfssl-ed25519", tmp_path, progress=False)


def test_gdrive_creates_the_folder_and_reuses_extraction(tmp_path, monkeypatch):
    calls = []

    def fake_download(*, id, output, quiet, resume):
        from pathlib import Path
        assert Path(output).parent.is_dir()                 # gdown lists this folder first
        import zipfile
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("traces.bin", b"\x00" * 10)
        calls.append(id)
        return output

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_download))
    monkeypatch.setattr(download, "hash_matches", lambda *a, **k: True)
    first = download_dataset("present-2021-randomized-clock", tmp_path, progress=False)
    stamp = [p.stat().st_mtime_ns for p in first]
    again = download_dataset("present-2021-randomized-clock", tmp_path, progress=False)
    assert len(calls) == 1
    assert [p.stat().st_mtime_ns for p in again] == stamp  # not unzipped again


def test_folder_backends_are_not_rechecked_once_complete(tmp_path, monkeypatch, cleanup):
    calls = []

    def fake_dataverse(*, output, **_):
        output.mkdir(parents=True, exist_ok=True)
        (output / "a.bin").write_bytes(b"x" * 100)
        calls.append(1)
        return output

    monkeypatch.setattr(download, "_download_dataverse", fake_dataverse)
    register_dataset("t-dv", {"adapter": "npy", "metadata": {**META, "format": "numpy"}, "files": [{
        "backend": "dataverse", "url": "https://dv.example", "persistent_id": "doi:x", "path_prefix": "p/"}]})
    download_dataset("t-dv", tmp_path, progress=False)
    download_dataset("t-dv", tmp_path, progress=False)
    assert len(calls) == 1                                  # second call: no network, no hashing
    (tmp_path / "t-dv" / "a.bin").write_bytes(b"x")         # damaged -> fetched again
    download_dataset("t-dv", tmp_path, progress=False)
    assert len(calls) == 2


def test_flat_hdf5_byte_columns(tmp_path, cleanup):
    path = tmp_path / "ctf.h5"
    data = np.arange(3 * 48, dtype=np.uint8).reshape(3, 48)
    with h5py.File(path, "w") as f:
        f["profiling_traces"] = np.zeros((3, 10), np.float32)
        f["profiling_data"] = data
    register_dataset("t-flat", {
        "adapter": "flat-hdf5", "metadata": META,
        "adapter_config": {"splits": {"profiling": "profiling_traces"}, "default_split": "profiling",
                           "data": {"profiling": "profiling_data"},
                           "data_columns": {"plaintexts": [0, 16], "ciphertexts": [16, 32], "keys": [32, 48]}}},
        path=path)
    from mlsca_bench import load_dataset
    with load_dataset("t-flat") as ds:
        assert ds.plaintexts.shape == (3, 16)
        np.testing.assert_array_equal(ds.keys[1], data[1, 32:48])
        np.testing.assert_array_equal(ds[2].ciphertext, data[2, 16:32])
        assert {"plaintexts", "ciphertexts", "keys"} <= set(ds.available_fields)


def test_ches_ctf_2018_uses_the_real_file_layout():
    config = get_dataset("ches-ctf-2018").adapter_config
    assert get_dataset("ches-ctf-2018").adapter == "flat-hdf5"
    assert dict(config["splits"]) == {"profiling": "profiling_traces", "attack": "attacking_traces"}
    assert [list(v) for v in config["data_columns"].values()] == [[0, 16], [16, 32], [32, 48]]


def test_every_dataset_has_a_license_entry():
    for spec in list_datasets(builtin_only=True):
        assert spec.license, f"{spec.name}: use 'not stated' when the source gives no license"
        if spec.license_source:
            assert spec.license_source.startswith("https://"), spec.name
    assert get_dataset("aes-rd").license == "CC-BY-NC-4.0"          # non-commercial: keep visible


def test_connection_broken_mid_transfer_is_retried_and_explained(tmp_path, monkeypatch):
    calls = []

    class FakePooch:
        def retrieve(self, **kw):
            calls.append(1)
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")

    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    with pytest.raises(DatasetDownloadError, match="connection to .* broke"):
        try:
            download._retrieve_with_retries(FakePooch(), retries=2, retry_backoff=0)
        except requests.exceptions.ChunkedEncodingError as error:
            assert len(calls) == 3                                  # first try plus two retries
            raise download._explain_download_error(get_dataset("wolfssl-ed25519"),
                                                   get_dataset("wolfssl-ed25519").files[0], error)
