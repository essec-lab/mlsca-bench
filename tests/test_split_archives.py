# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Datasets shipped as split archives (.zip.partN, .tar.bz2.partN) or .tar.zstd."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from mlsca_bench import download_dataset, get_dataset, register_dataset, unregister_dataset
from mlsca_bench.datasets import download
from mlsca_bench.datasets.registry import RegistryValidationError

META = {"format": "binary", "algorithm": "DES", "measurement": "power"}


def _split(data: bytes, n: int) -> list[bytes]:
    size = len(data) // n + 1
    return [data[i * size:(i + 1) * size] for i in range(n)]


def _fake_server(chunks: dict[str, bytes], calls: list[str]):
    def fake(pooch, *, path, fname, **_):
        calls.append(fname)
        path.mkdir(parents=True, exist_ok=True)
        (path / fname).write_bytes(chunks[fname])
        return str(path / fname)
    return fake


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in files.items():
            z.writestr(name, body)
    return buf.getvalue()


def _tar_bytes(files: dict[str, bytes], mode: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as t:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            t.addfile(info, io.BytesIO(body))
    return buf.getvalue()


@pytest.fixture
def split_dataset(request):
    names = []

    def make(name, base, archive, pieces):
        files = [{"url": f"https://example.org/{base}.part{i}", "filename": f"{base}.part{i}",
                  "known_hash": "sha256:" + "0" * 64, "archive": archive} for i in range(len(pieces))]
        files.reverse()                       # order in the registry must not matter
        register_dataset(name, {"adapter": "agilent-wave", "metadata": META, "files": files})
        names.append(name)
        return {f"{base}.part{i}": p for i, p in enumerate(pieces)}

    yield make
    for name in names:
        unregister_dataset(name)


@pytest.mark.parametrize("archive, base, build", [
    ("zip-parts", "traces.zip", lambda files: _zip_bytes(files)),
    ("tar.bz2-parts", "traces.tar.bz2", lambda files: _tar_bytes(files, "w:bz2")),
])
def test_parts_are_joined_extracted_and_reused(tmp_path, monkeypatch, split_dataset, archive, base, build):
    content = {"set/wave_k=01_m=02_c=03.bin": b"\x01" * 5000, "set/wave_k=01_m=04_c=05.bin": b"\x02" * 5000}
    chunks = split_dataset("t-split", base, archive, _split(build(content), 4))
    calls: list[str] = []
    monkeypatch.setattr(download, "_retrieve_with_retries", _fake_server(chunks, calls))

    files = download_dataset("t-split", tmp_path, progress=False)
    assert sorted(p.name for p in files) == sorted(n.split("/")[-1] for n in content)
    assert all(p.read_bytes() == content["set/" + p.name] for p in files)
    assert calls == [f"{base}.part{i}" for i in range(4)]            # joined in part order
    folder = tmp_path / "t-split"
    assert not list(folder.glob("*.part*")) and not (folder / base).exists()   # only extracted kept

    again = download_dataset("t-split", tmp_path, progress=False)
    assert sorted(again) == sorted(files) and len(calls) == 4           # no new download


def test_damaged_extraction_is_redone(tmp_path, monkeypatch, split_dataset):
    chunks = split_dataset("t-split", "t.zip", "zip-parts", _split(_zip_bytes({"a.bin": b"x" * 999}), 2))
    calls: list[str] = []
    monkeypatch.setattr(download, "_retrieve_with_retries", _fake_server(chunks, calls))
    (extracted,) = download_dataset("t-split", tmp_path, progress=False)
    extracted.write_bytes(b"x")                                          # truncated
    (again,) = download_dataset("t-split", tmp_path, progress=False)
    assert again.read_bytes() == b"x" * 999 and len(calls) == 4


def test_parts_must_be_named_partN():
    with pytest.raises(RegistryValidationError, match="part0"):
        register_dataset("t-bad", {"adapter": "agilent-wave", "metadata": META, "files": [
            {"url": "https://example.org/a.zip", "filename": "a.zip", "known_hash": None, "archive": "zip-parts"}]})


def test_builtin_split_datasets_are_marked():
    for name in ("dpacontest-v1-secmatv1-asic", "dpacontest-v2", "dpacontest-v3-secmatv3-des-20071219"):
        assert all(f.archive in ("zip-parts", "tar.bz2-parts") for f in get_dataset(name).files), name
    for name in ("smaesh-a7", "smaesh-s6"):
        assert all(f.archive == "tar.zstd" for f in get_dataset(name).files), name


def _zstd_or_skip():
    try:
        from compression import zstd
        return zstd.compress
    except ImportError:
        zstandard = pytest.importorskip("zstandard")
        return zstandard.ZstdCompressor().compress


def test_tar_zstd_is_extracted_once(tmp_path):
    compress = _zstd_or_skip()
    archive = tmp_path / "smaesh-dataset-A7_d2-vk0.tar.zstd"
    archive.write_bytes(compress(_tar_bytes({"smaesh-dataset-A7_d2-vk0/manifest.json": b"{}",
                                             "smaesh-dataset-A7_d2-vk0/traces/0000.npy": b"n" * 100}, "w")))
    processor = download.ZstdTarProcessor()
    files = processor(str(archive), "download", None)
    assert sorted(download.Path(p).name for p in files) == ["0000.npy", "manifest.json"]
    assert "vk0" in files[0]                                             # the reader picks splits by folder name
    stamp = [__import__("os").stat(p).st_mtime_ns for p in files]
    again = processor(str(archive), "fetch", None)
    assert [__import__("os").stat(p).st_mtime_ns for p in again] == stamp


def test_tar_with_unsafe_paths_is_refused(tmp_path):
    compress = _zstd_or_skip()
    archive = tmp_path / "evil.tar.zstd"
    archive.write_bytes(compress(_tar_bytes({"../escaped.txt": b"x"}, "w")))
    with pytest.raises(Exception):
        download.ZstdTarProcessor()(str(archive), "download", None)
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_zip_rejected_by_python_falls_back_to_7z(tmp_path, monkeypatch):
    archive = tmp_path / "old.zip"
    archive.write_bytes(_zip_bytes({"a.bin": b"x"}))

    class Broken:
        def __init__(self, *a, **k):
            raise zipfile.BadZipFile("Bad magic number for file header")

    used = []
    monkeypatch.setattr(zipfile, "ZipFile", Broken)
    monkeypatch.setattr(download.Deflate64ZipProcessor, "extract_with_7z",
                        staticmethod(lambda a, t: (used.append(a), t.mkdir(parents=True, exist_ok=True),
                                                   (t / "a.bin").write_bytes(b"x"))))
    download._extract_archive(archive, "zip", tmp_path / "out")
    assert used == [archive] and (tmp_path / "out" / "a.bin").read_bytes() == b"x"


# ---- partial downloads from folder sources (Dataverse, Google Cloud, Hugging Face) ------------

def test_dataverse_patterns_select_files_and_keep_separate_records(tmp_path, monkeypatch):
    listing = [
        {"directoryLabel": "sw3/random_key", "dataFile": {"id": 1, "filename": "rkey_sw3_0.npz", "filesize": 3}},
        {"directoryLabel": "sw3/fixed_key/key_0", "dataFile": {"id": 2, "filename": "fkey_sw3_K0_0.npz", "filesize": 3}},
        {"directoryLabel": "sw3/fixed_key/key_1", "dataFile": {"id": 3, "filename": "fkey_sw3_K1_0.npz", "filesize": 3}},
    ]
    fetched = []

    class Response:
        def __init__(self, payload=None, body=b"abc"):
            self.payload, self.body = payload, body
        def raise_for_status(self):
            pass
        def json(self):
            return self.payload
        def iter_content(self, chunk_size):
            yield self.body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_get(url, params=None, stream=False, timeout=None):
        if "/api/access/datafile/" in url:
            fetched.append(url.rsplit("/", 1)[1])
            return Response()
        return Response({"data": {"latestVersion": {"files": listing}}})

    monkeypatch.setattr(download.requests, "get", fake_get)
    assert download.download_estimate("ches-ctf-2020-spook-sw3", "fkey_sw3_K0_*", tmp_path) == 3
    download_dataset("ches-ctf-2020-spook-sw3", tmp_path, files="fkey_sw3_K0_*", progress=False)
    assert fetched == ["2"]
    download_dataset("ches-ctf-2020-spook-sw3", tmp_path, files="fkey_sw3_K0_*", progress=False)
    assert fetched == ["2"]                                    # selection recorded: no new request
    download_dataset("ches-ctf-2020-spook-sw3", tmp_path, progress=False)
    assert sorted(fetched) == ["1", "2", "3"]                  # a partial record never counts as complete
    with pytest.raises(ValueError, match="matches no file"):
        download.download_estimate("ches-ctf-2020-spook-sw3", "nope*", tmp_path / "other")


def test_spook_and_scaaml_declare_split_files():
    spook = get_dataset("ches-ctf-2020-spook-hw2").adapter_config["split_files"]
    assert list(spook["attack"]) == ["fkey_hw2_K0_*.npz"] and list(spook["profiling"]) == ["rkey_hw2_*.npz"]
    scaaml = get_dataset("scaaml-ecc-gpam-cm1").adapter_config["split_files"]
    assert all("info.json" in v for v in scaaml.values())


@pytest.mark.parametrize("name, link", [("../escaped.txt", None), ("/abs.txt", None),
                                        ("ok/../../escaped.txt", None), ("ln", "../../outside"), ("ln", "/etc/passwd")])
def test_unsafe_tar_entries_are_refused_on_every_python(tmp_path, monkeypatch, name, link):
    from mlsca_bench.datasets.errors import DatasetDownloadError

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as t:
        info = tarfile.TarInfo(name)
        if link:
            info.type, info.linkname = tarfile.SYMTYPE, link
            t.addfile(info)
        else:
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
    archive = tmp_path / "evil.tar.bz2"
    archive.write_bytes(buf.getvalue())
    original = tarfile.TarFile.extractall

    def without_filter(self, path=".", members=None, *, numeric_owner=False, **kwargs):
        if "filter" in kwargs:                          # behave like Python before 3.10.12
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")
        return original(self, path, members, numeric_owner=numeric_owner)

    monkeypatch.setattr(tarfile.TarFile, "extractall", without_filter)
    with pytest.raises(DatasetDownloadError, match="unsafe"):
        download._extract_archive(archive, "tar.bz2", tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path.parent / "escaped.txt").exists()
