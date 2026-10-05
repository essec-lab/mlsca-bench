# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""The free-space check and downloading only some files of a dataset."""

from __future__ import annotations

import pytest

from mlsca_bench import download_dataset, get_dataset
from mlsca_bench.datasets import download
from mlsca_bench.datasets.errors import InsufficientDiskSpace


def test_not_enough_space_is_refused_before_downloading(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(download, "_retrieve_with_retries", lambda *a, **k: called.append(1))
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: type("U", (), {"free": 10**6})())
    with pytest.raises(InsufficientDiskSpace, match="needs at least 12.2 GB .including room to unpack.*destination="):
        download_dataset("ascadf", tmp_path, progress=False)
    assert not called
    monkeypatch.setattr(download, "_retrieve_with_retries",
                        lambda pooch, *, path, fname, **_: (path.mkdir(parents=True, exist_ok=True),
                                                            (path / fname).write_bytes(b"x"), str(path / fname))[2])
    download_dataset("ascadf", tmp_path, progress=False, check_space=False)   # override works


def test_only_selected_files_are_downloaded(tmp_path, monkeypatch, capsys):
    fetched = []

    def fake(pooch, *, path, fname, **_):
        fetched.append(fname)
        path.mkdir(parents=True, exist_ok=True)
        (path / fname).write_bytes(b"x")
        return str(path / fname)

    monkeypatch.setattr(download, "_retrieve_with_retries", fake)
    monkeypatch.setattr(download, "_processor_for", lambda *a: None)
    download_dataset("smaesh-a7", tmp_path, files="*-vk0*", check_space=False)
    assert fetched == ["smaesh-dataset-A7_d2-vk0.tar.zstd"]
    assert "1 of 3 files" in capsys.readouterr().out
    with pytest.raises(ValueError, match="matches none"):
        download_dataset("smaesh-a7", tmp_path, files="*nope*")


def test_split_files_are_declared_for_smaesh():
    for name in ("smaesh-a7", "smaesh-s6"):
        split_files = get_dataset(name).adapter_config["split_files"]
        assert set(split_files) == {"profiling", "attack"}




def test_measured_disk_size_is_used_for_the_space_check(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: type("U", (), {"free": 10**6})())
    spec = get_dataset("galactics-attack-data")
    assert spec.disk_bytes > 5 * spec.size_bytes                      # 22.6 GB download, 115.6 GB unpacked
    with pytest.raises(InsufficientDiskSpace, match="115.6 GB"):
        download_dataset("galactics-attack-data", tmp_path, progress=False)
    v2 = get_dataset("dpacontest-v2")                                 # split archive: parts + joined file too
    with pytest.raises(InsufficientDiskSpace, match="19.1 GB"):
        download_dataset("dpacontest-v2", tmp_path, progress=False)
    assert v2.disk_bytes + 2 * v2.size_bytes > 19e9
