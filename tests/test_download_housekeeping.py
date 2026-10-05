# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Leftovers of interrupted downloads are cleaned up; the size is announced first."""

from __future__ import annotations

import os
import time

from mlsca_bench import download_dataset, get_dataset, list_datasets
from mlsca_bench.datasets import download


def _fake_retrieve(pooch, *, path, fname, **_):
    target = path / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"data")
    return str(target)


def test_old_partial_is_removed_and_fresh_one_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_retrieve_with_retries", _fake_retrieve)
    folder = tmp_path / "aes-hd-zaid"
    folder.mkdir()
    old, fresh = folder / "tmpold123", folder / "tmpnew456"
    old.write_bytes(b"x" * 1000)
    fresh.write_bytes(b"y" * 1000)
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    download_dataset("aes-hd-zaid", tmp_path, progress=False)

    assert not old.exists()          # interrupted long ago: removed
    assert fresh.exists()            # could be a download in progress: kept


def test_size_is_announced_before_first_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download, "_retrieve_with_retries", _fake_retrieve)
    download_dataset("aes-hd-zaid", tmp_path, progress=True)
    out = capsys.readouterr().out
    assert f"about {get_dataset('aes-hd-zaid').size}" in out
    assert "86.2 MB" in out

    download_dataset("aes-hd-zaid", tmp_path, progress=True)      # already there: silent
    assert "Downloading" not in capsys.readouterr().out


def test_size_field_is_optional_and_human_readable():
    assert get_dataset("scaaml-ecc-gpam-cm3").size == "4.0 TB"
    assert get_dataset("chameleon-base").size == "68.8 GB"
    from mlsca_bench.datasets.registry import format_size
    assert format_size(None) == "unknown"
    assert all(spec.size_bytes for spec in list_datasets(builtin_only=True))
