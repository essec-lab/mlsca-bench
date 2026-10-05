# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Windows behaviour, simulated: tool lookup, RAR via 7-Zip, 260-character paths."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from mlsca_bench.datasets import download
from mlsca_bench.datasets.errors import LongPathsRequired


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(download, "_on_windows", lambda: True)
    monkeypatch.setattr(download.shutil, "which", lambda name: None)       # nothing on PATH


def test_7zip_is_found_in_its_install_folder(windows, tmp_path, monkeypatch):
    exe = tmp_path / "7-Zip" / "7z.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert download._seven_zip() == str(exe)


def test_rar_falls_back_to_7zip(windows, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download, "_find_tool", lambda *names: "C:/7z.exe" if "7z" in names else None)

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        out = next(a for a in cmd if a.startswith("-o"))[2:]
        (download.Path(out) / "data.h5").write_bytes(b"x")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    archive = tmp_path / "set.rar"
    archive.write_bytes(b"Rar!")
    files = download.RarProcessor()(str(archive), "download", None)
    assert calls == ["C:/7z.exe"] and files[0].endswith("data.h5")


def _zip_with_long_name(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("d/" + "x" * 240 + ".bin", b"1")


def test_long_paths_are_refused_with_instructions(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_windows_long_paths_enabled", lambda: False)
    archive = tmp_path / "a.zip"
    _zip_with_long_name(archive)
    with pytest.raises(LongPathsRequired, match="LongPathsEnabled"):
        download._extract_archive(archive, "zip", tmp_path / "out")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as t:
        info = tarfile.TarInfo("d/" + "y" * 240)
        info.size = 1
        t.addfile(info, io.BytesIO(b"1"))
    (tmp_path / "b.tar.bz2").write_bytes(buf.getvalue())
    with pytest.raises(LongPathsRequired):
        download._extract_archive(tmp_path / "b.tar.bz2", "tar.bz2", tmp_path / "out2")


def test_long_paths_are_fine_when_enabled(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_windows_long_paths_enabled", lambda: True)
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("d/short.bin", b"1")
    download._extract_archive(archive, "zip", tmp_path / "out")
    assert (tmp_path / "out" / "d" / "short.bin").exists()


def test_a_failing_extraction_tool_is_explained(tmp_path, monkeypatch):
    from mlsca_bench.datasets.errors import DatasetDownloadError

    def failing(cmd, **kw):
        raise download.subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(download.subprocess, "run", failing)
    archive = tmp_path / "broken.rar"
    archive.write_bytes(b"Rar!")
    with pytest.raises(DatasetDownloadError, match="broken.rar with 7-Zip failed.*download it again"):
        download._run_tool(["7z", "x", str(archive)], archive, "7-Zip")
