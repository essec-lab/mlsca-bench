# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RAR archive processor (AES-PTv2 gdrive files are RAR archives)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlsca_bench.datasets import download as dl
from mlsca_bench.datasets.errors import ArchiveToolMissing


def test_rar_is_a_registered_processor() -> None:
    assert "rar" in dl.PROCESSORS

    class _F:
        archive = "rar"

    proc = dl._processor_for(_F(), None)
    assert isinstance(proc, dl.RarProcessor)


def test_rar_missing_tool_gives_actionable_error(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "AES_PTv2_STM32F4_D1.h5"
    archive.write_bytes(b"Rar!\x1a\x07\x01\x00fake")
    monkeypatch.setattr(dl.shutil, "which", lambda name: None)  # no unar/unrar
    monkeypatch.setattr(dl, "_find_tool", lambda *names: None)   # and no 7-Zip (Windows runners have one)

    with pytest.raises(ArchiveToolMissing) as excinfo:
        dl.RarProcessor()(str(archive), "download", None)
    msg = str(excinfo.value)
    assert "unar" in msg and "unrar" in msg
    assert "path=" in msg              # manual-download fallback instructions


def test_rar_extraction_returns_inner_files(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "AES_PTv2_STM32F4_D1.h5"
    archive.write_bytes(b"Rar!\x1a\x07\x01\x00fake")
    monkeypatch.setattr(dl.shutil, "which", lambda name: "/usr/bin/unar" if name == "unar" else None)

    def fake_run(cmd, check):
        # Simulate unar writing the inner .h5 into the output directory.
        out = Path(str(archive) + ".extracted")
        out.mkdir(parents=True, exist_ok=True)
        (out / "AES_PTv2_D1.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
        return None

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    files = dl.RarProcessor()(str(archive), "download", None)
    assert len(files) == 1 and files[0].endswith("AES_PTv2_D1.h5")

    # A second call reuses the existing extraction (idempotent, no re-run).
    monkeypatch.setattr(dl.subprocess, "run",
                        lambda *a, **k: pytest.fail("should not re-extract"))
    again = dl.RarProcessor()(str(archive), "download", None)
    assert again == files


def test_aesptv2_entries_are_automatic_rar() -> None:
    from mlsca_bench.datasets.registry import get_dataset

    for name in ("aes-ptv2-pinata", "aes-ptv2-stm32f4", "aes-ptv2-stm32f4-ms2"):
        spec = get_dataset(name)
        assert spec.availability == "automatic"
        assert all(f.archive == "rar" for f in spec.files)
