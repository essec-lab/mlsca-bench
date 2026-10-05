# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Saved user datasets (4), the mlsca-bench command (5) and cache management (6)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlsca_bench import (
    cache_usage,
    cached_datasets,
    get_dataset,
    list_datasets,
    register_dataset,
    remove_cached_dataset,
    saved_datasets,
    unregister_dataset,
    user_registry_path,
)
from mlsca_bench.cli import main
from mlsca_bench.datasets.registry import DatasetRegistryError, is_user_dataset

ENTRY = {
    "adapter": "hdf5",
    "adapter_config": {"splits": {"all": "/"}, "default_split": "all", "traces_dataset": "traces"},
    "metadata": {"format": "hdf5", "algorithm": "AES-128", "measurement": "power",
                 "platform": "other", "countermeasures": ["none"], "key": "fixed"},
}


@pytest.fixture
def user_file(tmp_path, monkeypatch):
    path = tmp_path / "config" / "datasets.json"
    monkeypatch.setenv("MLSCA_BENCH_USER_REGISTRY", str(path))
    yield path
    for name in ("saved-lab", "session-lab", "file-lab"):
        try:
            unregister_dataset(name)
        except Exception:
            pass


def _new_session(code: str, env_file: Path) -> str:
    env = {**os.environ, "MLSCA_BENCH_USER_REGISTRY": str(env_file)}
    return subprocess.run([sys.executable, "-c", code], env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_save_keeps_the_dataset_for_new_sessions(user_file, tmp_path):
    data = tmp_path / "traces.h5"
    data.write_bytes(b"")
    register_dataset("saved-lab", ENTRY, path=data, save=True)
    assert user_registry_path() == user_file
    assert saved_datasets() == ("saved-lab",)
    stored = json.loads(user_file.read_text())["saved-lab"]
    assert stored["path"] == str(data.resolve())          # absolute, works from any folder

    out = _new_session("from mlsca_bench.datasets.registry import local_dataset_path as p;"
                       "print(p('saved-lab'))", user_file)
    assert out == str(data.resolve())


def test_session_only_is_not_saved(user_file):
    register_dataset("session-lab", ENTRY)
    assert saved_datasets() == ()
    assert not user_file.exists()


def test_registering_a_saved_dataset_again_replaces_it(user_file):
    register_dataset("saved-lab", ENTRY, save=True)
    # an old script that registers it every session must keep working
    out = _new_session("from mlsca_bench import register_dataset as r;"
                       f"print(r('saved-lab', {ENTRY!r}).name)", user_file)
    assert out == "saved-lab"


def test_forget_removes_it_from_the_file(user_file):
    register_dataset("saved-lab", ENTRY, save=True)
    unregister_dataset("saved-lab", forget=True)
    assert saved_datasets() == ()
    assert _new_session("from mlsca_bench.datasets.registry import has_dataset;"
                        "print(has_dataset('saved-lab'))", user_file) == "False"


def test_broken_saved_file_names_the_file(user_file):
    user_file.parent.mkdir(parents=True)
    user_file.write_text('{"bad": {"adapter": "nope"}}')
    env = {**os.environ, "MLSCA_BENCH_USER_REGISTRY": str(user_file)}
    result = subprocess.run([sys.executable, "-c", "import mlsca_bench"], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0 and str(user_file) in result.stderr


def test_builtin_only_excludes_user_datasets(user_file):
    register_dataset("session-lab", ENTRY)
    assert len(list_datasets(builtin_only=True)) == 64
    assert "session-lab" in [s.name for s in list_datasets()]


# ---- command line --------------------------------------------------------------------------

def test_cli_list_info_and_version(capsys):
    assert main(["list", "--algorithm", "ascon"]) == 0
    out = capsys.readouterr().out
    assert "ascon-cw-protected" in out and "3 dataset(s)" in out
    assert main(["info", "ascad-f"]) == 0
    out = capsys.readouterr().out
    assert "ascadf" in out and "4.4 GB" in out and "ATMEGA8515" in out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "mlsca-bench" in capsys.readouterr().out


def test_cli_unknown_dataset_is_a_clean_error(capsys):
    assert main(["info", "ascadd"]) == 1
    err = capsys.readouterr().err
    assert "Did you mean" in err and "Traceback" not in err


def test_cli_add_and_forget(user_file, tmp_path, capsys):
    registry = tmp_path / "mine.json"
    registry.write_text(json.dumps({"file-lab": {**ENTRY, "path": "data/traces.h5"}}))
    assert main(["add", str(registry)]) == 0
    assert saved_datasets() == ("file-lab",)
    stored = json.loads(user_file.read_text())["file-lab"]["path"]
    assert stored == str((tmp_path / "data" / "traces.h5").resolve())
    assert main(["list", "--mine"]) == 0
    assert "file-lab" in capsys.readouterr().out
    assert main(["forget", "file-lab"]) == 0
    assert saved_datasets() == () and not is_user_dataset("file-lab")


def test_python_dash_m_works():
    result = subprocess.run([sys.executable, "-m", "mlsca_bench", "list", "--measurement", "electromagnetic"],
                            capture_output=True, text=True, check=True)
    assert "13 dataset(s)" in result.stdout


# ---- cache -------------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    (root / "ascadf").mkdir(parents=True)
    (root / "ascadf" / "ASCAD.h5").write_bytes(b"x" * 3000)
    (root / "aes-hd-zaid").mkdir()
    (root / "aes-hd-zaid" / "a.npy").write_bytes(b"x" * 1000)
    (root / "leftover").mkdir()
    monkeypatch.setenv("MLSCA_BENCH_DATA", str(root))
    return root


def test_cache_listing_and_usage(cache):
    entries = cached_datasets()
    assert [(e.name, e.size_bytes, e.known) for e in entries] == [
        ("ascadf", 3000, True), ("aes-hd-zaid", 1000, True), ("leftover", 0, False)]
    assert cache_usage() == 4000


def test_remove_by_alias_frees_space(cache):
    assert remove_cached_dataset("ascad-f") == 3000
    assert not (cache / "ascadf").exists() and (cache / "aes-hd-zaid").exists()
    with pytest.raises(FileNotFoundError, match="not in the cache"):
        remove_cached_dataset("ascadf")


@pytest.mark.parametrize("name", ["..", "../cache", "/etc", "ascadf/../.."])
def test_remove_never_leaves_the_cache(cache, name):
    with pytest.raises((DatasetRegistryError, FileNotFoundError)):
        remove_cached_dataset(name)
    assert (cache / "ascadf" / "ASCAD.h5").exists() and cache.parent.exists()


def test_cli_cache(cache, capsys):
    assert main(["cache"]) == 0
    out = capsys.readouterr().out
    assert "ascadf" in out and "unknown folder" in out and "Total:" in out
    assert main(["cache", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(cache)
    assert main(["cache", "remove", "aes-hd-zaid"]) == 0
    assert not (cache / "aes-hd-zaid").exists()


def test_misspelled_adapter_is_rejected_with_a_suggestion():
    with pytest.raises(DatasetRegistryError, match=r"unknown adapter 'hdf'; did you mean 'hdf5'"):
        register_dataset("session-lab", {**ENTRY, "adapter": "hdf"})


def test_observed_fields_are_validated_and_shown(capsys):
    from mlsca_bench.datasets.registry import RegistryValidationError

    spook = get_dataset("ches-ctf-2020-spook-sw3")
    assert "masks" in spook.fields["profiling"] and "masks" not in spook.fields["attack"]
    assert get_dataset("smaesh-a7").fields is None              # not checked through the reader yet
    with pytest.raises(RegistryValidationError, match="unknown fields"):
        register_dataset("session-lab", {**ENTRY, "fields": ["traces", "plain"]})
    assert main(["list", "--with", "plaintexts", "ciphertexts", "keys"]) == 0
    out = capsys.readouterr().out
    assert "ches-ctf-2018" in out and "aes-hd-git" not in out
    assert main(["info", "ches-ctf-2020-spook-sw3"]) == 0
    assert "attack: traces, plaintexts, keys" in capsys.readouterr().out
