# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Check that every downloadable registry file is reachable and is real data.

Usage:  python scripts/check_links.py [--all]

For each HTTP file it fetches only the first bytes and flags:
- links that fail (HTTP errors, timeouts),
- Git LFS pointer files (a small text stub instead of the data),
- HTML pages served instead of data (e.g. a Google Drive "view" page).

Files on the ``gdrive`` backend are checked through Drive's direct-download
address, which is what gdown fetches.

Hugging Face, Google Cloud and Dataverse datasets are checked by listing their
files at the source: the listing must succeed and must not be empty (this
needs the ``huggingface`` extra for Hugging Face).

Exits with status 1 if a problem is found, so it can run on a schedule
(.github/workflows/links.yml).
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

from mlsca_bench import list_datasets
from mlsca_bench.datasets.download import FOLDER_BACKENDS, _remote_folder_files

HEADERS = {"Range": "bytes=0-511", "User-Agent": "mlsca-bench-link-check"}


def check(url: str, timeout: float = 30, retry_slow: bool = True) -> str | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as response:
            head = response.read(512)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}"
    except TimeoutError:
        # Some hosts (e.g. scidb.cn) are slow to send the first bytes: one more, longer try.
        if retry_slow:
            return check(url, timeout=120, retry_slow=False)
        return f"no answer within {timeout:.0f} s (server slow or down)"
    except Exception as error:  # noqa: BLE001 - report every failure
        return f"{type(error).__name__}: {error}"
    if head.startswith(b"version https://git-lfs"):
        return "Git LFS pointer instead of data (use the media.githubusercontent.com URL)"
    if "text/html" in content_type and b"<html" in head.lower():
        return "HTML page instead of data"
    return None


def _drive_direct(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    parts = parsed.path.split("/")
    file_id = parts[parts.index("d") + 1] if "d" in parts else parse_qs(parsed.query)["id"][0]
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def main() -> int:
    problems = 0
    checked = 0
    for spec in list_datasets(builtin_only=True):
        for dataset_file in spec.files:
            url = dataset_file.url or ""
            if dataset_file.backend == "gdrive":
                url = _drive_direct(url)
            elif dataset_file.backend in FOLDER_BACKENDS:
                checked += 1
                where = dataset_file.url or dataset_file.repo_id
                try:
                    listed = _remote_folder_files(dataset_file, timeout=60)
                    issue = None if listed else "no files at the source"
                except Exception as error:  # noqa: BLE001 - report every failure
                    issue = f"{type(error).__name__}: {error}"
                if issue:
                    problems += 1
                    print(f"PROBLEM {spec.name}: {issue}\n        {where}")
                continue
            elif dataset_file.backend != "http":
                continue
            checked += 1
            issue = check(url)
            if issue:
                problems += 1
                print(f"PROBLEM {spec.name}: {issue}\n        {url}")
    print(f"checked {checked} files, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
