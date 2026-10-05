# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Download and cache datasets described by the mlsca-bench registry."""

from __future__ import annotations

import base64
import contextlib
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from pooch.hashes import hash_matches

from .errors import (
    ArchiveToolMissing,
    CustomDownloaderRequired,
    DatasetDownloadError,
    DatasetUnavailable,
    InsufficientDiskSpace,
    LongPathsRequired,
    ManualDownloadRequired,
    MissingDependencyError,
)
from .registry import DatasetFile, DatasetSpec, format_size, get_dataset


logger = logging.getLogger(__name__)


DATA_DIRECTORY_ENV = "MLSCA_BENCH_DATA"
DEFAULT_DOWNLOAD_TIMEOUT = 120.0
DEFAULT_DOWNLOAD_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

def _on_windows() -> bool:
    return os.name == "nt"


def _find_tool(*names: str) -> str | None:
    """A command on PATH, or (on Windows) in its usual install folder.

    The Windows installers of 7-Zip and WinRAR do not add themselves to PATH.
    """

    for name in names:
        found = shutil.which(name)
        if found:
            return found
    if _on_windows():
        folders = {"7z": "7-Zip", "7za": "7-Zip", "unrar": "WinRAR"}
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                     os.environ.get("ProgramW6432")):
            for name in names:
                if base and name in folders:
                    candidate = Path(base) / folders[name] / f"{name}.exe"
                    if candidate.is_file():
                        return str(candidate)
    return None


_SEVEN_ZIP_HINT = (
    "Install 7-Zip and try again (Windows: https://www.7-zip.org, found automatically in its "
    "default folder; macOS: 'brew install sevenzip'; Debian/Ubuntu: 'apt-get install p7zip-full')."
)


def _seven_zip() -> str | None:
    return _find_tool("7z", "7zz", "7za")


def _run_tool(cmd: list[str], archive: Path, tool: str, *, quiet: bool = False) -> None:
    """Run an extraction tool; a failure explains itself instead of a raw CalledProcessError."""

    try:
        if quiet:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as error:
        raise DatasetDownloadError(
            f"Extracting {archive.name} with {tool} failed (exit code {error.returncode}). The "
            f"archive may be damaged or incomplete: delete {archive} and download it again."
        ) from error


def _windows_long_paths_enabled() -> bool:
    """Whether Windows allows paths over 260 characters (LongPathsEnabled)."""

    if not _on_windows():
        return True
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


WINDOWS_MAX_PATH = 259


def _check_path_lengths(target: Path, member_names, archive: Path) -> None:
    """On Windows without long paths, refuse an extraction whose paths would be too long."""

    if not _on_windows() or _windows_long_paths_enabled():
        return
    base = len(str(target.resolve())) + 1
    longest = max((base + len(name.replace("/", "\\")) for name in member_names), default=0)
    if longest <= WINDOWS_MAX_PATH:
        return
    shortest_root = len(r"C:\mlsca") + (base - len(str(cache_root().resolve()))) - 1
    shorter = longest - base + shortest_root
    advice = (
        f"a shorter cache folder is not enough for this dataset (still {shorter} characters with "
        "C:\\mlsca), so long paths must be enabled"
        if shorter > WINDOWS_MAX_PATH else
        "or use a shorter cache folder, e.g. set MLSCA_BENCH_DATA=C:\\mlsca"
    )
    raise LongPathsRequired(
        f"Extracting {archive.name} would create paths of up to {longest} characters, but Windows "
        f"only allows 260 unless long paths are enabled. Enable them once (PowerShell as "
        "administrator: New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
        "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force), then start Python again; "
        f"{advice}."
    )


class Deflate64ZipProcessor:
    """Extract ZIP archives that use Deflate64 compression."""

    @staticmethod
    def extract_with_7z(archive: Path, extract_dir: Path) -> None:
        seven_zip = _seven_zip()
        if seven_zip is None:
            raise MissingDependencyError(
                "7-Zip",
                feature="extracting Deflate64 ZIP archives and old zip files over 4 GB",
                install_hint=_SEVEN_ZIP_HINT,
            )
        extract_dir.mkdir(parents=True, exist_ok=True)
        _run_tool([seven_zip, "x", str(archive), f"-o{extract_dir}", "-y"], archive, "7-Zip", quiet=True)

    def __call__(self, fname, action, pooch):
        archive = Path(fname)
        extract_dir = Path(str(archive) + ".unzip")

        if extract_dir.exists():
            return [
                str(path)
                for path in extract_dir.rglob("*")
                if path.is_file()
            ]

        self.extract_with_7z(archive, extract_dir)

        return [
            str(path)
            for path in extract_dir.rglob("*")
            if path.is_file()
        ]

class SevenZipProcessor:
    """Extract a .7z archive after download."""

    def __call__(self, fname, action, pooch):
        try:
            import py7zr
        except ImportError as error:
            raise MissingDependencyError(
                "py7zr",
                extra="sevenzip",
                feature="extracting .7z archives",
            ) from error

        archive = Path(fname)
        extract_dir = archive.with_suffix("")

        if extract_dir.exists():
            return [
                str(path)
                for path in extract_dir.rglob("*")
                if path.is_file()
            ]

        extract_dir.mkdir(exist_ok=True)

        with py7zr.SevenZipFile(archive, mode="r") as seven_zip:
            seven_zip.extractall(path=extract_dir)

        return [
            str(path)
            for path in extract_dir.rglob("*")
            if path.is_file()
        ]

class RarProcessor:
    """Extract a RAR archive (incl. RAR5) using the ``unar`` or ``unrar`` tool.

    RAR is proprietary and has no pure-Python decoder, and ``7z`` fails on some
    RAR5 archives ("unsupported method"), so this shells out to a system tool.
    Used by AES-PTv2, whose Google-Drive files are RAR archives (some misnamed
    ``.h5``). If no tool is installed, a clear error explains how to install one
    or extract manually and load via ``path=``.
    """

    def __call__(self, fname, action, pooch):
        archive = Path(fname)
        extract_dir = Path(str(archive) + ".extracted")

        if extract_dir.exists():
            existing = [str(p) for p in extract_dir.rglob("*") if p.is_file()]
            if existing:
                return existing

        unar = _find_tool("unar")
        unrar = _find_tool("unrar")
        seven_zip = None if (unar or unrar) else _seven_zip()
        if unar is None and unrar is None and seven_zip is None:
            raise ArchiveToolMissing("unar, unrar, 7z", feature="Extracting RAR archives")

        extract_dir.mkdir(parents=True, exist_ok=True)
        if seven_zip is not None:
            # 7-Zip reads RAR5 (the Windows build and 7-Zip 22+ elsewhere); older p7zip may not.
            _run_tool([seven_zip, "x", str(archive), f"-o{extract_dir}", "-y"], archive, "7-Zip", quiet=True)
        elif unar is not None:
            _run_tool([unar, "-force-overwrite", "-output-directory", str(extract_dir), str(archive)],
                      archive, "unar")
        else:
            _run_tool([unrar, "x", "-y", str(archive), f"{extract_dir}{os.sep}"], archive, "unrar")

        extracted = [str(p) for p in extract_dir.rglob("*") if p.is_file()]
        if not extracted:
            raise RuntimeError(f"RAR extraction produced no files: {archive.name}")
        return extracted


def _load_pooch() -> Any:
    """Import Pooch with an actionable error for incomplete installations."""

    try:
        import pooch
    except ImportError as error:
        raise MissingDependencyError(
            "pooch",
            feature="dataset downloading",
        ) from error
    return pooch


def cache_root(destination: str | os.PathLike[str] | None = None) -> Path:
    """Return the user-selected or platform-default dataset cache directory.

    Precedence is: the explicit ``destination`` argument, the
    ``MLSCA_BENCH_DATA`` environment variable, then Pooch's operating-system
    cache directory for ``mlsca-bench``.
    """

    if destination is not None:
        return Path(destination).expanduser()

    configured = os.environ.get(DATA_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser()

    pooch = _load_pooch()
    return Path(pooch.os_cache("mlsca-bench"))

@contextlib.contextmanager
def _open_zstd_tar(archive: Path) -> Iterator[Any]:
    """Open a .tar.zstd for streaming extraction (built in from Python 3.14)."""

    import tarfile

    try:
        from compression import zstd  # noqa: F401  (Python 3.14+)
    except ImportError:
        zstd = None
    if zstd is not None:
        with tarfile.open(archive, mode="r|zst") as tar:
            yield tar
        return
    try:
        import zstandard
    except ImportError as error:
        raise MissingDependencyError(
            "zstandard", extra="zstd", feature="extracting .tar.zstd archives (before Python 3.14)"
        ) from error
    with archive.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader, \
            tarfile.open(fileobj=reader, mode="r|") as tar:
        yield tar


def _unsafe_tar_member(member: Any, root: Path) -> str | None:
    """Why a tar entry would write outside ``root`` (absolute path, '..', escaping link), else None."""

    name = member.name
    if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":") or ".." in Path(name).parts:
        return f"path {name!r}"
    if not (root / name).resolve().is_relative_to(root):
        return f"path {name!r}"
    if member.issym() or member.islnk():
        link = member.linkname
        if link.startswith(("/", "\\")) or (len(link) > 1 and link[1] == ":"):
            return f"link {name!r} -> {link!r}"
        base = root if member.islnk() else (root / name).parent
        if not (base / link).resolve().is_relative_to(root):
            return f"link {name!r} -> {link!r}"
    if member.isdev():
        return f"device file {name!r}"
    return None


def _extract_tar(tar: Any, target: Path, archive: Path | None = None) -> None:
    root = target.resolve()

    def checked(members):
        for member in members:       # streamed: check each entry before it is written
            problem = _unsafe_tar_member(member, root)
            if problem:
                # Checked here on every Python version: the 'data' filter below
                # does not exist before Python 3.10.12 / 3.11.4.
                raise DatasetDownloadError(
                    f"{(archive or target).name} contains an unsafe {problem} that would be written "
                    "outside the dataset folder; nothing more was extracted. The archive may have been "
                    "tampered with: please report it."
                )
            _check_path_lengths(target, [member.name], archive or target)
            yield member

    try:
        tar.extractall(target, members=checked(tar), filter="data")   # refuses absolute paths and '..'
    except TypeError:                              # Python < 3.12 without the filter argument
        tar.extractall(target, members=checked(tar))


def _extract_archive(archive: Path, kind: str, target: Path) -> None:
    """Extract ``archive`` (zip, tar.bz2 or tar.zstd) into ``target``."""

    import tarfile
    import zipfile

    target.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        try:
            with zipfile.ZipFile(archive) as zipped:
                _check_path_lengths(target, zipped.namelist(), archive)
                deflate64 = any(info.compress_type == 9 for info in zipped.infolist())
                if not deflate64:
                    zipped.extractall(target)
                    return
        except zipfile.BadZipFile as error:
            # Some old archives over 4 GB were written without zip64, so their
            # header offsets overflowed (DPA Contest v3, 2007-12-19). Python's
            # reader rejects them; 7-Zip reads them correctly.
            logger.warning("%s: %s; extracting with 7-Zip instead.", archive.name, error)
            shutil.rmtree(target, ignore_errors=True)
        Deflate64ZipProcessor.extract_with_7z(archive, target)
    elif kind == "tar.bz2":
        with tarfile.open(archive, mode="r:bz2") as tar:
            _extract_tar(tar, target, archive)
    elif kind == "tar.zstd":
        with _open_zstd_tar(archive) as tar:
            _extract_tar(tar, target, archive)
    else:
        raise ValueError(f"Unsupported archive type: {kind!r}")


EXTRACTED_RECORD = ".mlsca-extracted.json"


def _extracted_files(target: Path) -> list[Path] | None:
    """Files of a completed extraction, or None when it is missing or incomplete."""

    record = target / EXTRACTED_RECORD
    if not _complete_record_ok(target, record):
        return None
    return [target / name for name in json.loads(record.read_text(encoding="utf-8"))["files"]]


def _extract_once(archive: Path, kind: str, target: Path) -> list[Path]:
    """Extract into ``target`` via a temporary folder, then record the result."""

    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    _extract_archive(archive, kind, partial)
    shutil.rmtree(target, ignore_errors=True)
    try:
        partial.rename(target)
    except OSError as error:
        raise DatasetDownloadError(
            f"Could not replace {target}: {error}. On Windows a folder cannot be replaced while a "
            "file in it is open; close the datasets that use it (or restart Python) and try again."
        ) from error
    _write_complete_record(target, target / EXTRACTED_RECORD)
    return _extracted_files(target) or []


class ZstdTarProcessor:
    """Extract a .tar.zstd archive after download (Pooch processor)."""

    def __call__(self, fname, action, pooch):
        archive = Path(fname)
        target = Path(str(archive) + ".untar")
        done = _extracted_files(target)
        if done is not None and action == "fetch":
            return [str(p) for p in done]
        return [str(p) for p in _extract_once(archive, "tar.zstd", target)]


_PART_KINDS = {"zip-parts": "zip", "tar.bz2-parts": "tar.bz2"}
_PART_SUFFIX = re.compile(r"\.part(\d+)$")


def _parts_group(dataset: DatasetSpec, dataset_file: DatasetFile) -> tuple[str, list[DatasetFile]]:
    """The joined archive's name and all of its parts, in order."""

    base = _PART_SUFFIX.sub("", dataset_file.filename)
    parts = [
        f for f in dataset.files
        if f.archive == dataset_file.archive and f.filename and _PART_SUFFIX.sub("", f.filename) == base
    ]
    parts.sort(key=lambda f: int(_PART_SUFFIX.search(f.filename).group(1)))
    return base, parts


def _join_and_extract(directory: Path, base: str, kind: str, part_paths: list[Path]) -> list[Path]:
    """Join downloaded parts in order, extract, then delete parts and joined file."""

    joined = directory / base
    joining = directory / (base + ".joining")
    with joining.open("wb") as out:
        for part in part_paths:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=16 * 1024 * 1024)
    joining.replace(joined)
    files = _extract_once(joined, kind, directory / (base + ".extracted"))
    joined.unlink(missing_ok=True)
    for part in part_paths:                       # the extracted files are what is kept
        part.unlink(missing_ok=True)
    return files


PROCESSORS = {
    "zip": lambda p: p.Unzip(),
    "zip-deflate64": lambda p: Deflate64ZipProcessor(),
    "tar": lambda p: p.Untar(),
    "tar.gz": lambda p: p.Untar(),
    "gz": lambda p: p.Decompress(),
    "7z": lambda p: SevenZipProcessor(),
    "rar": lambda p: RarProcessor(),
    "tar.zstd": lambda p: ZstdTarProcessor(),
}

def _processor_for(dataset_file, pooch):
    if dataset_file.archive is None or dataset_file.archive in _PART_KINDS:
        return None                       # parts are joined after all are downloaded

    try:
        return PROCESSORS[dataset_file.archive](pooch)
    except KeyError:
        raise ValueError(f"Unsupported archive type: {dataset_file.archive!r}")

def _assert_automatic(dataset: DatasetSpec) -> None:
    information_url = dataset.homepage or dataset.paper
    if dataset.availability == "manual":
        raise ManualDownloadRequired(dataset.name, information_url)
    if dataset.availability == "unavailable":
        raise DatasetUnavailable(dataset.name, information_url)
    if dataset.availability == "custom":
        raise CustomDownloaderRequired(dataset.name, information_url)
    if dataset.availability != "automatic":
        raise DatasetUnavailable(dataset.name, information_url)


def _as_paths(result: str | os.PathLike[str] | list[str] | tuple[str, ...]) -> tuple[Path, ...]:
    if isinstance(result, (str, os.PathLike)):
        return (Path(result),)
    return tuple(Path(path) for path in result)


def _retrieve_with_retries(
    pooch: Any,
    *,
    retries: int,
    retry_backoff: float,
    **kwargs: Any,
) -> str | os.PathLike[str] | list[str] | tuple[str, ...]:
    """Retry a Pooch HTTP transfer after transient connection failures."""

    from requests.exceptions import ChunkedEncodingError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout

    # ChunkedEncodingError: the connection broke in the middle of the transfer.
    retryable_errors = (RequestsConnectionError, RequestsTimeout, ChunkedEncodingError, TimeoutError)
    for attempt in range(retries + 1):
        try:
            return pooch.retrieve(**kwargs)
        except retryable_errors:
            if attempt == retries:
                raise
            time.sleep(retry_backoff * (2**attempt))

    raise AssertionError("unreachable")


def _file_hash(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _download_dataverse(
    *,
    base_url: str,
    persistent_id: str,
    path_prefix: str,
    output: Path,
    timeout: float,
    patterns: Sequence[str] | None = None,
) -> Path:
    """Download a directory subset of a public Dataverse dataset."""

    output.mkdir(parents=True, exist_ok=True)

    if not persistent_id:
        raise ValueError(
            "Dataverse backend requires persistent_id"
        )

    if not path_prefix:
        raise ValueError(
            "Dataverse backend requires path_prefix"
        )

    logger.debug(
        "Dataverse request: base_url=%s, persistent_id=%s, path_prefix=%s",
        base_url,
        persistent_id,
        path_prefix,
    )

    response = requests.get(
        f"{base_url.rstrip('/')}/api/datasets/:persistentId/",
        params={"persistentId": persistent_id},
        timeout=timeout,
    )
    response.raise_for_status()

    files = response.json()["data"]["latestVersion"]["files"]

    prefix = path_prefix.strip("/")

    selected = [
        entry
        for entry in files
        if (
            entry.get("directoryLabel", "") == prefix
            or entry.get("directoryLabel", "").startswith(prefix + "/")
        )
    ]

    if not selected:
        raise RuntimeError(
            f"No Dataverse files found under {path_prefix!r}"
        )
    if patterns:
        selected = [
            entry for entry in selected
            if _matches(entry["dataFile"]["filename"], _relative(entry, prefix), patterns)
        ]
        if not selected:
            raise ValueError(f"files={list(patterns)!r} matches no file under {path_prefix!r}.")

    total_size = sum(
        entry["dataFile"].get("filesize", 0)
        for entry in selected
    )

    logger.info(
        "Dataverse subset %s contains %d files (%.2f GiB)",
        path_prefix,
        len(selected),
        total_size / 1024**3,
    )

    for index, entry in enumerate(selected, start=1):
        data_file = entry["dataFile"]

        file_id = data_file["id"]
        filename = data_file["filename"]
        directory = entry.get("directoryLabel", "")

        # Preserve hierarchy relative to selected prefix.
        relative_dir = directory[len(prefix):].lstrip("/")

        destination = output / relative_dir / filename
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checksum = data_file.get("checksum", {})
        algorithm = checksum.get("type", "").lower()
        expected = checksum.get("value", "").lower()

        if destination.exists():
            size_ok = (
                destination.stat().st_size
                == data_file.get("filesize")
            )

            # No published checksum: a file of the right size is the best check available.
            hash_ok = size_ok and not (algorithm and expected)

            if size_ok and algorithm and expected:
                hash_ok = (
                    _file_hash(destination, algorithm)
                    == expected
                )

            if size_ok and hash_ok:
                logger.info(
                    "[%d/%d] Using cached file %s",
                    index,
                    len(selected),
                    destination.name,
                )
                continue

        logger.info(
            "[%d/%d] Downloading %s",
            index,
            len(selected),
            destination.name,
        )

        download_url = (
            f"{base_url.rstrip('/')}"
            f"/api/access/datafile/{file_id}"
        )

        temporary = destination.with_suffix(
            destination.suffix + ".part"
        )

        with requests.get(
            download_url,
            stream=True,
            timeout=timeout,
        ) as download:
            download.raise_for_status()

            with temporary.open("wb") as stream:
                for chunk in download.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):
                    if chunk:
                        stream.write(chunk)

        if algorithm and expected:
            actual = _file_hash(
                temporary,
                algorithm,
            )

            if actual != expected:
                temporary.unlink(missing_ok=True)

                raise RuntimeError(
                    f"Checksum mismatch for {filename}: "
                    f"expected {algorithm}:{expected}, "
                    f"got {algorithm}:{actual}"
                )

        temporary.replace(destination)

    return output

def _gcs_md5_ok(path: Path, blob: Any) -> bool:
    """Check a local file against a GCS blob's server MD5 (base64).

    Composite (multipart) objects have no ``md5_hash``; there is nothing to
    verify against, so treat that as OK (size is still checked by the caller).
    """

    expected = getattr(blob, "md5_hash", None)
    if not expected:
        return True
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode() == expected


def _download_gcs(
    *,
    url: str,
    output: Path,
    patterns: Sequence[str] | None = None,
) -> Path:
    """Recursively download a public Google Cloud Storage prefix."""

    try:
        from google.cloud import storage
    except ImportError as error:
        raise MissingDependencyError(
            "google-cloud-storage",
            extra="gcs",
            feature="Google Cloud Storage downloads",
        ) from error

    # Keep the rest of the existing function here.
    if not url.startswith("gs://"):
        raise ValueError(
            f"Expected a gs:// URL, got {url!r}"
        )

    bucket_path = url[5:]
    bucket_name, _, prefix = bucket_path.partition("/")

    if not bucket_name or not prefix:
        raise ValueError(
            f"Expected a GCS bucket and prefix, got {url!r}"
        )

    prefix = prefix.rstrip("/") + "/"
    output.mkdir(parents=True, exist_ok=True)

    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(
        bucket_name,
        prefix=prefix,
    )

    found = False

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        found = True

        relative_name = blob.name[len(prefix):]
        if patterns and not _matches(relative_name.rsplit("/", 1)[-1], relative_name, patterns):
            continue

        destination = output / relative_name
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Don't download an already complete object again.
        if (
            destination.exists()
            and blob.size is not None
            and destination.stat().st_size == blob.size
            and _gcs_md5_ok(destination, blob)
        ):
            logger.info(
                "Using cached GCS file %s",
                destination,
            )
            continue

        logger.info(
            "Downloading gs://%s/%s",
            bucket_name,
            blob.name,
        )

        blob.download_to_filename(destination)

        # Verify the object's server-side MD5 (base64) when present; composite
        # objects have no md5Hash, so integrity there falls back to size.
        if not _gcs_md5_ok(destination, blob):
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for gs://{bucket_name}/{blob.name}"
            )

    if not found:
        if patterns:
            raise ValueError(f"files={list(patterns)!r} matches no file under {url}.")
        raise RuntimeError(
            f"No files found under {url}"
        )

    return output

def _download_google_drive(
    *,
    url: str,
    output: Path,
    max_retries: int = 5,
    progress: bool = True,
) -> Path:
    try:
        import gdown
    except ImportError as error:
        raise MissingDependencyError(
            "gdown",
            extra="gdrive",
            feature="Google Drive downloads",
        ) from error

    # Keep the rest of the existing function here.

    parsed = urlparse(url)
    file_id = None

    if parsed.netloc == "drive.google.com":
        parts = parsed.path.split("/")

        if "d" in parts:
            file_id = parts[parts.index("d") + 1]
        else:
            file_id = parse_qs(parsed.query).get("id", [None])[0]

    elif parsed.netloc == "drive.usercontent.google.com":
        file_id = parse_qs(parsed.query).get("id", [None])[0]

    if file_id is None:
        raise ValueError(
            f"Cannot determine Google Drive file id from {url}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)   # gdown lists this folder first

    for attempt in range(1, max_retries + 1):
        try:
            result = gdown.download(
                id=file_id,
                output=str(output),
                quiet=not progress,
                resume=True,
            )

            if result is None:
                raise RuntimeError(
                    f"Google Drive download failed for file {file_id}"
                )

            return Path(result)

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout,
        ) as error:

            if attempt == max_retries:
                raise RuntimeError(
                    f"Google Drive download failed after "
                    f"{max_retries} attempts"
                ) from error

            delay = min(5 * attempt, 30)

            logger.warning(
                "Google Drive connection interrupted "
                "(attempt %d/%d); retrying in %d seconds",
                attempt,
                max_retries,
                delay,
            )

            time.sleep(delay)

    raise RuntimeError("Google Drive download failed")

def _download_huggingface(
    dataset_file: DatasetFile,
    cache_dir: Path,
    patterns: Sequence[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise MissingDependencyError(
            "huggingface-hub",
            extra="huggingface",
            feature="Hugging Face downloads",
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)

    allow = dataset_file.allow_patterns
    if patterns:
        # Exact repo paths inside the registered subset that match the user's patterns.
        allow = [name for name, _size in _remote_folder_files(dataset_file) if _matches(
            name.rsplit("/", 1)[-1], name, patterns)]
        if not allow:
            raise ValueError(f"files={list(patterns)!r} matches no file of {dataset_file.repo_id}.")
    snapshot_download(
        repo_id=dataset_file.repo_id,
        repo_type=dataset_file.repo_type or "dataset",
        revision=dataset_file.revision,
        local_dir=cache_dir,
        allow_patterns=allow,
    )

    return cache_dir

STALE_PARTIAL_SECONDS = 3600


def _remove_stale_partials(directory: Path) -> None:
    """Delete temporary files left by interrupted downloads.

    Pooch writes each download to a ``tmp*`` file and renames it when complete.
    A file not written to for an hour belongs to a download that died; a running
    download keeps updating its file, so it is never touched.
    """

    if not directory.is_dir():
        return
    now = time.time()
    for candidate in directory.glob("tmp*"):
        if candidate.is_file() and now - candidate.stat().st_mtime > STALE_PARTIAL_SECONDS:
            size = candidate.stat().st_size
            candidate.unlink()
            logger.warning(
                "Removed %s: an incomplete earlier download (%.1f GB).", candidate, size / 1e9
            )


def _looks_downloaded(directory: Path) -> bool:
    """True when the dataset folder already holds files (then nothing is announced)."""

    return directory.is_dir() and any(not p.name.startswith("tmp") for p in directory.iterdir())


ISSUES_URL = "https://github.com/essec-lab/mlsca-bench/issues"


def _explain_download_error(
    dataset: DatasetSpec, dataset_file: DatasetFile, error: BaseException
) -> DatasetDownloadError | None:
    """Turn a network or checksum failure into a message that says what to do."""

    where = dataset_file.url or dataset_file.repo_id or ""
    host = urlparse(where).netloc or where
    kept = "Files that finished downloading are kept and are not downloaded again."
    more = f" Dataset page: {dataset.homepage}." if dataset.homepage else ""
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        code = error.response.status_code
        return DatasetDownloadError(
            f"Could not download {dataset.name}: {host} answered HTTP {code} for {where}. "
            f"The file may have moved or be temporarily unavailable.{more} If it persists, "
            f"please report the broken link at {ISSUES_URL}."
        )
    if isinstance(error, requests.exceptions.ChunkedEncodingError):
        return DatasetDownloadError(
            f"The connection to {host} broke while downloading {dataset.name} (after retrying). "
            f"This is usually a network drop or the computer going to sleep; try again. {kept}"
        )
    if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError)):
        # requests reports a read timeout during a streamed download as a ConnectionError
        timed_out = isinstance(error, (requests.exceptions.Timeout, TimeoutError)) or "timed out" in str(error)
        reason = "the server stopped responding" if timed_out else "no connection"
        return DatasetDownloadError(
            f"Could not reach {host} to download {dataset.name} ({reason}). Check your internet "
            f"connection or proxy and try again; for a slow server pass a larger timeout= "
            f"(default {DEFAULT_DOWNLOAD_TIMEOUT:.0f} s). {kept}"
        )
    if isinstance(error, ValueError) and "hash" in str(error).lower():
        return DatasetDownloadError(
            f"A file of {dataset.name} ({dataset_file.filename or where}) was downloaded but its "
            "checksum does not match the registry, so it was deleted. Usually the transfer was "
            "corrupted: try again. If it keeps happening, the authors may have changed the file; "
            f"please report it at {ISSUES_URL}."
        )
    return None


def _file_sizes(directory: Path) -> dict[str, int]:
    if not directory.is_dir():
        return {}
    return {
        str(p.relative_to(directory)): p.stat().st_size
        for p in directory.rglob("*")
        if p.is_file() and not p.name.startswith(".mlsca-complete-") and not p.name.endswith(".part")
    }


def _write_complete_record(directory: Path, record: Path) -> None:
    """Remember the files of a finished, verified folder download (name and size)."""

    record.write_text(json.dumps({"files": _file_sizes(directory)}, indent=0), encoding="utf-8")


RECORD_FULL_CHECK_LIMIT = 5000
RECORD_SAMPLE = 1000


def _complete_record_ok(directory: Path, record: Path) -> bool:
    """A folder recorded as complete still has every file, at its full size."""

    try:
        files = json.loads(record.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return False
    names = list(files)
    if len(names) > RECORD_FULL_CHECK_LIMIT:
        # Stat'ing hundreds of thousands of files takes most of a minute; an
        # evenly spread sample still catches a deleted or half-removed folder.
        step = len(names) / RECORD_SAMPLE
        names = [names[int(i * step)] for i in range(RECORD_SAMPLE)] + [names[-1]]
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size != files[name]:
            return False
    return bool(files)


FOLDER_BACKENDS = ("dataverse", "gcs", "huggingface")


def _matches(basename: str, relative: str, patterns: Sequence[str]) -> bool:
    """A file matches when its name or its path inside the dataset matches a pattern."""

    return any(fnmatch.fnmatch(basename, p) or fnmatch.fnmatch(relative, p) for p in patterns)


def _relative(entry: dict, prefix: str) -> str:
    directory = entry.get("directoryLabel", "")[len(prefix):].strip("/")
    name = entry["dataFile"]["filename"]
    return f"{directory}/{name}" if directory else name


def _remote_folder_files(dataset_file: DatasetFile, timeout: float = DEFAULT_DOWNLOAD_TIMEOUT) -> list[tuple[str, int | None]]:
    """(path inside the dataset, size) of every file behind a Dataverse, GCS or Hugging Face entry."""

    if dataset_file.backend == "dataverse":
        response = requests.get(
            f"{dataset_file.url.rstrip('/')}/api/datasets/:persistentId/",
            params={"persistentId": dataset_file.persistent_id}, timeout=timeout,
        )
        response.raise_for_status()
        prefix = dataset_file.path_prefix.strip("/")
        return [
            (_relative(e, prefix), e["dataFile"].get("filesize"))
            for e in response.json()["data"]["latestVersion"]["files"]
            if e.get("directoryLabel", "") == prefix or e.get("directoryLabel", "").startswith(prefix + "/")
        ]
    if dataset_file.backend == "gcs":
        bucket, _, prefix = dataset_file.url[5:].partition("/")
        prefix = prefix.rstrip("/") + "/"
        found: list[tuple[str, int | None]] = []
        token = None
        while True:
            params = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
            if token:
                params["pageToken"] = token
            response = requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o",
                                    params=params, timeout=timeout)
            response.raise_for_status()
            page = response.json()
            found += [(i["name"][len(prefix):], int(i["size"])) for i in page.get("items", [])
                      if not i["name"].endswith("/")]
            token = page.get("nextPageToken")
            if not token:
                return found
    if dataset_file.backend == "huggingface":
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise MissingDependencyError("huggingface-hub", extra="huggingface",
                                         feature="Hugging Face downloads") from error
        allow = dataset_file.allow_patterns
        allow = [allow] if isinstance(allow, str) else (allow or ["*"])
        tree = HfApi().list_repo_tree(dataset_file.repo_id, repo_type=dataset_file.repo_type or "dataset",
                                      revision=dataset_file.revision, recursive=True)
        return [(x.path, x.size) for x in tree
                if getattr(x, "size", None) is not None and any(fnmatch.fnmatch(x.path, p) for p in allow)]
    raise ValueError(f"{dataset_file.backend} is not a folder backend")


def remote_files(name: str, timeout: float = DEFAULT_DOWNLOAD_TIMEOUT) -> list[tuple[str, int | None]]:
    """Every file of a dataset at its source, as (name, size in bytes or None).

    For Dataverse, Google Cloud and Hugging Face this asks the source; use the
    names (or patterns over them) with ``files=`` to download only part.
    """

    dataset = get_dataset(name)
    listed: list[tuple[str, int | None]] = []
    for dataset_file in dataset.files:
        if dataset_file.backend in FOLDER_BACKENDS:
            listed += _remote_folder_files(dataset_file, timeout)
        else:
            listed.append((dataset_file.filename or dataset_file.url or "", None))
    return listed


def _select_files(dataset: DatasetSpec, patterns: str | Sequence[str] | None) -> list[DatasetFile]:
    """The files to download: all, or those whose name matches ``patterns``."""

    if patterns is None:
        return list(dataset.files)
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    if any(f.backend in FOLDER_BACKENDS for f in dataset.files):
        return list(dataset.files)       # filtered by the backend itself, against the source's listing
    chosen = {f.filename for f in dataset.files if any(fnmatch.fnmatch(f.filename, p) for p in patterns)}
    if not chosen:
        names = ", ".join(f.filename for f in dataset.files)
        raise ValueError(f"files={patterns!r} matches none of the files of {dataset.name}: {names}.")
    for f in dataset.files:                       # a split archive is only usable complete
        if f.filename in chosen and f.archive in _PART_KINDS:
            chosen.update(part.filename for part in _parts_group(dataset, f)[1])
    return [f for f in dataset.files if f.filename in chosen]


def _estimated_bytes(dataset: DatasetSpec, selected: list[DatasetFile]) -> int | None:
    """Download size of the selected files (proportional when only some are chosen)."""

    if dataset.size_bytes is None:
        return None
    return round(dataset.size_bytes * len(selected) / max(len(dataset.files), 1))


def _record_path(directory: Path, index: int, patterns: Sequence[str] | None) -> Path:
    """Completion record of a folder download: one for everything, one per file selection."""

    if not patterns:
        return directory / f".mlsca-complete-{index}.json"
    tag = hashlib.sha1("\n".join(sorted(patterns)).encode()).hexdigest()[:12]
    return directory / f".mlsca-complete-{index}-{tag}.json"


def _folder_complete(directory: Path, index: int, patterns: Sequence[str] | None) -> bool:
    """Everything, or at least this selection, was downloaded and verified before."""

    if _complete_record_ok(directory, _record_path(directory, index, None)):
        return True
    return bool(patterns) and _complete_record_ok(directory, _record_path(directory, index, patterns))


def _folder_estimate(dataset: DatasetSpec, patterns: Sequence[str], directory: Path) -> int:
    """Size of the selected files, from the source's listing (not yet on disk)."""

    total = matched = 0
    for index, dataset_file in enumerate(dataset.files):
        if dataset_file.backend not in FOLDER_BACKENDS or _folder_complete(directory, index, patterns):
            continue
        for name, size in _remote_folder_files(dataset_file):
            if _matches(name.rsplit("/", 1)[-1], name, patterns):
                matched += 1
                total += size or 0
    if not matched and not any(_folder_complete(directory, i, patterns) for i in range(len(dataset.files))):
        raise ValueError(
            f"files={list(patterns)!r} matches no file of {dataset.name}; "
            f"list them with `mlsca-bench info {dataset.name} --files`."
        )
    return total


def download_estimate(
    name: str, files: str | Sequence[str] | None = None, destination: str | os.PathLike[str] | None = None
) -> int | None:
    """Bytes a download of ``name`` (optionally only ``files``) will fetch, when known.

    For a selection on Dataverse, Google Cloud or Hugging Face this asks the
    source for its file sizes; otherwise it uses the registered size.
    """

    dataset = get_dataset(name)
    selected = _select_files(dataset, files)
    if files is not None and any(f.backend in FOLDER_BACKENDS for f in selected):
        patterns = [files] if isinstance(files, str) else list(files)
        return _folder_estimate(dataset, patterns, cache_root(destination) / dataset.name)
    return _estimated_bytes(dataset, selected)


def _already_there(dataset: DatasetSpec, dataset_file: DatasetFile, directory: Path) -> bool:
    """Cheap test that a file no longer needs downloading (no hashing, no folder walk)."""

    if dataset_file.archive in _PART_KINDS:
        base = _parts_group(dataset, dataset_file)[0]
        return (directory / (base + ".extracted" ) / EXTRACTED_RECORD).is_file()
    if dataset_file.filename:
        return (directory / dataset_file.filename).is_file()
    return any(directory.glob(".mlsca-complete-*.json"))


def _check_free_space(
    dataset: DatasetSpec, selected: list[DatasetFile], estimate: int | None, directory: Path,
    patterns: Sequence[str] | None = None,
) -> None:
    """Refuse to start a download that clearly cannot fit on the disk."""

    missing = [
        f for i, f in enumerate(dataset.files) if f in selected and not (
            _folder_complete(directory, i, patterns) if f.backend in FOLDER_BACKENDS
            else _already_there(dataset, f, directory))
    ]
    if not estimate or not missing:
        return
    whole = patterns is None and len(selected) == len(dataset.files)
    if whole and dataset.disk_bytes:
        # Measured size once unpacked, plus the parts and the joined archive that
        # exist for a moment while a split archive is put together.
        joining = 2 * (dataset.size_bytes or 0) if any(f.archive in _PART_KINDS for f in missing) else 0
        needed = round((dataset.disk_bytes + joining) * len(missing) / len(selected))
        unpack = 2 if dataset.disk_bytes > (dataset.size_bytes or 0) else 1
    else:
        unpack = max((3 if f.archive in _PART_KINDS else 2 if f.archive else 1) for f in missing)
        needed = round(estimate * len(missing) / len(selected)) * unpack
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed > free:
        extra = " (including room to unpack the archives)" if unpack > 1 else ""
        raise InsufficientDiskSpace(
            f"{dataset.name} needs at least {format_size(needed)}{extra}, but only "
            f"{format_size(free)} is free at {probe}. Download to a bigger disk with "
            f"destination= or the MLSCA_BENCH_DATA environment variable, download only some "
            f"files with files=, or pass check_space=False (command line: --no-space-check) to try anyway."
        )


def download_dataset(
    name: str,
    destination: str | os.PathLike[str] | None = None,
    *,
    progress: bool = True,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    files: str | Sequence[str] | None = None,
    check_space: bool = True,
) -> tuple[Path, ...]:
    """Download, verify, cache, and optionally extract a registered dataset.

    Parameters
    ----------
    name:
        Canonical dataset name or registered alias.
    destination:
        Parent cache directory. The dataset is stored in a canonical-name
        subdirectory. If omitted, ``MLSCA_BENCH_DATA`` or the operating-system
        cache directory is used.
    progress:
        Whether Pooch should display a download progress bar.
    timeout:
        Maximum time in seconds for HTTP connection and read operations.
    retries:
        Number of retries after transient HTTP connection failures.
    retry_backoff:
        Initial retry delay in seconds; it doubles after each failure.
    files:
        Only download the files whose names match these patterns (e.g.
        ``"*-vk0*"``); see ``mlsca-bench info NAME`` for a dataset's files.
        The parts of a split archive are always downloaded together.
    check_space:
        Refuse to start when the disk clearly has too little free space.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Downloaded files, or extracted files when an archive processor is used.

    Notes
    -----
    Pooch downloads to a temporary file, validates ``known_hash``, and only then
    moves the file into the cache. Existing valid files are reused.
    """

    dataset = get_dataset(name)
    _assert_automatic(dataset)
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if retry_backoff < 0:
        raise ValueError("retry_backoff must be non-negative")

    pooch = _load_pooch()
    downloader = pooch.HTTPDownloader(
        progressbar=progress,
        timeout=timeout,
    )
    dataset_directory = cache_root(destination) / dataset.name
    _remove_stale_partials(dataset_directory)
    selected = _select_files(dataset, files)
    patterns = None if files is None else ([files] if isinstance(files, str) else list(files))
    estimate = download_estimate(dataset.name, files, destination)
    if check_space:
        _check_free_space(dataset, selected, estimate, dataset_directory, patterns)
    if progress and not _looks_downloaded(dataset_directory):
        what = "" if len(selected) == len(dataset.files) else f", {len(selected)} of {len(dataset.files)} files"
        print(
            f"Downloading {dataset.display_name} ({dataset.name}{what}): about "
            f"{format_size(estimate)} into {dataset_directory}",
            flush=True,
        )
    downloaded: list[Path] = []

    for index, dataset_file in enumerate(dataset.files):
        if dataset_file not in selected:
            continue
        try:

            if dataset_file.backend == "http" and dataset_file.archive in _PART_KINDS:

                base, parts = _parts_group(dataset, dataset_file)
                if dataset_file is not parts[0]:
                    continue                      # the whole group is handled with its first part
                extracted = _extracted_files(dataset_directory / (base + ".extracted"))
                if extracted is None:
                    part_paths = []
                    for part in parts:
                        part_paths.extend(_as_paths(_retrieve_with_retries(
                            pooch,
                            retries=retries,
                            retry_backoff=retry_backoff,
                            url=part.url,
                            known_hash=part.known_hash,
                            fname=part.filename,
                            path=dataset_directory,
                            downloader=downloader,
                            progressbar=progress,
                        )))
                    extracted = _join_and_extract(
                        dataset_directory, base, _PART_KINDS[dataset_file.archive], part_paths
                    )
                downloaded.extend(extracted)

            elif dataset_file.backend == "http":

                result = _retrieve_with_retries(
                    pooch,
                    retries=retries,
                    retry_backoff=retry_backoff,
                    url=dataset_file.url,
                    known_hash=dataset_file.known_hash,
                    fname=dataset_file.filename,
                    path=dataset_directory,
                    processor=_processor_for(dataset_file, pooch),
                    downloader=downloader,
                    progressbar=progress,
                )

                downloaded.extend(_as_paths(result))

            elif dataset_file.backend == "gdrive":

                output = dataset_directory / dataset_file.filename

                # Reuse a complete, valid cached download.
                if output.exists() and hash_matches(
                    output,
                    dataset_file.known_hash,
                    strict=False,
                ):
                    logger.info("Using cached file %s", output)
                    downloaded_file = output
                    action = "fetch"          # archive processors then reuse what was extracted

                else:
                    downloaded_file = _download_google_drive(
                        url=dataset_file.url,
                        output=output,
                        progress=progress,
                    )

                    hash_matches(
                        downloaded_file,
                        dataset_file.known_hash,
                        strict=True,
                    )
                    action = "download"

                processor = _processor_for(dataset_file, pooch)

                if processor is None:
                    downloaded.append(downloaded_file)
                else:
                    downloaded.extend(
                        _as_paths(
                            processor(
                                str(downloaded_file),
                                action,
                                None,
                            )
                        )
                    )

            elif dataset_file.backend in ("gcs", "dataverse"):

                if _folder_complete(dataset_directory, index, patterns):
                    downloaded.append(dataset_directory)   # verified earlier; no network, no re-hashing
                    continue
                if dataset_file.backend == "gcs":
                    _download_gcs(url=dataset_file.url, output=dataset_directory, patterns=patterns)
                else:
                    _download_dataverse(
                        base_url=dataset_file.url,
                        persistent_id=dataset_file.persistent_id,
                        path_prefix=dataset_file.path_prefix,
                        output=dataset_directory,
                        timeout=timeout,
                        patterns=patterns,
                    )
                _write_complete_record(dataset_directory, _record_path(dataset_directory, index, patterns))
                downloaded.append(dataset_directory)
            elif dataset_file.backend == "huggingface":

                downloaded.append(
                    _download_huggingface(
                        dataset_file,
                        dataset_directory,
                        patterns=patterns,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported download backend: {dataset_file.backend!r}"
                )

        except (requests.exceptions.RequestException, TimeoutError, ValueError) as error:
            friendly = _explain_download_error(dataset, dataset_file, error)
            if friendly is None:
                raise
            raise friendly from error

    return tuple(downloaded)


__all__ = [
    "DATA_DIRECTORY_ENV",
    "DEFAULT_DOWNLOAD_RETRIES",
    "DEFAULT_DOWNLOAD_TIMEOUT",
    "DEFAULT_RETRY_BACKOFF",
    "cache_root",
    "download_dataset",
    "download_estimate",
    "remote_files",
]
