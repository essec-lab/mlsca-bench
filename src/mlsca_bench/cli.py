# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""The ``mlsca-bench`` command: browse, download and manage datasets from a terminal."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .datasets import (
    DatasetDownloadError,
    MissingDependencyError,
    cache_root,
    cache_usage,
    cached_datasets,
    download_dataset,
    get_dataset,
    list_datasets,
    load_registry_file,
    remove_cached_dataset,
    saved_datasets,
    unregister_dataset,
    user_registry_path,
)
from .datasets.registry import (
    DatasetRegistryError,
    format_size,
    is_user_dataset,
    local_dataset_path,
)


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    keep = [i for i in range(len(header)) if header[i] or any(row[i] for row in rows)]
    header = tuple(header[i] for i in keep)
    rows = [tuple(row[i] for i in keep) for row in rows]
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip()
             for row in [header, *rows]]
    lines.insert(1, "  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)


_SHORT = {"plaintexts": "pt", "ciphertexts": "ct", "keys": "k", "masks": "m", "labels": "l"}


def _provides_short(spec) -> str:
    if not spec.fields:
        return "?"
    return " ".join(_SHORT[f] for f in _SHORT if f in spec.provides) or "traces only"


def _fields_text(spec) -> str:
    if not spec.fields:
        return "not checked on real data yet"
    groups = {}
    for split, names in spec.fields.items():
        groups.setdefault(names, []).append(split)
    parts = [", ".join(names) if splits == ["all"] else f"{'/'.join(splits)}: {', '.join(names)}"
             for names, splits in groups.items()]
    return "; ".join(parts) + " (seen in the real files)"


def _matches(value: str | None, wanted: str | None) -> bool:
    return wanted is None or (value is not None and wanted.lower() in value.lower())


def _cmd_list(args: argparse.Namespace) -> int:
    specs = [s for s in list_datasets(builtin_only=args.builtin) if not args.mine or is_user_dataset(s.name)]
    rows = []
    for spec in specs:
        meta = spec.metadata
        if not (_matches(meta and meta.algorithm, args.algorithm)
                and _matches(meta and meta.measurement, args.measurement)
                and _matches(meta and meta.platform, args.platform)):
            continue
        if args.unprotected and (meta is None or meta.countermeasures):
            continue
        if args.with_fields and not set(args.with_fields) <= set(spec.provides):
            continue
        rows.append((
            spec.name,
            meta.algorithm if meta else "",
            meta.measurement if meta else "",
            meta.platform or "" if meta else "",
            ", ".join(meta.countermeasures) or "none" if meta else "",
            spec.size if spec.size_bytes else "",
            _provides_short(spec),
            "yours" if is_user_dataset(spec.name) else "",
        ))
    if not rows:
        print("No datasets match.")
        return 0
    print(_table(rows, ("name", "algorithm", "measurement", "platform", "countermeasures", "size", "fields", "")))
    print(f"\n{len(rows)} dataset(s). Details: mlsca-bench info NAME")
    print("fields: pt plaintexts, ct ciphertexts, k keys, m masks, l labels (besides traces); ? = not checked on real data yet")
    return 0


def _print_remote_files(spec, show_all: bool) -> None:
    from collections import defaultdict

    from .datasets.download import remote_files

    listed = remote_files(spec.name)
    total = sum(size or 0 for _, size in listed)
    print(f"{spec.name}: {len(listed)} files" + (f", {format_size(total)}" if total else ""))
    if len(listed) > 40 and not show_all:
        folders: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for name, size in listed:
            folder = name.rsplit("/", 1)[0] + "/" if "/" in name else "(top level)"
            folders[folder][0] += 1
            folders[folder][1] += size or 0
        print(_table([(f, str(n), format_size(b) if b else "") for f, (n, b) in sorted(folders.items())],
                     ("folder", "files", "size")))
        print(f"\nFirst files: {', '.join(n for n, _ in listed[:5])}, ...  (all of them: --files --all)")
    else:
        print(_table([(n, format_size(sz) if sz else "") for n, sz in listed], ("file", "size")))
    split_files = dict((spec.adapter_config or {}).get("split_files") or {})
    if split_files:
        print("\nPer split: " + "; ".join(f"--split {k} = {', '.join(v)}" for k, v in split_files.items()))
    print(f"Download only some with: mlsca-bench download {spec.name} --files PATTERN")


def _cmd_info(args: argparse.Namespace) -> int:
    spec = get_dataset(args.name)
    if args.files:
        _print_remote_files(spec, args.all)
        return 0
    meta = spec.metadata
    fields: list[tuple[str, object]] = [("name", spec.name), ("title", spec.display_name)]
    if spec.aliases:
        fields.append(("aliases", ", ".join(spec.aliases)))
    if meta is not None:
        fields += [
            ("algorithm", meta.algorithm),
            ("measurement", meta.measurement),
            ("platform", meta.platform),
            ("device", meta.device),
            ("countermeasures", ", ".join(meta.countermeasures) or "none"),
            ("key", meta.key),
            ("samples per trace", meta.n_samples),
            ("file format", meta.format),
            ("year", meta.year),
        ]
    fields += [
        ("fields", _fields_text(spec)),
        ("download size", spec.size if spec.size_bytes else None),
        ("space needed", format_size(spec.disk_bytes) + " once unpacked" if spec.disk_bytes else None),
        ("availability", spec.availability),
        ("homepage", spec.homepage),
        ("paper", spec.paper),
        ("license", spec.license),
        ("license source", spec.license_source),
        ("notes", spec.notes),
    ]
    named = [f.filename for f in spec.files if f.filename]
    if named:
        shown = ", ".join(named[:6]) + (f", ... ({len(named)} files)" if len(named) > 6 else "")
        fields.append(("files", shown))
    elif spec.files:
        fields.append(("files", f"hosted on {spec.files[0].backend}; list them with --files"))
    split_files = dict((spec.adapter_config or {}).get("split_files") or {})
    if split_files:
        fields.append(("per-split download", ", ".join(f"--split {k}" for k in split_files)))
    if is_user_dataset(spec.name):
        saved = spec.name in saved_datasets()
        fields.append(("source", "your dataset" + (" (saved)" if saved else " (this session)")))
        fields.append(("local path", local_dataset_path(spec.name)))
    cached = {entry.name: entry for entry in cached_datasets()}
    if spec.name in cached:
        fields.append(("in cache", f"{cached[spec.name].path} ({cached[spec.name].size})"))
    elif spec.files:
        fields.append(("in cache", "no (mlsca-bench download " + spec.name + ")"))
    width = max(len(key) for key, value in fields if value not in (None, ""))
    for key, value in fields:
        if value not in (None, ""):
            print(f"{key.ljust(width)}  {value}")
    return 0


CONFIRM_ABOVE_BYTES = 50 * 10**9


def _files_for(spec, args: argparse.Namespace) -> list[str] | None:
    if args.files:
        return args.files
    if args.split:
        split_files = dict((spec.adapter_config or {}).get("split_files") or {})
        if args.split not in split_files:
            raise ValueError(
                f"{spec.name} has no per-split files; download it whole or pick files with --files."
            )
        return list(split_files[args.split])
    return None


def _cmd_download(args: argparse.Namespace) -> int:
    from .datasets.download import download_estimate

    for name in args.names:
        spec = get_dataset(name)
        files = _files_for(spec, args)
        estimate = download_estimate(spec.name, files, args.to)
        if (estimate or 0) > CONFIRM_ABOVE_BYTES and not args.yes and sys.stdin.isatty():
            answer = input(f"{spec.name} is about {format_size(estimate)}. Download it? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print(f"Skipped {spec.name}.")
                continue
        download_dataset(name, destination=args.to, progress=not args.quiet, files=files,
                         check_space=not args.no_space_check)
        canonical = spec.name
        folder = cache_root(args.to) / canonical
        print(f"{canonical}: ready in {folder}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    names = load_registry_file(args.file, overwrite=args.overwrite, save=True)
    print(f"Saved {len(names)} dataset(s) to {user_registry_path()}: {', '.join(names)}")
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    for name in args.names:
        canonical = get_dataset(name).name
        unregister_dataset(canonical, forget=True)
        print(f"Forgot {canonical}. Its files were not deleted.")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.cache_command == "remove":
        for name in args.names:
            freed = remove_cached_dataset(name, destination=args.dir)
            print(f"Removed {name} from the cache, freed {format_size(freed)}.")
        return 0
    if args.cache_command == "path":
        print(cache_root(args.dir))
        return 0
    entries = cached_datasets(args.dir)
    print(f"Cache: {cache_root(args.dir)}")
    if not entries:
        print("Empty.")
        return 0
    rows = [(e.name, e.size, "" if e.known else "unknown folder") for e in entries]
    print(_table(rows, ("name", "size", "")))
    print(f"\nTotal: {format_size(cache_usage(args.dir))}. Remove one with: mlsca-bench cache remove NAME")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlsca-bench",
        description="Browse, download and manage side-channel datasets.",
    )
    parser.add_argument("--version", action="version", version=f"mlsca-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("list", help="list datasets, optionally filtered")
    p.add_argument("--algorithm", help="e.g. aes, ascon, kyber (substring match)")
    p.add_argument("--measurement", help="power or em")
    p.add_argument("--platform", help="e.g. stm32, fpga (substring match)")
    p.add_argument("--unprotected", action="store_true", help="only datasets without countermeasures")
    p.add_argument("--with", dest="with_fields", nargs="+", metavar="FIELD",
                   choices=["plaintexts", "ciphertexts", "keys", "masks", "labels"],
                   help="only datasets that provide these fields (checked on the real files)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--mine", action="store_true", help="only your own datasets")
    group.add_argument("--builtin", action="store_true", help="only the built-in datasets")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("info", help="show everything known about one dataset")
    p.add_argument("name")
    p.add_argument("--files", action="store_true", help="list the dataset's files at the source, with sizes")
    p.add_argument("--all", action="store_true", help="with --files: list every file, not a summary")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("download", help="download one or more datasets to the cache")
    p.add_argument("names", nargs="+", metavar="NAME")
    p.add_argument("--to", metavar="DIR", help="download here instead of the cache")
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    p.add_argument("--files", nargs="+", metavar="PATTERN", help="only files matching these patterns (see info)")
    p.add_argument("--split", help="only the files of this split, where a dataset ships splits as separate files")
    p.add_argument("--no-space-check", action="store_true", help="start even if the disk looks too small")
    p.add_argument("--yes", "-y", action="store_true", help=f"do not ask before downloads over {CONFIRM_ABOVE_BYTES // 10**9} GB")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("add", help="save the datasets in a JSON registry file of your own")
    p.add_argument("file", metavar="FILE.json")
    p.add_argument("--overwrite", action="store_true", help="replace datasets with the same name")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("forget", help="remove your own saved datasets (files are kept)")
    p.add_argument("names", nargs="+", metavar="NAME")
    p.set_defaults(func=_cmd_forget)

    p = sub.add_parser("cache", help="see and free the space used by downloads")
    p.add_argument("--dir", metavar="DIR", help="a cache folder other than the default")
    cache_sub = p.add_subparsers(dest="cache_command", metavar="ACTION")
    cache_sub.add_parser("list", help="downloaded datasets and their size (default)")
    cache_sub.add_parser("path", help="print the cache folder")
    remove = cache_sub.add_parser("remove", help="delete downloaded datasets")
    remove.add_argument("names", nargs="+", metavar="NAME")
    p.set_defaults(func=_cmd_cache)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):     # e.g. a Windows console or redirected output
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetRegistryError, DatasetDownloadError, MissingDependencyError, FileNotFoundError, PermissionError, ValueError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"mlsca-bench: error: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
