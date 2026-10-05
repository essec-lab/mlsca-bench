# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the Riscure Inspector trace-set (``.trs``) format.

A ``.trs`` file begins with a tag-length-value header (number of traces, samples
per trace, sample coding, per-trace crypto-data length, and title length) that
ends with the trace-block tag ``0x5f``. Every trace record that follows is a
contiguous block of ``title | data | samples``, so the file is self-describing:
the sample matrix and the crypto-data block are read as zero-copy strided views
over the memory-mapped file. Because the meaning of the crypto-data bytes is
dataset-specific, a ``data_fields`` config maps byte ranges of that block onto
the standard plaintext/ciphertext/key/mask fields.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .npy import _ArrayFieldsDataset
from ..errors import describe_paths

# Header tags (Riscure Inspector trace-set specification).
_TAG_NUMBER_OF_TRACES = 0x41
_TAG_NUMBER_OF_SAMPLES = 0x42
_TAG_SAMPLE_CODING = 0x43
_TAG_DATA_LENGTH = 0x44
_TAG_TITLE_LENGTH = 0x45
_TAG_TRACE_BLOCK = 0x5F

_INTEGER_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}
_FLOAT_DTYPES = {4: np.float32, 8: np.float64}
_DATA_ALIASES = ("plaintexts", "ciphertexts", "keys", "masks", "labels")


def _first_trs_file(
    source: str | os.PathLike[str] | Iterable[Path],
    preferred_filename: str | None = None,
) -> Path:
    if isinstance(source, (str, os.PathLike)):
        candidates: tuple[Path, ...] = (Path(source).expanduser(),)
    else:
        candidates = tuple(Path(path).expanduser() for path in source)

    trs_files: list[Path] = []
    for path in candidates:
        if path.is_dir():
            trs_files.extend(item for item in path.rglob("*.trs") if item.is_file())
        elif path.is_file() and path.suffix.casefold() == ".trs":
            trs_files.append(path)
    if not trs_files:
        raise FileNotFoundError(f"No .trs file was found in {describe_paths(candidates)}.")

    if preferred_filename is not None:
        chosen = [p for p in trs_files if p.name == preferred_filename]
        if not chosen:
            raise FileNotFoundError(f"Requested .trs file {preferred_filename!r} was not found.")
        if len(chosen) > 1:
            raise RuntimeError(f"Multiple files named {preferred_filename} were found.")
        return chosen[0]

    if len(trs_files) > 1:
        raise RuntimeError(
            "Multiple .trs files were found; set preferred_filename to choose one."
        )
    return trs_files[0]


def _parse_header(buffer: np.memmap) -> tuple[dict[int, bytes], int]:
    """Return the raw tag values and the byte offset where traces begin."""

    fields: dict[int, bytes] = {}
    position = 0
    size = buffer.size
    while position < size:
        tag = int(buffer[position])
        position += 1
        length = int(buffer[position])
        position += 1
        if length & 0x80:
            length_bytes = length & 0x7F
            length = int.from_bytes(
                buffer[position : position + length_bytes].tobytes(), "little"
            )
            position += length_bytes
        value = buffer[position : position + length].tobytes()
        position += length
        if tag == _TAG_TRACE_BLOCK:
            return fields, position
        fields[tag] = value
    raise ValueError("Malformed .trs header: trace-block tag 0x5f was not found.")


class TRSDataset(_ArrayFieldsDataset):
    """Read a Riscure ``.trs`` trace set lazily via a memory-mapped view.

    ``data_fields`` maps a public field name to a ``[start, end]`` byte range
    within each trace's crypto-data block, e.g.
    ``{"plaintexts": [0, 16], "ciphertexts": [16, 32]}``.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | Iterable[Path],
        *,
        name: str,
        split: str | None = None,
        data_fields: Mapping[str, Sequence[int]] | None = None,
        preferred_filename: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        self._name = name.strip()
        self._split = split
        self._closed = False
        path = _first_trs_file(source, preferred_filename)
        self._mmap = np.memmap(path, dtype=np.uint8, mode="r")

        try:
            header, data_start = _parse_header(self._mmap)
            number_of_traces = self._header_int(header, _TAG_NUMBER_OF_TRACES)
            number_of_samples = self._header_int(header, _TAG_NUMBER_OF_SAMPLES)
            coding = header.get(_TAG_SAMPLE_CODING)
            if coding is None or not coding:
                raise ValueError("The .trs header is missing the sample coding tag.")
            sample_dtype, sample_size = self._sample_dtype(coding[0])
            data_length = self._header_int(
                header, _TAG_DATA_LENGTH, default=0
            )
            title_length = self._header_int(
                header, _TAG_TITLE_LENGTH, default=0
            )

            record_stride = title_length + data_length + number_of_samples * sample_size
            required = data_start + number_of_traces * record_stride
            if self._mmap.size < required:
                raise ValueError(
                    "The .trs file is truncated: header declares "
                    f"{number_of_traces} traces but the file is too small."
                )

            traces = np.ndarray(
                shape=(number_of_traces, number_of_samples),
                dtype=sample_dtype,
                buffer=self._mmap,
                offset=data_start + title_length + data_length,
                strides=(record_stride, sample_size),
            )
            self._arrays = {
                field: None
                for field in ("plaintexts", "ciphertexts", "keys", "masks", "labels")
            }
            self._arrays["traces"] = traces

            if data_fields:
                if data_length == 0:
                    raise ValueError(
                        "data_fields were given but the .trs header declares no "
                        "crypto-data block."
                    )
                data_block = np.ndarray(
                    shape=(number_of_traces, data_length),
                    dtype=np.uint8,
                    buffer=self._mmap,
                    offset=data_start + title_length,
                    strides=(record_stride, 1),
                )
                for field, byte_range in data_fields.items():
                    if field not in _DATA_ALIASES:
                        raise ValueError(
                            f"Unsupported .trs data field {field!r}; choose from "
                            f"{', '.join(_DATA_ALIASES)}."
                        )
                    start, end = int(byte_range[0]), int(byte_range[1])
                    if not 0 <= start < end <= data_length:
                        raise ValueError(
                            f"data_fields[{field!r}] range {list(byte_range)} is "
                            f"outside the {data_length}-byte crypto-data block."
                        )
                    self._arrays[field] = data_block[:, start:end]

            dataset_metadata = {
                "format": "trs",
                "path": str(path),
                "samples_per_trace": number_of_samples,
                "data_length": data_length,
            }
            dataset_metadata.update(metadata or {})
            self._metadata = MappingProxyType(dataset_metadata)
            self._finalize()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _header_int(
        header: Mapping[int, bytes], tag: int, *, default: int | None = None
    ) -> int:
        value = header.get(tag)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"The .trs header is missing required tag {tag:#x}.")
        return int.from_bytes(value, "little")

    @staticmethod
    def _sample_dtype(coding: int) -> tuple[np.dtype[Any], int]:
        length = coding & 0x0F
        is_float = bool(coding & 0x10)
        table = _FLOAT_DTYPES if is_float else _INTEGER_DTYPES
        if length not in table:
            kind = "float" if is_float else "integer"
            raise ValueError(
                f"Unsupported .trs {kind} sample length of {length} byte(s)."
            )
        return np.dtype(table[length]), length

    def _release(self) -> None:
        # Drop the memory map reference; the strided views still hold it until
        # the dataset is collected, avoiding use-after-free on escaped views.
        self._mmap = None  # type: ignore[assignment]


__all__ = ["TRSDataset"]
