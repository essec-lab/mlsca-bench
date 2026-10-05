# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Shared lazy views used by HDF5 dataset adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class OpenDataset(Protocol):
    def _ensure_open(self) -> None: ...


class CompoundFieldView:
    """Lazy array view of one field in an HDF5 compound dataset."""

    def __init__(
        self,
        owner: OpenDataset,
        records: Any,
        field_name: str,
    ) -> None:
        self._owner = owner
        self._records = records
        self._field_name = field_name

        field_dtype = records.dtype.fields[field_name][0]
        if field_dtype.subdtype is None:
            self._dtype = field_dtype
            field_shape: tuple[int, ...] = ()
        else:
            self._dtype, field_shape = field_dtype.subdtype
        self._shape = (len(records), *field_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._dtype)

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index: Any) -> Any:
        self._owner._ensure_open()
        records = self._records[index]
        return np.asarray(records[self._field_name])


__all__ = ["CompoundFieldView"]
