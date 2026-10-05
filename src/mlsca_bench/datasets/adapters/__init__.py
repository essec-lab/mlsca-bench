# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Format-specific adapters mapping datasets onto the shared interface."""

from ._concat import ConcatDataset
from .aes_hd_csv import AESHDCSVDataset
from .agilent import AgilentWaveDataset
from .ascad import ASCADDataset
from .asciiwave import AsciiWaveDataset
from .ascon import AsconHDF5Dataset
from .chameleon import ChameleonDataset
from .hdf5 import FlatHDF5Dataset, HDF5CompoundDataset
from .lecroy import LeCroyIndexDataset
from .manifest import ManifestNpyDataset
from .npy import AESHDZaidDataset, NpyDirectoryDataset, NpzArchiveDataset
from .pickled import PickleDataset, PickleTrustError
from .rawbinary import RawBinaryDataset
from .trs import TRSDataset
from .two_class import TwoClassNpyDataset

__all__ = [
    "AESHDCSVDataset",
    "AESHDZaidDataset",
    "AgilentWaveDataset",
    "ASCADDataset",
    "AsciiWaveDataset",
    "AsconHDF5Dataset",
    "ChameleonDataset",
    "ConcatDataset",
    "FlatHDF5Dataset",
    "HDF5CompoundDataset",
    "LeCroyIndexDataset",
    "ManifestNpyDataset",
    "NpyDirectoryDataset",
    "NpzArchiveDataset",
    "PickleDataset",
    "PickleTrustError",
    "RawBinaryDataset",
    "TRSDataset",
    "TwoClassNpyDataset",
]
