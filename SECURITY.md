# Security notes

## Pickle datasets require explicit opt-in (`trust_pickle=True`)

A few side-channel datasets are distributed as **Python pickle files**
(currently **GALACTICS / BLISS**, whose traces live in pandas-pickled
DataFrames).

**Unpickling executes arbitrary code.** A pickle is not just data: it can carry
instructions that run the moment the file is opened (via `__reduce__`). A
malicious or tampered pickle can therefore run any command on your machine
during loading. This is a property of Python's `pickle` format, not of this
library.

### How mlsca-bench protects you

- **Off by default.** Loading a pickle-backed dataset raises
  `PickleTrustError` unless you pass `trust_pickle=True`:

  ```python
  from mlsca_bench import load_dataset

  load_dataset("galactics")                      # -> PickleTrustError
  load_dataset("galactics", trust_pickle=True)   # opt in, at your own risk
  ```

- **Never enabled implicitly.** The shipped registry never sets `trust_pickle`.
  There is no config value or environment default that turns it on for you; the
  choice is made per call, by you.

- **Integrity checked on download.** Automatic downloads verify the archive
  against the hash pinned in the registry, so the file you unpickle is the exact
  artifact published by the dataset authors (it does not, and cannot, prove that
  artifact is itself benign).

- **Loud at load time.** When you do opt in, the loader emits a `UserWarning`
  naming the file it is about to unpickle.

### What you should do

Only pass `trust_pickle=True` for datasets whose source you trust (e.g. the
official Zenodo record, downloaded over the verified hash). If you obtained the
file from an untrusted place, do **not** enable it. When in doubt, inspect the
pickle in a sandbox/VM first, or ask the dataset authors for a non-pickle
export.

## Other formats

All other adapters avoid code execution: NumPy files are loaded with
`allow_pickle=False`, and HDF5/npz/MATLAB/TFRecord/`.trc`/`.trs`/Agilent/ASCII
readers parse data only. The TFRecord reader (SCAAML) is a self-contained
decoder and does **not** pull in TensorFlow.
