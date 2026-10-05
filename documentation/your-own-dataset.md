# Using your own dataset

Register your traces once, then use the name everywhere a built-in dataset name works: `load_dataset`, `run_attack` and `run_classical`. Nothing in the installed package has to change.

```python
from mlsca_bench import register_dataset, load_dataset
from mlsca_bench.models import run_attack

register_dataset(
    "my-lab-aes",
    {
        "adapter": "hdf5",
        "adapter_config": {
            "splits": {"all": "/"},
            "traces_dataset": "traces",
            "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]},
        },
        "metadata": {
            "format": "hdf5",
            "algorithm": "AES-128",
            "measurement": "power",
            "platform": "arm-cortex-m",
            "countermeasures": ["none"],
            "key": "fixed",
        },
    },
    path="/data/my_traces.h5",          # already on disk: nothing is downloaded
    save=True,                          # remember it in every future session
)

with load_dataset("my-lab-aes") as ds:
    print(ds.shape)

result = run_attack("my-lab-aes", model="mlp", epochs=10)
```

## The entry

- **`adapter`** (required): the reader for your file format, see below.
- **`adapter_config`**: where the reader finds the traces and the other fields in your files.
- **`metadata`**: `format`, `algorithm` and `measurement` are required. `platform`, `device`, `countermeasures`, `key` (`"fixed"` or `"variable"`), `n_samples` and `year` are optional.
- **`path`**: your file or folder.

The entry is checked when you register it. A misspelled reader or field is rejected with a message that names the expected values, and a field path that does not exist in your file is reported with the paths that do exist.

## Readers

| `adapter` | Files | Main options |
|---|---|---|
| `hdf5` | one HDF5 file | `traces_dataset`; `field_aliases` (paths of plaintexts, ciphertexts, keys, masks); `labels_dataset`; `splits` (group per split, e.g. `{"profiling": "Profiling_traces", "attack": "Attack_traces"}`) |
| `npy` | NumPy `.npy` files | `layout`: file name per field, e.g. `{"traces": "traces.npy", "plaintexts": "plaintexts.npy", "keys": "keys.npy"}` |
| `npz` | NumPy `.npz` archives | `layout`: array name per field inside the archive |
| `matlab` | MATLAB `.mat` | `fields`: variable name per field |
| `trs` | Riscure `.trs` | `data_fields`: byte range per field in each trace's data, e.g. `{"plaintexts": [0, 16], "keys": [16, 32]}` |
| `raw-binary` | raw sample files | `record`: the binary layout of one trace |

Many files of the same kind (one per chunk of traces) are read as one dataset with `"chunked": true` and `"chunk_glob": "*.h5"`. The 64 built-in entries in [`registry.json`](../src/mlsca_bench/datasets/registry.json) cover every reader and are good templates: `mlsca-bench info NAME` shows which reader a dataset uses.

## Splits

Name the splits your files contain in `splits`. With a single split, nothing else is needed. With several, `default_split` chooses the one used when none is given. If your files have no profiling/attack split, a seeded 70 / 15 / 15 train / validation / test split is made, the same for everyone.

## Keeping it

- **`save=True`** stores the entry in your personal file (`mlsca_bench.user_registry_path()`, on macOS `~/Library/Application Support/mlsca-bench/datasets.json`), loaded automatically on every import. `saved_datasets()` lists them; `mlsca-bench forget my-lab-aes` removes one (your files are kept). Without `save=True` a registration lasts for the current Python session.
- **A file of your own**, for example to share with your group: it maps names to entries in the same format, with the location as `"path"` (a relative path is resolved against the file).

  ```json
  {
    "my-lab-aes": {
      "adapter": "hdf5",
      "adapter_config": {"splits": {"all": "/"}, "traces_dataset": "traces",
                         "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]}},
      "metadata": {"format": "hdf5", "algorithm": "AES-128", "measurement": "power"},
      "path": "/data/my_traces.h5"
    }
  }
  ```

  Load it with `load_registry_file("my_datasets.json")`, save its entries for every session with `mlsca-bench add my_datasets.json`, or load it automatically with `export MLSCA_BENCH_REGISTRY=/path/to/my_datasets.json` (several files separated by `:`, `;` on Windows).

Built-in names cannot be replaced. To make your dataset downloadable for others, list its files (with checksums) instead of a `path`, and consider [contributing it](../CONTRIBUTING.md).
