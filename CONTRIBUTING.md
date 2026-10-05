# Contributing to MLSCA-Bench

Thank you for helping. The most useful contributions are new public datasets, new models and bug reports, especially broken download links.

## Development setup

```bash
git clone https://github.com/essec-lab/mlsca-bench
cd mlsca-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"          # add ",eval" to also run the deep-learning tests
pytest -q
```

The deep-learning tests skip themselves when PyTorch is not installed. CI runs the full suite on Python 3.10 to 3.14, once without and once with PyTorch, and checks that the package builds.

## Adding a public dataset

Only datasets that anyone can download without a login or a request to the authors are added to the built-in registry. For private or unreleased data, use `register_dataset` locally instead (see [using your own dataset](documentation/your-own-dataset.md)).

1. **Add an entry** to `src/mlsca_bench/datasets/registry.json`. Copy a similar existing entry and adapt it:
   - `files`: the download URLs, with a checksum for every HTTP or Google Drive file (`"known_hash": "sha256:..."`; compute it with `shasum -a 256 <file>`). Hugging Face files need a pinned `revision`.
   - `adapter` and `adapter_config`: the reader for the file format. Reuse an existing reader whenever possible; a new format needs a new adapter in `src/mlsca_bench/datasets/adapters/`.
   - `metadata`: format, algorithm, measurement, platform, device, countermeasures, key regime, trace length and year. Use the controlled vocabularies in `registry.py`.
2. **Check every metadata value against the downloaded files, not only against the paper or README.** Papers often describe a different version of the data. Read the trace length, the number of traces and the keys from the files themselves; mention the source of the remaining values (platform, device, countermeasures, year) in the pull request. Where the paper and the data disagree, follow the data and add a note. Record the license exactly as the source states it, with `license_source`, or `"not stated"`.
3. **Download it and load it** (required): `mlsca-bench download NAME`, then load every split with `load_dataset(NAME, split=...)` and check that the traces, plaintexts, keys and other fields are what the dataset's documentation says. Describe what you checked in the pull request; the maintainers check new datasets on the real files before merging. For datasets too large to download completely, download one file (`--files PATTERN`) and say so.
4. **Add a test** in `tests/` that loads a small real or synthetic file with the same layout.
5. **Check the download links:** `python scripts/check_links.py` flags broken links, Git LFS pointer files and HTML pages served instead of data. It also runs every week in CI and opens an issue when a link breaks.
6. **Record the fields you saw** (`"fields"`: traces, plaintexts, ciphertexts, keys, masks, labels, per split if they differ), from `ds.available_fields`.

## Adding a model

Register a builder by name. It receives the trace length and the number of classes and returns a `torch.nn.Module`:

```python
from mlsca_bench.models.registry import register_model

@register_model("my_cnn")
def _my_cnn(*, input_length: int, n_classes: int):
    return MyCNN(input_length, n_classes)
```

It then works like a baseline: `run_attack(..., model="my_cnn")`. For a model from a paper, cite it in the docstring and say whether it has been checked against the paper's published results. Only add code whose license allows it to be redistributed.

## Reporting a broken link or wrong metadata

Open an issue with the dataset name, what is wrong and, if possible, the correct source. These reports keep the benchmark usable.

## Pull requests

- Keep each pull request focused on one change.
- Run `pytest -q` before opening it.
- Describe what you verified and how.
