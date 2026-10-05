## What this changes

<!-- One or two sentences. -->

## Checklist

- [ ] `pytest -q` passes

**If this adds or changes a dataset** (see [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-public-dataset)):

- [ ] Downloaded with `mlsca-bench download NAME`; every HTTP file has a `known_hash`
- [ ] Trace length, number of traces and keys checked against the files, not only the paper
- [ ] License recorded as the source states it (`license` and `license_source`, or `"not stated"`)
- [ ] Every split loaded with `load_dataset(NAME, split=...)`; fields checked against the dataset's documentation (describe below)
- [ ] `python scripts/check_links.py` runs

**What I checked:**

