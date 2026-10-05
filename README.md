# MLSCA-Bench

A toolkit for machine-learning side-channel analysis (ML-SCA): **64 public datasets** behind one interface, baseline attacks, and one definition of guessing entropy and success rate, so that results are comparable.

With MLSCA-Bench you can:

1. **download and load** any of the 64 datasets with one line, whatever their original format;
2. **run the baselines**: 4 deep-learning models (MLP, ASCAD MLP, ASCAD CNN and Zaid CNN) and 3 classical attacks (CPA, DPA, templates);
3. **compute guessing entropy and success rate** and plot them;
4. **add your own dataset** and use it like a built-in one;
5. **add your own model** and run it like a baseline.

It works on Linux, macOS and Windows with Python 3.10 or newer.

- [Installation](#installation)
- [1. Download and load a dataset](#1-download-and-load-a-dataset)
- [2. Run a baseline](#2-run-a-baseline)
- [3. Guessing entropy, success rate and plots](#3-guessing-entropy-success-rate-and-plots)
- [4. Add your own dataset](#4-add-your-own-dataset)
- [5. Add your own model](#5-add-your-own-model)
- [The datasets](#the-datasets)
- [Contributing](#contributing)
- [Contact](#contact)
- [License / copyright](#license--copyright)

## Installation

```bash
pip install "mlsca-bench[hdf5,eval,plot]"
```

`hdf5` reads HDF5 datasets (most of them), `eval` adds PyTorch for the deep-learning baselines, and `plot` adds matplotlib for the plots. `pip install "mlsca-bench[all]"` adds every dataset reader and download backend (everything except PyTorch). If a dataset needs an extra you have not installed, the error message names it.

A few datasets are archives that need a system tool: the six AES-PTv2 datasets are RAR archives, and a few zip files need 7-Zip. Install [7-Zip](https://www.7-zip.org) (Windows; it also extracts the RAR archives), `brew install unar sevenzip` (macOS) or `apt-get install unar p7zip-full` (Debian/Ubuntu).

**Windows:** the DPA Contest v1 to v3 datasets contain file paths longer than 260 characters, which Windows only allows once long paths are enabled. The package tells you how when it is needed.

**Intel Macs:** PyTorch's last version for Intel Macs is 2.2, which needs Python 3.12 or older and NumPy 1. Install with `pip install "mlsca-bench[hdf5,eval,plot]" "numpy<2"` (everything without the `eval` extra works with any version).

## 1. Download and load a dataset

```python
from mlsca_bench import load_dataset

with load_dataset("ascadf", split="attack") as ds:     # downloads once (4.4 GB), then cached
    print(ds.shape, ds.available_fields)               # (10000, 700) ['ciphertexts', 'keys', ...]
    traces = ds.traces[:1000]                          # read lazily: only these rows
    print(ds[0].plaintext, ds[0].key)
```

Find a dataset and check what it contains before downloading:

```bash
mlsca-bench list                          # all 64, with size and the fields they provide
mlsca-bench list --with plaintexts keys   # only datasets that include plaintexts and keys
mlsca-bench info ascadf                   # metadata, download size and space needed, license, fields
mlsca-bench download ascadf               # download without loading
mlsca-bench cache                         # what is downloaded, and how much space it uses
```

Every download is checked against a checksum, a damaged or interrupted download is repaired on the next call, the disk space is checked first, and a downloaded dataset loads without internet. Datasets are stored in your user cache folder; set `MLSCA_BENCH_DATA=/path/to/folder` to use another place. Already have the files? `load_dataset("ascadf", path="/data/ASCAD.h5")` uses them directly.

**Very large datasets** (17 are over 50 GB, up to 4 TB) can be downloaded in part:

```bash
mlsca-bench info ches-ctf-2020-spook-sw3 --files              # the files at the source, with sizes
mlsca-bench download ches-ctf-2020-spook-sw3 --split attack   # only the attack split (12.5 of 87.5 GB)
mlsca-bench download scaaml-ecc-gpam-cm0 --files info.json "train/0_*"   # only matching files
```

In Python, `download_dataset(name, files="pattern")` does the same, and `load_dataset(name, split="attack")` downloads only that split where the splits are separate files (SMAesH, the CHES CTF 2020 Spook sets, SCAAML). A partly downloaded dataset loads from the files that are present. The command line asks before downloads over 50 GB.

## 2. Run a baseline

Deep-learning baselines (the `eval` extra) are trained and attacked with one call. The dataset's own profiling/attack split is used where the authors provide one, otherwise a fixed, seeded 70 / 15 / 15 split:

```python
from mlsca_bench.benchmark import LeakageModel
from mlsca_bench.models import run_attack

leakage = LeakageModel(byte=2)            # ASCAD's traces cover key byte 2; the default is byte 0
mlp = run_attack("ascadf", model="mlp", leakage=leakage, epochs=10)
print(mlp.guessing_entropy[-1], mlp.ranks.traces_to_disclosure())
```

The classical attacks need no deep-learning install:

```python
from mlsca_bench.benchmark import run_classical

cpa = run_classical("cpa", "ascadf", leakage=leakage)
template = run_classical("template", "ascadf", leakage=leakage)
```

| Baseline | Name | Default training | Notes |
|---|---|---|---|
| MLP | `mlp` | Adam, lr 1e-3, 50 epochs, batch 128, z-scored traces (our default) | fast on a CPU |
| ASCAD MLP (Prouff et al., 2018) | `ascad_mlp` | RMSprop, lr 1e-5, 200 epochs, batch 100, raw traces | fast on a CPU |
| Zaid CNN (Zaid et al., TCHES 2020) | `zaid_cnn` | Adam with one-cycle schedule (max lr 5e-3), 50 epochs, batch 50, z-scored then scaled to [0, 1] | fine on a CPU |
| ASCAD CNN (Prouff et al., 2018) | `ascad_cnn` | RMSprop, lr 1e-5, 75 epochs, batch 200, raw traces | needs a GPU |
| CPA, DPA, templates | `cpa`, `dpa`, `template` | no training | NumPy only |

**Training settings.** Each deep-learning baseline is trained by default with the settings of its authors' reference code: [ANSSI's ASCAD code](https://github.com/ANSSI-FR/ASCAD) (`ASCAD_train_models.py`, the settings of the published `mlp_best` and `cnn_best` models) and [Zaid et al.'s code](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA) (`ASCAD/N0=0/cnn_architecture.py` and its one-cycle schedule in `clr.py`). The weight initialisation and the optimizer constants follow those Keras scripts as well. One difference: Zaid et al. train on the first 45,000 ASCAD profiling traces and keep 5,000 for validation, while `run_attack` trains on all of them (`train_traces=45000` matches their setup). Any setting can be changed per call, e.g. `run_attack(..., epochs=10, lr=1e-4, optimizer="adam", scaling="standard")`; `get_training_settings("ascad_cnn")` shows the defaults and `result.training` the settings a run used. The raw-trace settings of the ASCAD models were made for ASCAD's 8-bit traces; for traces in other units, pass `scaling="standard"`. On ASCAD itself, `ascad_mlp` with `scaling="standard"` also needs fewer attack traces to find the key than with the original raw traces.

**Reproducing the baselines.** With these default settings, every deep-learning baseline finds the key of ASCAD (fixed key, `ascadf`, key byte 2) for each of three seeds. The numbers are the attack traces needed until the guessing entropy stays at 0 (100 random orderings of the 10,000 attack traces; one NVIDIA A100, PyTorch 2.14):

| Model | `seed=0` | `seed=1` | `seed=2` |
|---|---|---|---|
| `zaid_cnn` | 359 | 442 | 401 |
| `ascad_cnn` | 2,429 | 1,963 | 2,288 |
| `ascad_mlp` | 3,237 | 6,306 | 1,821 |

For example, `run_attack("ascadf", model="zaid_cnn", leakage=LeakageModel(byte=2), seed=0).ranks.traces_to_disclosure()` gives the first number. A run repeats exactly on the same hardware and software; another GPU, the CPU or another PyTorch version can give slightly different numbers.

Choose the device with `device="cuda"` (NVIDIA), `"mps"` (Apple silicon) or `"cpu"`; the default is CUDA if available, otherwise the CPU. Runs are seeded (`seed=0` by default) and repeat exactly. The default leakage model is the AES first-round S-box output of key byte 0 (256 classes); `LeakageModel(byte=2, leakage="hw")` targets byte 2 with the Hamming weight instead.

## 3. Guessing entropy, success rate and plots

Every attack returns the guessing entropy (the mean rank of the correct key, 0 = found) and the success rate against the number of attack traces, averaged over 100 random orderings of the attack traces:

```python
print(mlp.guessing_entropy)               # one value per number of attack traces
print(mlp.success_rate)
print(mlp.ranks.traces_to_disclosure())   # traces needed until GE stays at 0

from mlsca_bench.benchmark.plots import plot_guessing_entropy, plot_success_rate

plot_guessing_entropy({"mlp": mlp, "cpa": cpa, "template": template}, save="ge.png")
plot_success_rate({"mlp": mlp, "cpa": cpa, "template": template}, save="sr.png")
```

To compute them from your own predictions, use `guessing_entropy(probabilities, hypothesis_labels, correct_key)` and `success_rate(...)` from `mlsca_bench.benchmark.metrics`.

## 4. Add your own dataset

Describe your file once, then use the name everywhere: `load_dataset`, `run_attack`, `run_classical`.

```python
from mlsca_bench import register_dataset

register_dataset(
    "my-lab-aes",
    {
        "adapter": "hdf5",
        "adapter_config": {
            "splits": {"all": "/"},
            "traces_dataset": "traces",
            "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]},
        },
        "metadata": {"format": "hdf5", "algorithm": "AES-128", "measurement": "power"},
    },
    path="/data/my_traces.h5",
    save=True,                    # remember it in every future session
)

mlp = run_attack("my-lab-aes", model="mlp", epochs=10)
```

The entry is checked when you register it, and a misspelled field or reader names what was expected. Readers exist for HDF5, NumPy, MATLAB, Riscure `.trs`, raw binary and more; [using your own dataset](documentation/your-own-dataset.md) shows each option. `mlsca-bench forget my-lab-aes` removes a saved dataset (your files are kept).

## 5. Add your own model

A model is a function that receives the trace length and the number of classes and returns a PyTorch module:

```python
import torch.nn as nn
from mlsca_bench.models import register_model, run_attack

@register_model("my_cnn")
def my_cnn(*, input_length: int, n_classes: int, channels: int = 16):
    return nn.Sequential(
        nn.Unflatten(1, (1, input_length)),
        nn.Conv1d(1, channels, kernel_size=11, padding=5), nn.ReLU(), nn.AvgPool1d(2),
        nn.Flatten(), nn.Linear(channels * (input_length // 2), n_classes),
    )

result = run_attack("ascadf", model="my_cnn", model_kwargs={"channels": 32},
                    leakage=LeakageModel(byte=2), epochs=10)
```

It is then trained, attacked and scored exactly like the baselines, with Adam, lr 1e-3, 50 epochs, batch 128 and z-scored traces unless you say otherwise. To give your model its own defaults, register it with `register_model("my_cnn", training=TrainingSettings(optimizer="rmsprop", lr=1e-5, epochs=100, batch_size=200, scaling="none", source="my paper"))` (`TrainingSettings` is in `mlsca_bench.models`). The built-in names cannot be replaced by accident.

## The datasets

64 public datasets covering 12 algorithms (AES-128, AES-256, DES, PRESENT, Ascon-128, Clyde-128, ECC, Curve25519, Ed25519, Dilithium, BLISS and CRYSTALS-Kyber) on Arm Cortex-M, Arm Cortex-A, AVR and RISC-V microcontrollers, FPGAs and ASICs, with power and electromagnetic measurements, about 13 TB in total. `mlsca-bench list` and `mlsca-bench info NAME` show each one.

- **What they contain:** 61 of the 64 were downloaded and loaded from their real files (the largest on one file each), and the fields seen there are listed by `mlsca-bench list` ("?" = not checked yet).
- **Licenses and terms of use:** MLSCA-Bench does not host or redistribute any dataset. Each is downloaded from its original source and stays under its authors' terms: cite the original work, and check the license with `mlsca-bench info NAME` (for example, `aes-rd` is for non-commercial use only). Where no license is stated, ask the authors before redistributing the data.
- **Known problems in the published data**, such as empty records or a file that duplicates another dataset, are listed in [known dataset issues](documentation/known-dataset-issues.md).

## Contributing

New datasets and models are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

- Iris Dania Jimenez: iris.jimenez AT unibw.de
- Michael Hutter: michael.hutter AT unibw.de

## License / copyright

Copyright © 2026 Universität der Bundeswehr München / Research Institute CODE - ESSEC Lab.

The code is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE) and the SPDX headers in each file. The datasets are not part of MLSCA-Bench: each is downloaded from its original source and stays under its authors' terms (see [The datasets](#the-datasets)).
