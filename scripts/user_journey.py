# Copyright (c) 2026 Universität der Bundeswehr München / FI CODE - ESSEC Lab.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of the five core features, as a user would run them.

    python scripts/user_journey.py            # about 10 minutes on a laptop CPU
    python scripts/user_journey.py --small    # use ascadv (439 MB) instead of ascadf (4.4 GB)
    python scripts/user_journey.py --full     # heavy models on the full training set (GPU advised)
    python scripts/user_journey.py --no-readme

Checks, each reported as PASS or FAIL with its time:
  0. the README's Python examples, run as written (only the example file path is replaced)
  1. download and load a dataset; the command line
  2. every baseline: 4 deep-learning models and CPA, DPA, templates
  3. guessing entropy, success rate, traces to disclosure, plots
  4. add your own dataset (saved, reloaded in a new session, attacked)
  5. add your own model (registered, trained, built-in names protected)
Writes user_journey_report.txt in the current folder. Needs the hdf5, eval and plot extras.
Your own saved datasets are not touched: a temporary personal registry is used.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="mlsca-journey-"))
os.environ["MLSCA_BENCH_USER_REGISTRY"] = str(WORK / "saved-datasets.json")    # before importing the package

parser = argparse.ArgumentParser()
parser.add_argument("--small", action="store_true", help="use ascadv (439 MB) instead of ascadf (4.4 GB)")
parser.add_argument("--full", action="store_true", help="train the heavy models on the full training set")
parser.add_argument("--no-readme", action="store_true", help="skip running the README examples")
parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: CUDA if available)")
parser.add_argument("--ascad-path", default=None,
                    help="an existing ASCAD.h5 (or ascad-variable.h5 with --small): use it instead of downloading")
args = parser.parse_args()

DATASET = "ascadv" if args.small else "ascadf"
LOAD = {"path": args.ascad_path} if args.ascad_path else {}       # existing files, no download
REPO = Path(__file__).resolve().parent.parent
REPORT = Path.cwd() / "user_journey_report.txt"
results: list[tuple[str, str, float, str]] = []


def check(name: str):
    def wrap(fn):
        t = time.time()
        try:
            detail = fn() or ""
            results.append((name, "PASS", time.time() - t, str(detail)))
        except Exception:  # noqa: BLE001 - report every failure and continue
            results.append((name, "FAIL", time.time() - t, traceback.format_exc()[-3000:]))
        status, seconds = results[-1][1], results[-1][2]
        print(f"[{status}] {name} ({seconds:.0f} s)" + (f": {results[-1][3].splitlines()[-1][:150]}"
                                                         if status == "FAIL" else ""), flush=True)
        return fn
    return wrap


def cli(*argv: str) -> str:
    out = subprocess.run([sys.executable, "-m", "mlsca_bench", *argv], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=os.environ.copy(), timeout=3600)
    if out.returncode != 0:
        raise RuntimeError(f"mlsca-bench {' '.join(argv)} failed ({out.returncode}): {out.stderr[-1500:]}")
    return out.stdout


def synthetic_hdf5(path: Path, n: int = 4000, n_samples: int = 50) -> bytes:
    """AES-like traces: sample 7 leaks HW(Sbox[plaintext[0] ^ key[0]]) plus noise."""

    import h5py
    import numpy as np
    from mlsca_bench.benchmark import HAMMING_WEIGHT, SBOX

    rng = np.random.default_rng(1)
    key = rng.integers(0, 256, 16, dtype=np.uint8)
    plaintext = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    traces = rng.normal(0, 0.5, (n, n_samples)).astype(np.float32)
    traces[:, 7] += HAMMING_WEIGHT[SBOX[plaintext[:, 0] ^ key[0]]]
    with h5py.File(path, "w") as f:
        f["traces"] = traces
        f["metadata/plaintext"] = plaintext
        f["metadata/key"] = np.tile(key, (n, 1))
    return key.tobytes()


import numpy as np  # noqa: E402

import mlsca_bench  # noqa: E402
from mlsca_bench import get_dataset, list_datasets, load_dataset, register_dataset  # noqa: E402
from mlsca_bench.benchmark import LeakageModel, run_classical  # noqa: E402
from mlsca_bench.benchmark.metrics import guessing_entropy, success_rate  # noqa: E402

LEAK = LeakageModel(byte=2)            # both ASCAD v1 datasets are cut around key byte 2
QUICK = {} if args.full else {"train_traces": 500, "epochs": 1}
STATE: dict = {}


@check("environment")
def _():
    import platform
    import importlib.metadata as md

    versions = {p: md.version(p) for p in ("mlsca-bench", "numpy", "h5py", "torch", "matplotlib")}
    import torch
    gpu = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "none"
    return f"{platform.platform()} | Python {platform.python_version()} | {versions} | GPU: {gpu}"


# ---- 0. README examples, as written ---------------------------------------------------------
if not args.no_readme:
    @check("0. README examples (run as written)")
    def _():
        text = (REPO / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
        if len(blocks) < 6:
            raise AssertionError(f"expected the README's Python examples, found {len(blocks)} blocks")
        data = WORK / "readme_traces.h5"
        synthetic_hdf5(data)
        namespace: dict = {}
        for number, block in enumerate(blocks, 1):
            code = block.replace('"/data/my_traces.h5"', repr(str(data)))
            if args.ascad_path:          # every README call on ascadf uses the existing file
                code = code.replace('"ascadf",', f'"ascadf", path={args.ascad_path!r},')
            try:
                exec(compile(code, f"README python block {number}", "exec"), namespace)
            except Exception as error:
                raise AssertionError(f"README python block {number} failed:\n{code}\n{error!r}") from error
        for png in ("ge.png", "sr.png"):
            if not Path(png).is_file():
                raise AssertionError(f"README example did not create {png}")
            Path(png).unlink()
        return f"{len(blocks)} blocks ran; mlp final GE {namespace['mlp'].guessing_entropy[-1]:.1f}"


# ---- 1. download and load ---------------------------------------------------------------------
@check("1. list, info and fields")
def _():
    specs = list_datasets(builtin_only=True)
    assert len(specs) == 64, len(specs)
    spec = get_dataset(DATASET)
    assert spec.size_bytes and spec.license and spec.fields, spec
    out = cli("list", "--with", "plaintexts", "keys")
    assert DATASET in out, out[-500:]
    assert "fields" in cli("info", DATASET)
    return f"{DATASET}: {spec.size}, fields {dict(spec.fields)}"


@check(f"1. download and load {DATASET}")
def _():
    with load_dataset(DATASET, split="attack", **LOAD) as ds:
        assert len(ds) > 0 and ds.traces.ndim == 2
        assert {"traces", "plaintexts", "keys"} <= set(ds.available_fields), ds.available_fields
        rows = np.asarray(ds.traces[:100])
        assert rows.shape[0] == 100 and np.isfinite(rows.astype(float)).all()
        sample = ds[0]
        assert sample.plaintext is not None and sample.key is not None
        return f"attack split {ds.shape}, fields {sorted(ds.available_fields)}"


@check("1. command line: download, cache")
def _():
    if args.ascad_path:
        return "skipped: existing files given with --ascad-path"
    cli("download", DATASET, "--quiet")
    out = cli("cache")
    assert DATASET in out and "Total:" in out
    return out.strip().splitlines()[-1]


# ---- 2. baselines -------------------------------------------------------------------------------
from mlsca_bench.models import list_models, run_attack  # noqa: E402

LIGHT = {"mlp", "ascad_mlp", "zaid_cnn"}


for model_name in ("mlp", "ascad_mlp", "zaid_cnn", "ascad_cnn"):
    @check(f"2. baseline {model_name}")
    def _(model_name=model_name):
        settings = {"epochs": 5} if model_name in LIGHT else dict(QUICK)
        result = run_attack(DATASET, model=model_name, leakage=LEAK, device=args.device, n_experiments=20,
                            **settings, **LOAD)
        ge = result.guessing_entropy
        assert ge.ndim == 1 and len(ge) == result.n_test and np.isfinite(ge).all() and 0 <= ge.min() <= 255
        assert result.success_rate.shape == ge.shape
        STATE[model_name] = result
        return f"{settings}, trained on {result.n_train}, final GE {ge[-1]:.1f}"


for attack in ("cpa", "dpa", "template"):
    @check(f"2. baseline {attack}")
    def _(attack=attack):
        result = run_classical(attack, DATASET, leakage=LEAK, n_experiments=20, **LOAD)
        ge = result.guessing_entropy
        assert np.isfinite(ge).all() and result.success_rate.shape == ge.shape
        STATE[attack] = result
        return f"final GE {ge[-1]:.1f}"


@check("2. a light baseline learns (mlp, 10 epochs: GE close to 0)")
def _():
    result = run_attack(DATASET, model="mlp", leakage=LEAK, epochs=10, device=args.device, n_experiments=20, **LOAD)
    assert result.guessing_entropy[-1] <= 5, result.guessing_entropy[-1]
    STATE["mlp10"] = result
    return f"final GE {result.guessing_entropy[-1]:.2f}, disclosure after {result.ranks.traces_to_disclosure()}"


# ---- 3. GE, SR and plots ------------------------------------------------------------------------
@check("3. GE, SR and traces to disclosure")
def _():
    result = STATE["mlp10"]
    ge, sr = result.guessing_entropy, result.success_rate
    assert ge[0] >= ge[-1] and sr[-1] >= sr[0] and ((0 <= sr) & (sr <= 1)).all()
    rng = np.random.default_rng(0)                 # metrics from your own predictions
    hyp = rng.integers(0, 9, (200, 256))
    probs = rng.random((200, 9))
    probs[np.arange(200), hyp[:, 42]] += 5          # the model favours key 42
    probs /= probs.sum(1, keepdims=True)
    ge_own = guessing_entropy(probs, hyp, 42, n_experiments=10)
    sr_own = success_rate(probs, hyp, 42, n_experiments=10)
    assert ge_own[-1] == 0 and sr_own[-1] == 1
    return f"disclosure after {result.ranks.traces_to_disclosure()} traces; own-prediction metrics OK"


@check("3. plots (GE and SR, several attacks)")
def _():
    from mlsca_bench.benchmark.plots import plot_guessing_entropy, plot_success_rate

    compare = {k: STATE[k] for k in ("mlp10", "cpa", "template") if k in STATE}
    for fn, file in ((plot_guessing_entropy, WORK / "ge.png"), (plot_success_rate, WORK / "sr.png")):
        fn(compare, save=file)
        assert file.stat().st_size > 5000, file
    return f"{len(compare)} curves per plot"


# ---- 4. your own dataset ---------------------------------------------------------------------------
@check("4. register a dataset (saved), reload in a new session, attack it")
def _():
    path = WORK / "my_traces.h5"
    synthetic_hdf5(path)
    entry = {
        "adapter": "hdf5",
        "adapter_config": {"splits": {"all": "/"}, "traces_dataset": "traces",
                           "field_aliases": {"plaintexts": ["metadata/plaintext"], "keys": ["metadata/key"]}},
        "metadata": {"format": "hdf5", "algorithm": "AES-128", "measurement": "power"},
    }
    register_dataset("journey-lab", entry, path=path, save=True)
    code = textwrap.dedent("""
        from mlsca_bench import load_dataset
        with load_dataset("journey-lab") as ds:
            print(ds.shape)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0 and "4000" in out.stdout, out.stderr[-1500:]
    hw = LeakageModel(byte=0, leakage="hw")
    dl = run_attack("journey-lab", model="mlp", leakage=hw, epochs=10, device=args.device, n_experiments=20)
    cpa = run_classical("cpa", "journey-lab", leakage=hw, n_experiments=20)
    assert dl.guessing_entropy[-1] <= 5 and cpa.guessing_entropy[-1] <= 5, (dl.guessing_entropy[-1], cpa.guessing_entropy[-1])
    assert "Forgot" in cli("forget", "journey-lab")
    return f"new session sees it; mlp GE {dl.guessing_entropy[-1]:.1f}, CPA GE {cpa.guessing_entropy[-1]:.1f}"


@check("4. a misspelled entry is rejected with a clear message")
def _():
    from mlsca_bench.datasets.registry import RegistryValidationError
    try:
        register_dataset("journey-bad", {"adapter": "hdf", "metadata": {"format": "hdf5", "algorithm": "AES-128",
                                                                         "measurement": "power"}})
    except RegistryValidationError as error:
        assert "did you mean 'hdf5'" in str(error), error
        return "rejected: " + str(error)[:120]
    raise AssertionError("a misspelled adapter was accepted")


# ---- 5. your own model ------------------------------------------------------------------------------
@check("5. register a model, train and attack with it")
def _():
    import torch.nn as nn
    from mlsca_bench.models import register_model

    used = []

    @register_model("journey_cnn")
    def journey_cnn(*, input_length: int, n_classes: int, channels: int = 8):
        used.append(channels)
        return nn.Sequential(nn.Unflatten(1, (1, input_length)), nn.Conv1d(1, channels, 11, padding=5), nn.ReLU(),
                             nn.AvgPool1d(2), nn.Flatten(), nn.Linear(channels * (input_length // 2), n_classes))

    assert "journey_cnn" in list_models()
    result = run_attack(DATASET, model="journey_cnn", model_kwargs={"channels": 4}, leakage=LEAK,
                        epochs=1, train_traces=2000, device=args.device, n_experiments=20, **LOAD)
    assert used == [4] and np.isfinite(result.guessing_entropy).all()
    try:
        register_model("mlp")(journey_cnn)
    except ValueError:
        return f"trained and attacked; final GE {result.guessing_entropy[-1]:.1f}; built-in names protected"
    raise AssertionError("a built-in model name could be replaced")


passed = sum(1 for r in results if r[1] == "PASS")
failed = [r[0] for r in results if r[1] == "FAIL"]
lines = [f"MLSCA-Bench user journey ({DATASET}{', full' if args.full else ''}), {time.strftime('%Y-%m-%d %H:%M')}",
         f"{passed} passed, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""), ""]
for name, status, seconds, detail in results:
    lines += [f"=== [{status}] {name} ({seconds:.0f} s)", detail, ""]
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n{passed} passed, {len(failed)} failed. Report: {REPORT}")
sys.exit(1 if failed else 0)
