# Bringing the RIANN datasets into IdentiBench

This document describes the integration (now **implemented** in
`identibench/datasets/riann.py` + `_riann_prep/`) that ports the six IMU
orientation-estimation datasets used by the **RIANN** paper into IdentiBench, so
that anyone can `pip install identibench`, download the datasets with one call,
and evaluate their own model — neural network, classical filter, or anything
else — under the exact protocol RIANN used. It also doubles as the rationale
record for the design choices.

> Weber, Gühmann, Seel. *RIANN — A Robust Neural Network Outperforms Attitude
> Estimation Filters.* AI 2021, 2(3):444–463.
> [10.3390/ai2030028](https://doi.org/10.3390/ai2030028).

Source repos referenced below:
- **RIANN**: `../riann_dev` (datasets + prep + the reference evaluation recipe).
- **IdentiBench**: this repo (the benchmarking host).

## TL;DR

**Verdict: strong fit, low-friction.** IdentiBench already has every abstraction
this needs — `BenchmarkSpecSimulation`, a `split` mechanism, the
`inclination_rmse_deg` metric, the `Sequence` loader, and the
download-from-source `download_func` model. RIANN's on-disk HDF5 format is
*already* what IdentiBench's loader expects (per-channel 1-D datasets), so there
is essentially **no data reformatting**. The only non-trivial work is
reproducing RIANN's *masked + first-sample-aligned* evaluation, which slots into
the `custom_test_evaluation` hook. No core-library changes are required.

**Scope decided for this plan:**
- Expose **per-source datasets** (each of the six as its own `dataset_id` +
  benchmark) **and** a **combined RIANN corpus** that reproduces the paper's
  pooled-train / cross-dataset-test protocol.
- **Datasets + evaluation only** — no baseline models shipped (users plug in
  their own via `build_model`).

## Contents
- [Why IdentiBench is the right host](#why-identibench-is-the-right-host)
- [The integration contract](#the-integration-contract)
- [Data format: RIANN → IdentiBench](#data-format-riann--identibench)
- [The evaluation-fidelity problem (and its fix)](#the-evaluation-fidelity-problem-and-its-fix)
- [Design](#design)
- [The splits](#the-splits)
- [Packaging & dependencies](#packaging--dependencies)
- [Deferred: the perturbation scenarios](#deferred-the-perturbation-scenarios)
- [Decisions taken & open questions](#decisions-taken--open-questions)
- [Implementation roadmap](#implementation-roadmap)
- [Testing strategy](#testing-strategy)
- [Footgun: the stale install](#footgun-the-stale-install)

## Why IdentiBench is the right host

| What the RIANN benchmark needs | What IdentiBench already provides |
|---|---|
| Orientation-from-IMU, evaluated over the full sequence | `BenchmarkSpecSimulation` with `init_window=0`; `_test_simulation` runs the model once per file over the whole sequence (`benchmark.py:178`) |
| Tilt-error metric in degrees | `metrics.inclination_rmse_deg` — already RIANN's metric (`[w,x,y,z]`, `2·atan2(…)`), minus masking/alignment (`metrics.py:181`) |
| Per-channel 1-D HDF5 (`acc_x…gyr_z`, `opt_a…opt_d`, `dt`) | `_load_sequences_from_files` does `np.stack([f[col] …])` per column → **zero reformatting** (`utils.py:96`) |
| 7th input channel = per-sample `dt` | RIANN stores `dt` as a real `(N,)` array (`np.full(n, dt)` in `riann/prep/_common.py:write_hdf5`), so it is just another `u_col` |
| Multi-dataset train + cross-dataset test | the `split={"train":[…],"valid":[…],"test":[…]}` kwarg on every spec (`benchmark.py:66`, `:116`) |
| Per-source result breakdown | `custom_test_evaluation` → `custom_scores`, flattened to `cs_*` columns by `benchmark_results_to_dataframe` (`benchmark.py:411`) |
| Download-from-source + local cache | `download_func(save_path, force_download)` run in a spawn subprocess; `~/.identibench_data` cache (`benchmark.py:150`) |
| Reusable, validated prep | `riann/prep/{broad,euroc,tumvi,oxiod,repoimu,caruso}.py`, each exposing `download(raw_dir)` + `convert(raw_dir, out_dir)` |

The existing `BenchmarkIMU_Inclination` (Weygers & Kok, `datasets/imu.py`) is a
working template for *exactly this task type* — same metric, same
`BenchmarkSpecSimulation`, same `[w,x,y,z]` convention.

## The integration contract

The pieces of IdentiBench we build on, with their exact behaviour:

- **Spec** — `BenchmarkSpecSimulation(name, dataset_id, u_cols, y_cols,
  metric_func, download_func, sampling_time, init_window, split,
  custom_test_evaluation, data_root)`.
- **File resolution** — with a `split` dict, `*_files` properties return
  `sorted(dataset_path / f for f in split[subset])` (`benchmark.py:116-148`).
  The entries are **relative paths joined to `dataset_path`**, so they may
  contain sub-directories (e.g. `"OxIOD/handheld_data1_imu1.hdf5"`).
- **Download** — `ensure_dataset_exists` runs `download_func(dataset_path,
  force_download)` in a `spawn` subprocess if `dataset_path` is missing
  (`benchmark.py:150-175`). Side-effect isolation; must be self-contained.
- **Loader** — `_load_sequences_from_files(files, u_cols, y_cols)` yields
  `Sequence(u=(N,|u_cols|), y=(N,|y_cols|), attrs=dict(f.attrs))`
  (`utils.py:79-113`). **It reads only `u_cols`, `y_cols`, and file `attrs`** —
  any other dataset in the file (e.g. `movement_mask`) is invisible to the model
  and to `metric_func`.
- **Test driver** — `_test_simulation` iterates files in `spec.test_files`
  order, builds `Sequence(u, y[:init_window], attrs)`, calls `model(*seq)`, and
  appends `(y_pred, y_true)` (`benchmark.py:178-186`). Result order == file
  order — this is what lets a custom evaluator re-pair predictions with files.
- **Metric** — `aggregate_metric_score` calls `metric_func(y_pred, y_true)` per
  sequence and averages (`benchmark.py:24`, `:383`). Signature is
  `(inp=pred, targ=true)` (note the PyTorch-style order, fixed in commit
  `67fe06d`).
- **Custom eval** — `custom_test_evaluation(test_results, spec)` returns a dict
  merged into `results["custom_scores"]` (`benchmark.py:384`). This is the only
  hook with access to `spec` (hence `spec.test_files`).

## Data format: RIANN → IdentiBench

RIANN's `write_hdf5` (`riann/prep/_common.py`) emits exactly the layout
IdentiBench's loader wants — **1-D `float32` datasets, one per channel**:

| RIANN dataset | Role | IdentiBench `u_cols` / `y_cols` |
|---|---|---|
| `acc_x, acc_y, acc_z` (m/s²) | input | `u_cols[0:3]` |
| `gyr_x, gyr_y, gyr_z` (rad/s) | input | `u_cols[3:6]` |
| `dt` (s, stored as `(N,)`) | input | `u_cols[6]` |
| `opt_a, opt_b, opt_c, opt_d` (quaternion `w,x,y,z`) | target | `y_cols[0:4]` |
| `movement_mask` (`(N,)`, 1=moving) | **eval only** | not in `u`/`y` — recovered in custom eval |
| `mag_x, mag_y, mag_z` (optional) | unused | — |

So the canonical spec uses:

```python
u_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
y_cols = ["opt_a", "opt_b", "opt_c", "opt_d"]
```

Notes / gotchas, all verified:
- **`dt` is a 7-channel-compatible array**, not a scalar — a RIANN-style 7-input
  GRU runs unchanged. A model that doesn't want `dt` can drop `u_cols[6]`.
- **RIANN files carry no `fs` attr** (`f.attrs == {}`). Sampling rate lives only
  in the `dt` channel. `sampling_time` on the spec stays `None`; anything that
  needs the rate reads `dt`.
- **Quaternion convention is `[w,x,y,z]`**, matching `metrics.py` and
  `datasets/imu.py`. No reordering.
- **Units are already standardized** by the prep modules (acc m/s², gyr rad/s).
- `convert()` writes into `out_dir/<SourceName>/…` where `<SourceName>` ∈
  `{Myon, TUM-VI, OxIOD, EuRoC-MAV, RepoIMU, Caruso-Sassari,
  Caruso-Sassari_orig}`.

## The evaluation-fidelity problem (and its fix)

RIANN's published metric is **not** plain `inclination_rmse_deg`. Per
`riann_dev/scripts/validate_vqf.py` and `REPRODUCTION.md`, the recipe is:

1. **Align** the estimate to ground truth at the **first valid sample**
   (`q_offset = qmult(gt₀, qinv(est₀))`, applied to all samples) — removes the
   constant offset between the IMU's gravity-defined frame and the optical
   reference. **Required even for inclination** (there is a constant tilt
   offset), so un-aligned numbers are inflated and not comparable to the paper.
2. Compute the relative quaternion and take the **inclination angle**.
3. **Mask**: drop samples where GT is NaN and where `movement_mask == 0`.
4. Report `rmse = rad2deg(sqrt(mean(incl²)))` **and the 99th percentile**, **per
   dataset**.

Why this can't all live in `metric_func`: the loader (`utils.py:96`) stacks only
`u_cols`/`y_cols`, so **`movement_mask` never reaches `metric_func`**, and
`metric_func(y_pred, y_true) -> float` returns a single scalar (no room for the
99th-pct or per-source split). The data-flow trace confirms there is no path for
the mask through `metric_func`.

**The fix — split the evaluation across the two hooks IdentiBench gives us:**

- **`metric_func`** = a small *aligned* inclination-RMSE. It receives both full
  arrays, so it can do first-sample alignment itself (just not masking). This
  produces a sensible headline `metric_score`.
- **`custom_test_evaluation(test_results, spec)`** = the **fully faithful**
  recipe. Because `test_results` is in `spec.test_files` order, it re-opens each
  file to recover `movement_mask` (and GT NaNs), applies alignment + mask, and
  emits per-source RMSE **and** p99:

```python
def riann_eval(test_results, spec):
    scores = {}
    per_source = defaultdict(list)        # source -> list of (incl_deg array)
    for (y_pred, y_true), fpath in zip(test_results, spec.test_files):
        with h5py.File(fpath, "r") as f:
            mask = f["movement_mask"][()].astype(bool)
        valid = mask & ~np.isnan(y_true).any(-1)
        incl = aligned_inclination_deg(y_pred, y_true)   # pure-numpy, see below
        incl = incl[valid]
        source = Path(fpath).parent.name                 # "OxIOD", "EuRoC-MAV", …
        per_source[source].append(incl)
    all_incl = []
    for source, chunks in per_source.items():
        v = np.concatenate(chunks)
        scores[f"{source}/incl_rmse_deg"] = float(np.sqrt(np.mean(v**2)))
        scores[f"{source}/incl_p99_deg"]  = float(np.percentile(np.abs(v), 99))
        all_incl.append(v)
    v = np.concatenate(all_incl)
    scores["all/incl_rmse_deg"] = float(np.sqrt(np.mean(v**2)))
    scores["all/incl_p99_deg"]  = float(np.percentile(np.abs(v), 99))
    return scores
```

`benchmark_results_to_dataframe` flattens these to `cs_OxIOD/incl_rmse_deg` etc.,
so the paper's entire **per-dataset table comes out of one run**.

**No `qmt` dependency needed.** `aligned_inclination_deg` reuses the existing
pure-numpy helpers `metrics._quat_diff` / `metrics._inclination_angle`, adding
first-sample alignment via the same `_quat_diff`/multiply primitives. (`qmt`
could be an optional cross-check, not a requirement.)

> **Document this loudly:** `metric_score` is the aligned-but-unmasked headline;
> the paper-faithful numbers are in `custom_scores` (`cs_*`). This avoids the
> dual-metric confusion of having two "inclination RMSE" numbers.

## Design

A single new module, **`identibench/datasets/riann.py`**, plus the vendored prep
package and registration. **No core-library edits.**

### A. Per-source datasets & benchmarks

Each source is a standalone `dataset_id` named after the dataset itself (the
`riann_` prefix is reserved for the *protocol*, not the data — these are general
IMU orientation datasets that RIANN happened to use). Each has its own thin
download wrapper and a benchmark spec, all sharing `riann_eval` and the same
`u_cols`/`y_cols`. Per the "evaluate one dataset in isolation" goal, **every
file in a per-source dataset is treated as test**:

| `dataset_id` | `dl_*` | Spec | Files (all test) |
|---|---|---|---|
| `broad`   | `dl_broad`   | `BenchmarkBROAD_Inclination`   | 39 |
| `tumvi`   | `dl_tumvi`   | `BenchmarkTUMVI_Inclination`   | 6  |
| `oxiod`   | `dl_oxiod`   | `BenchmarkOxIOD_Inclination`   | 71 |
| `euroc`   | `dl_euroc`   | `BenchmarkEuRoC_Inclination`   | 6  |
| `repoimu` | `dl_repoimu` | `BenchmarkRepoIMU_Inclination` | 21 |
| `caruso`  | `dl_caruso`  | `BenchmarkCaruso_Inclination`  | 18 |

Each `dl_*` materializes all of that source's files into the dataset's `test/`
subdir (no `split` dict needed). These let a user grab a 6-file EuRoC download
and evaluate in isolation, exactly like the existing workshop/IMU benchmarks.
(The combined `riann` corpus, by contrast, applies RIANN's train/valid/test
roles — see B.)

> **Resolved — the orphaned `dl_broad` was replaced.** IdentiBench previously
> shipped a `dl_broad` (in `datasets/broad.py`) that fetched the *same* upstream
> (`dlaidig/broad`) but wrote an older `acc0/acc1…` format with no `dt`/mask, and
> had no active benchmark (its spec import was commented out). It was deleted;
> `dataset_id="broad"` now uses the vendored RIANN prep (richer: `dt`,
> `movement_mask`, units). BROAD keeps RIANN's "Myon"-order trial filenames — the
> data is byte-identical to upstream, only filenames are reordered.

### B. Combined RIANN corpus

One `dataset_id = "riann"` reproducing the paper protocol verbatim:

- **`dl_riann(save_path, force_download)`** stages **all six** sources via the
  vendored prep into `save_path/<SourceName>/…` (their native sub-dir layout —
  this also avoids the one real filename collision: `Caruso-Sassari/Marco::…`
  and `Caruso-Sassari_orig/Marco::…` share identical filenames and must stay in
  separate dirs).
- A **`split` dict with sub-dir-qualified relative paths**
  (`"OxIOD/…hdf5"`, `"Myon/29_…hdf5"`, …) computed by re-implementing
  `get_files()`'s rules (see [The splits](#the-splits)).
- **`BenchmarkRIANN_Inclination`** = the headline spec, with
  `custom_test_evaluation=riann_eval` (the per-source breakdown gives the paper's
  per-dataset panel from a single materialization + single run).

### C. Shared raw cache (avoid double downloads)

Per-source `riann_*` and the combined `riann` would otherwise download the same
upstream twice. Mitigate by pointing every `dl_*` at a **shared raw cache** (e.g.
`data_root/_riann_raw/`): the prep API already separates `download(raw_dir)` from
`convert(raw_dir, out_dir)`, so raw archives are fetched once and only `convert`
re-runs per `dataset_id`. (`convert` is cheap relative to download.)

### D. Registration

- `datasets/__init__.py`: add the `dl_*` funcs to `all_dataset_loaders`.
- `identibench/__init__.py`: export the specs; add to `simulation_benchmarks`;
  optionally add a `riann_benchmarks` group dict mirroring `workshop_benchmarks`.

## The splits

Re-implemented inline from `riann/data.py` (verified counts against the
materialized `data/`):

```python
MYON_VALID_IDS  = {14, 39, 21}
MYON_TEST_IDS   = {29, 22, 35}
TUMVI_TRAIN_ROOMS = {"room1", "room2", "room3"}
TEST_SOURCES = ["OxIOD", "EuRoC-MAV", "RepoIMU", "Caruso-Sassari", "Caruso-Sassari_orig"]
```

- **Myon (BROAD), 39 files** → 33 train / 3 valid (`14,39,21`) / 3 test
  (`29,22,35`), keyed on the leading integer of the filename.
- **TUM-VI, 6 files** (`TumVI::room1..6`) → rooms 1–3 train, 4–6 valid.
- **OxIOD 71, EuRoC-MAV 6, RepoIMU 21, Caruso-Sassari 18, Caruso-Sassari_orig
  18** → all test.

Combined corpus totals: **36 train / 6 valid / 137 test** (or **119 test** if
`Caruso-Sassari_orig` is excluded — see open questions).

## Packaging & dependencies

- **Vendor** `riann/prep/{_common,broad,caruso,euroc,oxiod,repoimu,tumvi}.py`
  (+ a `PREPARERS`-style registry) under
  `identibench/datasets/_riann_prep/`. **Do not `import riann`** — `riann_dev`'s
  `riann/__init__.py` is a 0-byte file that shadows the public `pip install
  riann` model package; vendoring keeps `dl_riann` self-contained inside the
  spawn subprocess.
- **No new heavy dependencies.** Prep needs `requests`, `h5py`, `numpy`, `tqdm`,
  and `gdown` (OxIOD, already present) + archive handling (`zip`/`tar` in stdlib;
  `rarfile` already a dep). The faithful metric is pure-numpy — **`qmt` not
  required**.
- Per-source download sizes vary (EuRoC ~tens of MB; OxIOD larger via Google
  Drive). The combined `dl_riann` is the heaviest; document approximate sizes.

Upstream sources (fetched at runtime, **not re-hosted** — same model as
IdentiBench's other datasets):

| Source | Upstream |
|---|---|
| BROAD (Myon) | `github.com/dlaidig/broad` |
| TUM-VI | `cvg.cit.tum.de/data/datasets/visual-inertial-dataset` |
| OxIOD | `deepio.cs.ox.ac.uk` (Google Drive mirror) |
| EuRoC-MAV | ETH Research Collection `doi.org/10.3929/ethz-b-000690084` |
| RepoIMU | `github.com/agnieszkaszczesna/RepoIMU` |
| Caruso-Sassari | `github.com/marcocaruso/mimu_optical_dataset_caruso_sassari` (v5.0) |

## Deferred: the perturbation scenarios

RIANN's headline `gae_4` also reports two robustness scenarios that this plan
**defers to a phase 2**, because they perturb the model *input* and so cannot be
done post-hoc in a metric:

- **biased** — add a constant gyro bias before the model runs. Needs a custom
  `test_model_func` that re-invokes the model on perturbed input. The exact bias
  magnitude is **not in this repo** — `REPRODUCTION.md` notes it must be
  recovered from the legacy `old_project` `_gyrbias.hdf5` files.
- **moving-start (ms)** — slice each sequence to `movement_mask` onset + 1000
  samples, then evaluate. Also a `test_model_func` variant.

Both fit as **additional specs** (`BenchmarkRIANN_Inclination_Biased`, `_MS`)
with a custom `test_model_func`, added once the bias constant is recovered. Ship
*normal + masked + p99* first (high fidelity, low risk).

## Decisions taken & open questions

**Taken & implemented:**
- Per-source datasets **and** a combined corpus.
- Datasets + evaluation only; **no baseline models** shipped.
- Faithful eval via `custom_test_evaluation`; aligned-but-unmasked headline in
  `metric_func`; pure-numpy (no `qmt`).
- **Naming**: the `riann` prefix marks only the *protocol*. The combined corpus
  is `dataset_id="riann"` / `BenchmarkRIANN_Inclination`; the six sources use
  bare `dataset_id`s (`broad`, `tumvi`, `oxiod`, `euroc`, `repoimu`, `caruso`)
  with `dl_<name>` loaders and `Benchmark<Name>_Inclination` specs.
- **`Caruso-Sassari_orig` excluded** from both the combined corpus and the
  standalone `caruso` dataset → 119 test files in the corpus.
- **Old `dl_broad` replaced** (orphaned `datasets/broad.py` deleted); `broad`
  now uses the vendored RIANN prep.
- **Shared raw cache** at `<data_root>/_riann_raw` so the per-source and combined
  datasets don't re-download the same upstream; converted copies still duplicate
  per `dataset_id` (inherent to the one-path-per-dataset_id model).

**Open / deferred:**
1. **Disk duplication** — the combined `riann` corpus materializes its own copy
   of every file rather than referencing the per-source datasets (identibench
   ties one spec to one `dataset_id` path). Acceptable; a custom multi-path spec
   could dedupe later.
2. **Biased / moving-start scenarios** — deferred (need a custom
   `test_model_func`; gyro-bias magnitude must be recovered from legacy
   `_gyrbias.hdf5`).
3. **Heading-aware orientation benchmarks** — not shipped; full-orientation RMSE
   is ill-posed for 6-axis IMU (heading unobservable without magnetometer, which
   is why RIANN reports inclination). The datasets carry `mag_*` if revisited.

## Implementation roadmap

1. **Vendor prep** → `identibench/datasets/_riann_prep/` (copy 7 modules + a
   registry); confirm they run standalone (no `riann` package import).
2. **`datasets/riann.py`**: `aligned_inclination_deg` + `riann_eval`; the six
   per-source `dl_*` wrappers + specs; `dl_riann` + the combined split builder +
   `BenchmarkRIANN_Inclination`.
3. **Register** in `datasets/__init__.py` and `identibench/__init__.py`
   (+ optional `riann_benchmarks` group).
4. **Smoke test** (see below).
5. **Docs**: a README section + a runnable example (`examples/`) showing a
   trivial `build_model` (e.g. identity/VQF-free) evaluated on
   `BenchmarkEuRoC_Inclination` and the combined `BenchmarkRIANN_Inclination`.
6. **Phase 2** (later): recover the gyro-bias constant; add `_Biased` / `_MS`
   specs with custom `test_model_func`s.

Estimated effort for steps 1–5: ~1 focused day; the only fiddly part is
`riann_eval` and validating it against RIANN's `validate_vqf.py` numbers.

## Testing strategy

- **Format contract test** — materialize one small source (EuRoC, 6 files) in a
  tmp `data_root` and assert the loader yields `u.shape[-1]==7`,
  `y.shape[-1]==4`, finite values, and that `movement_mask` is present on disk.
- **Eval-parity test** — pick a couple of `data/` files for which RIANN's
  `validate_vqf.py` produces a known masked-inclination RMSE, feed
  ground-truth-as-prediction (RMSE≈0) and a fixed-offset prediction, and assert
  `riann_eval` matches the alignment+mask behaviour (offset removed, masked
  samples excluded).
- **Registration test** — `BenchmarkRIANN_Inclination` resolves
  `train/valid/test_files` to the expected counts (36/6/137) from a `split` dict
  without touching the network (point `data_root` at a fixture).
- Mirror the existing `tests/` patterns for datasets/benchmarks.

## Footgun: the stale install

While scoping this, an automated check read
`riann_dev/.venv/.../site-packages/identibench` — a **stale `pip install
identibench 0.2.0`** that predates the IMU/`split`/quaternion-metric work — and
wrongly concluded those features didn't exist. They do, in this **source tree**
(git `e6e133a`: `inclination_rmse_deg`, `split`, `datasets/imu.py`, `Sequence`).
When validating, run against an editable install of *this* tree
(`uv sync` / `pip install -e .`), not whatever `import identibench` resolves to
in a sibling project's venv.
