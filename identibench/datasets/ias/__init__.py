"""IAS (instantaneous angular speed) estimation datasets and benchmarks.

Four rotating-machinery datasets ported from the IAS benchmark: estimate the
instantaneous angular speed ``y = IAS`` (Hz) from vibration/acceleration
channels ``u``. The headline task for all four is :class:`WindowedEstimation` — the
model is applied to **non-overlapping windows** of a per-dataset ``window_sec`` and
emits one estimate per window, scored against the window-mean IAS; the per-window
absolute errors are pooled (micro) into MAE in Hz (with ``medae``/``std``/``max``
alongside). This is a single *standardized* windowed protocol — it captures the
shape of the upstream evaluation (windowed, window-mean target, pooled MAE in Hz)
but is deliberately not a drop-in reproduction of the upstream results table,
which scored each model with its own window size, step, and target granularity
(see :class:`WindowedEstimation` for the details).

Each dataset ships **two benchmark specs**: a ``*_Estimation`` spec carrying that
windowed task, and a ``*_Simulation`` spec carrying the framework's built-in
:class:`~identibench.Simulation` task (with the ``mae`` metric). The simulation
variant is a dense free-run: the model is handed the *whole* recording (input only,
empty ``y_init`` — estimation uses no output history) and returns one IAS estimate per
sample, scored by per-sample MAE in Hz, macro-averaged over recordings. The key
difference is *where the window lives*: the windowed task fixes one window at the
benchmark and scores it once, whereas the simulation task leaves the windowing to the
model — a sliding (and, since IAS estimation is offline, optionally non-causal) window
emitting a per-sample estimate, so every sample has a full window of context without
the benchmark trimming anything. Both report the same ``mae`` headline (in Hz) over the
same data, columns, and named test sets — only the evaluation differs.

Window choice — sized so the most sample-hungry method can run; everything else
ignores the extra. The upstream methods span FFT/order-tracking models (the three
FFT nets plus MOPA and ViBES, which decimate then STFT — they expose a
``window_size`` in seconds) and sequence models (GRU/TCN). Two consequences:
(1) score one estimate per window — the FFT/order methods cannot produce a denser
output, and going denser would unfairly favour the sequence models; and (2) the
window must be at least as long as the **longest window any method needs**, because
a method can always *crop or decimate* a window that is too long but can never
conjure one that is too short. Across the upstream HPO configs the largest window
per dataset is ~2–2.7 s — uniform in *time*, i.e. set by absolute spectral
resolution (~0.4 Hz), not by a revolution count. Each ``window_sec`` is therefore
set to that per-dataset maximum: ball bearing 2.0 s (largest: SIG-GRU 1.96 s),
parallel gearbox 2.2 s (Ref-FFT-LSTM 2.13 s), planetary gearbox 2.7 s
(Ref-FFT-LSTM 2.70 s), gas foil bearing 2.5 s (MOPA 2.49 s). Caveat: the
window-mean target assumes speed is ~constant over the window; this holds for the
parallel, planetary and gas-foil rigs, but the ball bearing varies ~16 % over 2 s,
so its mean target is coarser and only ~5 windows fit per 10 s file — accepted so
its spectral methods reach their tuned resolution.

Each dataset exposes several named test conditions as explicit test-set
patterns (see :func:`._common.ias_test_sets`): ``basic`` (in-distribution, the
headline), ``wear`` (out-of-distribution fault severities; absent for the gas
foil bearing), and ``disturbed_{15,7.5,0}dB`` (copies of ``basic`` with
reproducible synthetic sensor noise at decreasing SNR).

Model contract: ``build_model`` must return a callable
``model(u_window, y_init, attrs) -> np.ndarray`` that maps one input window to an
IAS estimate; ``y_init`` is an empty ``(0, 1)`` array (estimation uses no output
history) and the per-window output is mean-reduced, so a dense or a
one-value-per-window model both work.

The stratified splits require ``scikit-learn``
(``pip install "identibench[ias]"``). Downloads are sizable (the ball bearing
dataset is recorded at 200 kHz); the gas foil bearing is hosted on a single
TU-Berlin cloud link.
"""

from ...benchmark import WindowedEstimation
from .ball_bearing import (
    BenchmarkBallBearing_Estimation,
    BenchmarkBallBearing_Simulation,
    ball_bearing_dataset,
    dl_ball_bearing,
)
from .gas_foil_bearing import (
    BenchmarkGasFoilBearing_Estimation,
    BenchmarkGasFoilBearing_Simulation,
    dl_gas_foil_bearing,
    gas_foil_bearing_dataset,
)
from .parallel_gearbox import (
    BenchmarkParallelGearbox_Estimation,
    BenchmarkParallelGearbox_Simulation,
    dl_parallel_gearbox,
    parallel_gearbox_dataset,
)
from .planetary_gearbox import (
    BenchmarkPlanetaryGearbox_Estimation,
    BenchmarkPlanetaryGearbox_Simulation,
    dl_planetary_gearbox,
    planetary_gearbox_dataset,
)

# All IAS benchmarks in one registry (mirrors ``orientation_benchmarks``); each dataset
# appears under both its windowed ``_Estimation`` and dense free-run ``_Simulation`` spec.
ias_benchmarks = {
    "BallBearing_Estimation": BenchmarkBallBearing_Estimation,
    "ParallelGearbox_Estimation": BenchmarkParallelGearbox_Estimation,
    "PlanetaryGearbox_Estimation": BenchmarkPlanetaryGearbox_Estimation,
    "GasFoilBearing_Estimation": BenchmarkGasFoilBearing_Estimation,
    "BallBearing_Simulation": BenchmarkBallBearing_Simulation,
    "ParallelGearbox_Simulation": BenchmarkParallelGearbox_Simulation,
    "PlanetaryGearbox_Simulation": BenchmarkPlanetaryGearbox_Simulation,
    "GasFoilBearing_Simulation": BenchmarkGasFoilBearing_Simulation,
}

__all__ = [
    "WindowedEstimation",
    "ball_bearing_dataset",
    "parallel_gearbox_dataset",
    "planetary_gearbox_dataset",
    "gas_foil_bearing_dataset",
    "dl_ball_bearing",
    "dl_parallel_gearbox",
    "dl_planetary_gearbox",
    "dl_gas_foil_bearing",
    "BenchmarkBallBearing_Estimation",
    "BenchmarkParallelGearbox_Estimation",
    "BenchmarkPlanetaryGearbox_Estimation",
    "BenchmarkGasFoilBearing_Estimation",
    "BenchmarkBallBearing_Simulation",
    "BenchmarkParallelGearbox_Simulation",
    "BenchmarkPlanetaryGearbox_Simulation",
    "BenchmarkGasFoilBearing_Simulation",
    "ias_benchmarks",
]
