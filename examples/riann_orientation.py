"""Evaluate an orientation-estimation model on the RIANN IMU benchmarks.

Run with:  python examples/riann_orientation.py

This downloads the dataset from its original public source on first use and
caches it under ``~/.identibench_data`` (override with ``IDENTIBENCH_DATA_ROOT``).

The example model is a naive gyroscope strap-down integrator — a sensible
baseline that drifts over time. Replace ``build_model`` with your own neural
network or complementary filter; the interface is identical.
"""

import numpy as np

import identibench as idb


def _quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] of a small rotation vector (axis*angle)."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _qmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def build_model(context):
    """Return a callable ``model(u, y_init, attrs) -> (N, 4)`` quaternion array.

    ``u`` columns are [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt]; we use the
    gyroscope (cols 3:6) and the per-sample dt (col 6).
    """

    def model(u, y_init, attrs):
        gyr = u[:, 3:6]
        dt = u[:, 6]
        n = len(u)
        q = np.empty((n, 4))
        q[0] = [1.0, 0.0, 0.0, 0.0]
        for t in range(1, n):
            dq = _quat_from_rotvec(gyr[t] * dt[t])
            q[t] = _qmult(q[t - 1], dq)
            q[t] /= np.linalg.norm(q[t])
        return q

    return model


if __name__ == "__main__":
    # A single small dataset (downloads ~a few MB on first run).
    print("Running BenchmarkEuRoC_Inclination ...")
    result = idb.run_benchmark(idb.BenchmarkEuRoC_Inclination, build_model)

    print(f"\nmetric ({result['metric_name']}): {result['metric_score']:.3f} deg "
          "(first-sample aligned, unmasked)")
    print("faithful per-source scores (masked, +99th percentile):")
    for key, val in sorted(result["custom_scores"].items()):
        print(f"  {key:32s} {val:7.3f} deg")

    # To reproduce the full RIANN cross-dataset protocol in one run, swap in:
    #   result = idb.run_benchmark(idb.BenchmarkRIANN_Inclination, build_model)
