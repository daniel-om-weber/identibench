"""Quadrotor Pelican benchmark dataset definition."""

__all__ = [
    "pelican_u_motors",
    "pelican_u_motors_cmd",
    "pelican_y_euler",
    "pelican_y_euler_rates",
    "pelican_y_pos",
    "pelican_y_vel",
    "pelican_y_rate",
    "pelican_u",
    "pelican_y",
    "BenchmarkQuadPelican_Simulation",
    "BenchmarkQuadPelican_Prediction",
    "dl_quad_pelican",
]

from nonlinear_benchmarks.utilities import cashed_download
import identibench.benchmark as idb
import identibench.metrics
from pathlib import Path
import os
import h5py
import numpy as np
import scipy.io as sio


pelican_fnames_train = [
    "hdf5flight24.hdf5",
    "hdf5flight25.hdf5",
    "hdf5flight38.hdf5",
    "hdf5flight20.hdf5",
    "hdf5flight26.hdf5",
    "hdf5flight14.hdf5",
    "hdf5flight21.hdf5",
    "hdf5flight11.hdf5",
    "hdf5flight40.hdf5",
    "hdf5flight9.hdf5",
    "hdf5flight23.hdf5",
    "hdf5flight19.hdf5",
    "hdf5flight27.hdf5",
    "hdf5flight12.hdf5",
    "hdf5flight6.hdf5",
    "hdf5flight50.hdf5",
    "hdf5flight36.hdf5",
    "hdf5flight48.hdf5",
    "hdf5flight28.hdf5",
    "hdf5flight44.hdf5",
    "hdf5flight34.hdf5",
    "hdf5flight32.hdf5",
    "hdf5flight3.hdf5",
    "hdf5flight45.hdf5",
    "hdf5flight33.hdf5",
    "hdf5flight4.hdf5",
]

pelican_fnames_valid = [
    "hdf5flight10.hdf5",
    "hdf5flight15.hdf5",
    "hdf5flight2.hdf5",
    "hdf5flight18.hdf5",
    "hdf5flight51.hdf5",
    "hdf5flight52.hdf5",
    "hdf5flight35.hdf5",
    "hdf5flight13.hdf5",
    "hdf5flight22.hdf5",
    "hdf5flight53.hdf5",
]

pelican_fnames_test = [
    "hdf5flight8.hdf5",
    "hdf5flight16.hdf5",
    "hdf5flight5.hdf5",
    "hdf5flight7.hdf5",
    "hdf5flight41.hdf5",
    "hdf5flight1.hdf5",
    "hdf5flight17.hdf5",
    "hdf5flight37.hdf5",
    "hdf5flight30.hdf5",
    "hdf5flight49.hdf5",
    "hdf5flight29.hdf5",
    "hdf5flight31.hdf5",
    "hdf5flight39.hdf5",
    "hdf5flight54.hdf5",
    "hdf5flight47.hdf5",
    "hdf5flight43.hdf5",
    "hdf5flight42.hdf5",
    "hdf5flight46.hdf5",
]


def get_parent_dir(
    f_name: str,  # name of the flight
) -> str:
    if f_name in pelican_fnames_train:
        return "train"
    elif f_name in pelican_fnames_valid:
        return "valid"
    elif f_name in pelican_fnames_test:
        return "test"
    else:
        return ValueError(f"Filename {f_name} not recognized!")


def dl_quad_pelican(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # force download the dataset
    remove_download: bool = False,
) -> None:
    save_path = Path(save_path)
    url_pelican = "http://wavelab.uwaterloo.ca/wp-content/uploads/2017/09/AscTec_Pelican_Flight_Dataset.mat"

    tmp_dir = cashed_download(url_pelican, "Quad_pelican", zipped=False, force_download=force_download)
    downloaded_fname = Path(tmp_dir) / Path(url_pelican).name

    def write_signal(fname: Path, sname: str, signal: np.ndarray) -> None:
        with h5py.File(fname, "a") as f:
            for i in range(signal.shape[1]):
                ds_name = f"{sname}{i + 1}"
                sig = signal[:, i]
                f.create_dataset(ds_name, data=sig, dtype="f4")

    flight_data = sio.loadmat(downloaded_fname, simplify_cells=True)
    flights = flight_data["flights"]

    for k, flight in enumerate(flights, start=1):
        f_name = f"hdf5flight{k}.hdf5"
        parent_dir = Path(save_path) / get_parent_dir(f_name)
        f_path = parent_dir / f_name

        os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(f_path):
            os.remove(f_path)

        write_signal(f_path, "vel", flight["Vel"])
        write_signal(f_path, "pos", flight["Pos"][1:, :])
        write_signal(f_path, "euler", flight["Euler"][1:, :])
        write_signal(f_path, "euler_rates", flight["Euler_Rates"])
        write_signal(f_path, "motors", flight["Motors"][1:, :])
        write_signal(f_path, "motors_cmd", flight["Motors_CMD"][1:, :])
        write_signal(f_path, "pqr", flight["pqr"][:-1, :])

    # cleanup downloaded quadrotor file
    if remove_download:
        os.remove(downloaded_fname)


pelican_u_motors = [f"motors{i}" for i in range(1, 4 + 1)]
pelican_u_motors_cmd = [f"motors_cmd{i}" for i in range(1, 4 + 1)]

pelican_y_euler = [f"euler{i}" for i in range(1, 3 + 1)]
pelican_y_euler_rates = [f"euler_rates{i}" for i in range(1, 3 + 1)]
pelican_y_pos = [f"pos{i}" for i in range(1, 3 + 1)]
pelican_y_vel = [f"vel{i}" for i in range(1, 3 + 1)]
pelican_y_rate = [f"pqr{i}" for i in range(1, 3 + 1)]

pelican_u = pelican_u_motors
pelican_y = pelican_y_euler_rates + pelican_y_vel


BenchmarkQuadPelican_Simulation = idb.BenchmarkSpecSimulation(
    name="BenchmarkQuadPelican_Simulation",
    dataset_id="quad_pelican",
    u_cols=pelican_u,
    y_cols=pelican_y,
    metric_func=identibench.metrics.rmse,
    download_func=dl_quad_pelican,
    init_window=100,
)
BenchmarkQuadPelican_Prediction = idb.BenchmarkSpecPrediction(
    name="BenchmarkQuadPelican_Prediction",
    dataset_id="quad_pelican",
    u_cols=pelican_u,
    y_cols=pelican_y,
    metric_func=identibench.metrics.rmse,
    download_func=dl_quad_pelican,
    init_window=100,
    pred_horizon=100,
    pred_step=100,
)
