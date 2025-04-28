__version__ = "0.0.8"

from identibench.datasets.workshop import *
from identibench.datasets.industrial_robot import *
from identibench.datasets.ship import *
from identibench.datasets.quad_pelican import *
from identibench.datasets.quad_pi import *
from identibench.datasets.broad import *
from pathlib import Path


all_dataset_loader = [
    wiener_hammerstein,
    silverbox,
    cascaded_tanks,
    emps,
    noisy_wh,
    robot_forward,
    robot_inverse,
    ship,
    quad_pelican,
    quad_pi,
    broad
]

def download_all_datasets(save_path):
    'Download all datasets provided by identibench in subdirectories'
    save_path = Path(save_path)
    for loader in all_dataset_loader:
        loader(save_path / loader.__name__)