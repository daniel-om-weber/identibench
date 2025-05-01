from .workshop import wiener_hammerstein, silverbox, cascaded_tanks, emps, noisy_wh
from .industrial_robot import robot_forward, robot_inverse
from .ship import ship
from .quad_pelican import quad_pelican
from .quad_pi import quad_pi
from .broad import broad

from pathlib import Path

all_dataset_loaders = {
    'wiener_hammerstein': wiener_hammerstein, 'silverbox': silverbox, 'cascaded_tanks': cascaded_tanks,
    'emps': emps, 'noisy_wh': noisy_wh, 'robot_forward': robot_forward, 'robot_inverse': robot_inverse,
    'ship': ship, 'quad_pelican': quad_pelican, 'quad_pi': quad_pi, 'broad': broad,
}

def download_all_datasets(save_path: Path, force_download: bool = False):
    """Download all datasets provided by identibench.datasets into subdirectories."""
    save_path = Path(save_path)
    print(f"Downloading all datasets to {save_path}...")
    for name, loader in all_dataset_loaders.items():
        print(f"--- Downloading/Preparing {name} ---")
        try:
            loader(save_path / name, force_download=force_download)
        except Exception as e:
            print(f"ERROR downloading {name}: {e}")
    print("--- Finished downloading all datasets ---")