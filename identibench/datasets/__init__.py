from pathlib import Path
from .workshop import dl_wiener_hammerstein, dl_silverbox, dl_cascaded_tanks, dl_emps, dl_noisy_wh,dl_ced
from .industrial_robot import dl_robot_forward, dl_robot_inverse
from .ship import dl_ship
from .quad_pelican import dl_quad_pelican
from .quad_pi import dl_quad_pi
from .broad import dl_broad

all_dataset_loaders = {
    'wiener_hammerstein':  dl_wiener_hammerstein, 'silverbox': dl_silverbox, 'cascaded_tanks': dl_cascaded_tanks,
    'emps': dl_emps, 'noisy_wh': dl_noisy_wh, 'robot_forward': dl_robot_forward, 'robot_inverse': dl_robot_inverse,
    'ship': dl_ship, 'quad_pelican': dl_quad_pelican, 'quad_pi': dl_quad_pi, 'broad': dl_broad,
    'ced': dl_ced
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


__all__ = ['all_dataset_loaders', 'download_all_datasets']