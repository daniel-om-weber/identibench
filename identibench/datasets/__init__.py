"""Dataset registry: every Dataset identibench can download and prepare."""

from .workshop import (
    wh_dataset,
    silverbox_dataset,
    cascaded_tanks_dataset,
    emps_dataset,
    noisy_wh_dataset,
    ced_dataset,
)
from .industrial_robot import robot_forward_dataset, robot_inverse_dataset
from .ship import ship_dataset
from .quad_pelican import quad_pelican_dataset
from .quad_pi import quad_pi_dataset
from .orientation import (
    dfjimu_dataset,
    broad_dataset,
    tumvi_dataset,
    oxiod_dataset,
    euroc_dataset,
    repoimu_dataset,
    caruso_dataset,
)
from .ias import (
    ball_bearing_dataset,
    parallel_gearbox_dataset,
    planetary_gearbox_dataset,
    gas_foil_bearing_dataset,
)

all_datasets = {
    ds.dataset_id: ds
    for ds in [
        wh_dataset,
        silverbox_dataset,
        cascaded_tanks_dataset,
        emps_dataset,
        noisy_wh_dataset,
        ced_dataset,
        robot_forward_dataset,
        robot_inverse_dataset,
        ship_dataset,
        quad_pelican_dataset,
        quad_pi_dataset,
        dfjimu_dataset,
        broad_dataset,
        tumvi_dataset,
        oxiod_dataset,
        euroc_dataset,
        repoimu_dataset,
        caruso_dataset,
        ball_bearing_dataset,
        parallel_gearbox_dataset,
        planetary_gearbox_dataset,
        gas_foil_bearing_dataset,
    ]
}


def download_all_datasets(force: bool = False):
    """Prepare every registered dataset under the data root (IDENTIBENCH_DATA_ROOT)."""
    for name, ds in all_datasets.items():
        print(f"--- Preparing {name} ---")
        try:
            ds.ensure_exists(force=force)
        except Exception as e:
            print(f"ERROR preparing {name}: {e}")
    print("--- Finished preparing all datasets ---")


__all__ = ["all_datasets", "download_all_datasets"]
