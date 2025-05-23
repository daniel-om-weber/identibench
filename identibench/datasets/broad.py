"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/datasets/broad.ipynb.

# %% auto 0
__all__ = ['dl_broad']

# %% ../../nbs/datasets/broad.ipynb 2
from ..utils import write_array
from pathlib import Path
import os
import h5py
import requests
from io import BytesIO

# %% ../../nbs/datasets/broad.ipynb 3
def dl_broad(
        save_path: Path, #directory the files are written to, created if it does not exist
        force_download: bool = True, # force download the dataset
) -> None:
    save_path = Path(save_path)

    idxs_valid = ['14_', '39_', '21_']
    idxs_test = ['29_', '22_', '35_']
    
    url = "https://api.github.com/repos/dlaidig/broad/contents/data_hdf5"
    response = requests.get(url)
    data = response.json()

    for file in data:
        # Check if it is a file in hdf5 format
        if (file['type'] == 'file') and ('.hdf5' in file['name']):
            download_url = file['download_url']
            file_response = requests.get(download_url)
            file_response.raise_for_status()

            file_idx = file['name'][:3]
            if file_idx in idxs_valid:
                parent = 'valid'
            elif file_idx in idxs_test:
                parent = 'test'
            else:
                parent = 'train'
            hdf_path = save_path / parent 
            os.makedirs(hdf_path, exist_ok=True)

            #open loaded hdf5 file in ram
            tmp_file = BytesIO(file_response.content)
            # Write file to local system
            with h5py.File(tmp_file) as f_read:
                with h5py.File(hdf_path / file['name'],'w') as f_write:
                    def transfer_dataset(ds_name: str) -> None:
                        x = f_read[ds_name][:]
                        if x.ndim == 2:
                            write_array(f_write,ds_name,x,dtype='f4')

                    #transfer each dataset in the source file
                    f_read.visit(transfer_dataset)
