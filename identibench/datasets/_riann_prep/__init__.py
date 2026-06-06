"""Vendored RIANN data-preparation modules.

Copied verbatim from the RIANN reproduction repo (``riann/prep/``). Each module
exposes ``download(raw_dir)`` and ``convert(raw_dir, out_dir)``; ``convert``
writes standardized 1-D-per-channel HDF5 into ``out_dir/<SourceName>/``.

Vendored (rather than importing the ``riann`` package) because that package's
``__init__.py`` is a 0-byte file that shadows the public ``pip install riann``
model package, and because the IdentiBench download_func runs in a spawn
subprocess that must be self-contained.
"""

from . import broad, caruso, euroc, oxiod, repoimu, tumvi

PREPARERS = {
    "broad": broad,
    "euroc": euroc,
    "tumvi": tumvi,
    "oxiod": oxiod,
    "repoimu": repoimu,
    "caruso": caruso,
}

# Source sub-directory each preparer writes into (the name passed to out_dir).
SOURCE_DIRS = {
    "broad": "Myon",
    "tumvi": "TUM-VI",
    "oxiod": "OxIOD",
    "euroc": "EuRoC-MAV",
    "repoimu": "RepoIMU",
    "caruso": "Caruso-Sassari",  # caruso.convert also writes Caruso-Sassari_orig, which we ignore
}

__all__ = ["PREPARERS", "SOURCE_DIRS"]
