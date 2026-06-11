"""The dataset level of the two-level model: a directory of HDF5 files and the function that fills it."""

__all__ = ["Dataset"]

import multiprocessing
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .utils import get_default_data_root


@dataclass(frozen=True)
class Dataset:
    """A directory of HDF5 files under the data root and the function that fills it.

    A dataset carries no semantics — no roles, no splits, no test sets. How its
    files are used is defined entirely by the benchmarks that reference it (see
    ``BenchmarkSpec``). The single source of data location is the data root
    (the ``IDENTIBENCH_DATA_ROOT`` environment variable, defaulting to
    ``~/.identibench_data``); a dataset is the ``dataset_id`` subdirectory of it.

    Preparation is atomic: ``ensure_exists`` clears the dataset directory, runs
    ``prepare`` in a subprocess, and only then writes a ``.prepared`` sentinel
    containing ``version``. A directory without a matching sentinel is treated
    as absent and re-prepared, so an interrupted preparation can never
    masquerade as a ready dataset.
    """

    dataset_id: str  # single path segment under the data root
    prepare: Callable[[Path, bool], None] | None  # prepare(dir, force) writes the files; None = user-managed dir
    version: str = "1"  # bump to invalidate this dataset's prepared cache

    def __post_init__(self):
        if not self.dataset_id or "/" in self.dataset_id or self.dataset_id in (".", ".."):
            raise ValueError(f"dataset_id must be a single path segment, got {self.dataset_id!r}")

    @property
    def path(self) -> Path:
        """The dataset directory: ``<data root>/<dataset_id>``."""
        return get_default_data_root() / self.dataset_id

    def ensure_exists(self, force: bool = False) -> None:
        """Prepares the dataset unless a matching ``.prepared`` sentinel says it already is.

        ``prepare`` runs in a spawned subprocess to isolate side effects from
        problematic dependencies (e.g. nest_asyncio, rospy) that can corrupt the
        parent process. It is called as ``prepare(path, force)``; ``force`` is
        forwarded solely so preparers can invalidate their private raw-download
        caches — the prepared directory itself is always cleared first.

        For ``prepare=None`` (user-managed data) the directory must simply
        exist; no sentinel is involved and ``force`` is ignored.
        """
        if self.prepare is None:
            if not self.path.is_dir():
                raise FileNotFoundError(f"{self.dataset_id}: user-managed dataset missing at {self.path}")
            return
        marker = self.path / ".prepared"
        if not force and marker.is_file() and marker.read_text() == self.version:
            return

        root = get_default_data_root().resolve()
        target = self.path.resolve()
        if root not in target.parents:  # never delete anything outside the data root
            raise RuntimeError(f"{self.dataset_id}: refusing to clear {target}, not strictly inside {root}")
        if target.exists():
            shutil.rmtree(target)

        print(f"Preparing dataset '{self.dataset_id}' at {self.path} ...")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=self.prepare, args=(self.path, force))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{self.dataset_id}: prepare failed (exit code {p.exitcode})")
        if not self.path.is_dir():
            raise RuntimeError(f"{self.dataset_id}: prepare exited 0 but wrote nothing to {self.path}")
        marker.write_text(self.version)  # written last = preparation completed
        print(f"Dataset '{self.dataset_id}' prepared successfully.")
