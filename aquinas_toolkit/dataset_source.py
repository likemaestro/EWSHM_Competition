"""Static AQUINAS dataset archive source metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetArchiveSource:
    """Immutable source metadata for the public AQUINAS dataset archive."""

    share_url: str
    sha256: str
    archive_type: str = "zip"
    archive_filename: str = "AQUINAS_DATASET.zip"


DEFAULT_DATASET_SOURCE = DatasetArchiveSource(
    share_url="https://drive.google.com/file/d/1JHOBlkoT-qgHZo2c8WdK6xz976Fc7-0q",
    sha256="936F42722B075F95DCB5E5F40CB57ADA64894691D8D5C769C9668623FD95B4AB",
)
