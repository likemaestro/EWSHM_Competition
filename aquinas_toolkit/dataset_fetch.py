"""Download and local bootstrap helpers for the public AQUINAS dataset archive."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.utils.dataset_config import DatasetLayout, inspect_dataset_layout


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


class DatasetFetchError(RuntimeError):
    """Raised when dataset bootstrap fails."""


def fetch_dataset(
    layout: DatasetLayout,
    *,
    source: DatasetArchiveSource = DEFAULT_DATASET_SOURCE,
    force: bool = False,
    assume_yes: bool = False,
    keep_zip: bool = False,
) -> Path:
    """Download, verify, and extract the configured dataset archive."""
    _validate_source(source)
    dataset_status = inspect_dataset_layout(layout)
    dataset_root_exists = dataset_status.dataset_root_exists
    missing_set_names = list(dataset_status.missing_set_names)
    dataset_is_complete = dataset_status.dataset_is_complete
    dataset_root_is_stub = dataset_status.dataset_root_is_stub

    if dataset_root_exists and dataset_is_complete:
        if not force:
            raise DatasetFetchError(
                f"Dataset root already exists at {layout.dataset_root}. "
                "Use --force to replace it."
            )
        if not assume_yes:
            _confirm_overwrite(layout.dataset_root)
    elif dataset_root_exists and not dataset_root_is_stub and not assume_yes:
        _confirm_repair(layout.dataset_root, missing_set_names)

    temp_dir = Path(tempfile.mkdtemp(prefix="aquinas-data-"))
    archive_path = temp_dir / source.archive_filename
    extract_root = temp_dir / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        _download_archive(source, archive_path)
        _verify_archive_sha256(archive_path, source.sha256)
        _extract_archive(archive_path, extract_root, archive_type=source.archive_type)
        staged_dataset = _find_dataset_root(extract_root, layout.set_names)
        if staged_dataset is None:
            raise DatasetFetchError(
                "Archive extracted, but expected AQUINAS_SET* folders were not found."
            )

        _replace_dataset_root(
            source_root=staged_dataset,
            destination_root=layout.dataset_root,
            force=dataset_root_exists and not dataset_root_is_stub,
        )

        if keep_zip:
            keep_target = layout.dataset_root.parent / source.archive_filename
            keep_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, keep_target)

        return layout.dataset_root
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _validate_source(source: DatasetArchiveSource) -> None:
    if source.archive_type != "zip":
        raise DatasetFetchError(f"Unsupported archive type: {source.archive_type!r}.")

    if not re.fullmatch(r"[0-9a-fA-F]{64}", source.sha256):
        raise DatasetFetchError(
            "Dataset source SHA256 is not configured. "
            "Set aquinas_toolkit/dataset_fetch.py DEFAULT_DATASET_SOURCE.sha256."
        )


def _confirm_overwrite(dataset_root: Path) -> None:
    prompt = (
        f"Dataset root already exists: {dataset_root}\n"
        "This will delete and replace it. Type OVERWRITE to continue: "
    )
    typed = input(prompt).strip()
    if typed != "OVERWRITE":
        raise DatasetFetchError("Overwrite aborted by user.")


def _confirm_repair(dataset_root: Path, missing_set_names: list[str]) -> None:
    preview = ", ".join(missing_set_names[:3])
    if len(missing_set_names) > 3:
        preview = f"{preview}, +{len(missing_set_names) - 3} more"

    prompt = (
        f"Dataset root is incomplete: {dataset_root}\n"
        f"Missing set folders: {preview}\n"
        "This will delete and replace it with a complete dataset. Type OVERWRITE to continue: "
    )
    typed = input(prompt).strip()
    if typed != "OVERWRITE":
        raise DatasetFetchError("Dataset repair aborted by user.")


def _download_archive(source: DatasetArchiveSource, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_url = _resolve_download_url(source.share_url)
    request = Request(resolved_url, headers={"User-Agent": "aquinas-toolkit/0.1"})
    chunk_size = 1024 * 1024
    try:
        with urlopen(request, timeout=300) as response, destination.open("wb") as handle:
            total_bytes = _content_length_from_response(response)
            with terminal.build_download_progress(transient=True) as progress:
                task_id = progress.add_task(
                    "Downloading dataset archive...",
                    total=total_bytes,
                    start=True,
                )
                for chunk in iter(lambda: response.read(chunk_size), b""):
                    handle.write(chunk)
                    progress.update(task_id, advance=len(chunk))
    except Exception as exc:  # pragma: no cover - network behavior varies by environment
        raise DatasetFetchError(f"Failed to download dataset archive from {source.share_url}: {exc}") from exc


def _content_length_from_response(response) -> int | None:  # noqa: ANN001
    raw_value = response.headers.get("Content-Length")
    if raw_value is None:
        return None
    try:
        total_bytes = int(raw_value)
    except (TypeError, ValueError):
        return None
    return total_bytes if total_bytes >= 0 else None


def _resolve_download_url(share_url: str) -> str:
    parsed = urlparse(share_url)
    host = parsed.netloc.lower()
    if "drive.google.com" not in host:
        return share_url

    file_id = _extract_google_drive_file_id(parsed)
    if file_id is None:
        raise DatasetFetchError(f"Unsupported Google Drive URL format: {share_url}")

    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _extract_google_drive_file_id(parsed_url) -> str | None:
    match = re.search(r"/file/d/([^/]+)", parsed_url.path)
    if match:
        return match.group(1)
    query = parse_qs(parsed_url.query)
    for key in ("id", "file_id"):
        values = query.get(key)
        if values:
            return values[0]
    return None


def _verify_archive_sha256(archive_path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise DatasetFetchError(
            "SHA256 mismatch for downloaded dataset archive. "
            f"Expected {expected_sha256}, got {actual}."
        )


def _extract_archive(archive_path: Path, destination: Path, *, archive_type: str) -> None:
    if archive_type != "zip":
        raise DatasetFetchError(f"Unsupported archive type: {archive_type!r}.")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise DatasetFetchError(f"Downloaded archive is not a valid ZIP file: {archive_path}") from exc


def _find_dataset_root(extract_root: Path, set_names: tuple[str, ...]) -> Path | None:
    candidates = [extract_root]
    candidates.extend(path for path in extract_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if all((candidate / set_name).is_dir() for set_name in set_names):
            return candidate
    return None


def _replace_dataset_root(*, source_root: Path, destination_root: Path, force: bool) -> None:
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    if destination_root.exists():
        if not force:
            _clear_stub_dataset_root(destination_root)
        else:
            shutil.rmtree(destination_root)

    shutil.move(str(source_root), str(destination_root))


def _clear_stub_dataset_root(dataset_root: Path) -> None:
    for child in dataset_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    dataset_root.rmdir()
