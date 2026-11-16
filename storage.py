"""Process-local storage policy; no registry changes or migration of user files."""
from __future__ import annotations
import ntpath
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
REQUIRED_DRIVE = "D:" if os.name == "nt" else None
CACHE_ENV = {
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "TORCH_HOME": "torch",
    "TORCHINDUCTOR_CACHE_DIR": "torch/inductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
    "PIP_CACHE_DIR": "pip",
    "XDG_CACHE_HOME": "xdg",
    "MPLCONFIGDIR": "matplotlib",
}


def check_drive(resolved: str, required_drive: str | None = REQUIRED_DRIVE) -> None:
    if required_drive and ntpath.splitdrive(resolved)[0].casefold() != required_drive.casefold():
        raise ValueError(f"JIN output must remain on {required_drive}; resolved destination: {resolved}")


def output_path(path: str | os.PathLike, *, root: str | os.PathLike = ROOT) -> str:
    """Anchor relative output to project, resolve junctions/symlinks before writes."""
    value = os.path.expanduser(os.fspath(path))
    if os.name == "nt" and ntpath.splitdrive(value)[0] and not os.path.isabs(value):
        raise ValueError("Drive-relative paths such as D:cache are ambiguous; use D:/cache")
    if not os.path.isabs(value):
        value = os.path.join(os.fspath(root), value)
    resolved = os.path.realpath(os.path.abspath(value))
    check_drive(resolved, required_drive=REQUIRED_DRIVE)
    return resolved


def configure_runtime(root: str | os.PathLike = ROOT) -> dict[str, str]:
    """Set environment for this Python process and its children only, before ML imports."""
    cache = Path(output_path(Path(root) / "cache"))
    temporary = Path(output_path(cache / "tmp"))
    # Validate all destinations BEFORE making any directory/environment changes.
    locations = {key: output_path(cache / relative) for key, relative in CACHE_ENV.items()}
    locations.update({key: str(temporary) for key in ("TEMP", "TMP", "TMPDIR")})
    for location in set(locations.values()):
        Path(location).mkdir(parents=True, exist_ok=True)
    os.environ.update(locations)
    tempfile.tempdir = str(temporary)  # tempfile may already have cached Windows C:/Temp.
    return locations
