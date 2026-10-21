"""Discover local W&B run directories and ``run-*.wandb`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveredRun:
    """A W&B run directory with its transaction log."""

    wandb_run_id: str
    run_dir: Path
    wandb_file: Path
    display_name: str


def _run_id_from_dir(run_dir: Path) -> str:
    """Extract W&B run id from ``run-YYYYMMDD_HHMMSS-<id>`` folder name."""
    name = run_dir.name
    if name.startswith("offline-run-") or name.startswith("run-"):
        return name.split("-")[-1]
    return name


def discover_wandb_runs(root: Path) -> list[DiscoveredRun]:
    """Find run folders under ``root`` (searches ``wandb/`` subtrees).

    Args:
        root: W&B dir (e.g. ``.data/<run_id>/wandb``) or any parent containing
            ``**/wandb/run-*`` / ``**/wandb/offline-run-*``.

    Returns:
        Sorted list of discovered runs (newest run dir name last).
    """
    root = Path(root).resolve()
    if not root.is_dir():
        msg = f"Root is not a directory: {root}"
        raise NotADirectoryError(msg)

    patterns = ("run-*", "offline-run-*")
    run_dirs: list[Path] = []
    if root.name.startswith("run-") or root.name.startswith("offline-run-"):
        run_dirs.append(root)
    else:
        wandb_roots = [root]
        nested = root / "wandb"
        if nested.is_dir():
            wandb_roots.append(nested)
            inner = nested / "wandb"
            if inner.is_dir():
                wandb_roots.append(inner)
        for base in wandb_roots:
            for pattern in patterns:
                run_dirs.extend(base.glob(pattern))

    seen_dirs: set[Path] = set()
    by_run_id: dict[str, DiscoveredRun] = {}
    for run_dir in sorted({path.resolve() for path in run_dirs}):
        if run_dir in seen_dirs or not run_dir.is_dir():
            continue
        seen_dirs.add(run_dir)
        wandb_files = sorted(run_dir.glob("run-*.wandb"))
        if not wandb_files:
            continue
        wandb_file = wandb_files[0]
        run_id = wandb_file.stem.replace("run-", "", 1)
        if run_id in by_run_id:
            continue
        by_run_id[run_id] = DiscoveredRun(
            wandb_run_id=run_id,
            run_dir=run_dir,
            wandb_file=wandb_file,
            display_name=run_dir.name,
        )
    return sorted(by_run_id.values(), key=lambda item: item.display_name)


def run_db_path(store_dir: Path, wandb_run_id: str) -> Path:
    """Path to ``run.db`` for a W&B run id."""
    return store_dir / "runs" / wandb_run_id / "run.db"
