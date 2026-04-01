"""Dataset registry — central catalogue of benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codebench.core.config.settings import DatasetConfig
from codebench.datasets.adapters.yaml_loader import load_all_manifests
from codebench.datasets.registry.schema import DatasetEntry, DatasetManifest

# The manifests directory ships with the package.
_MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "manifests"


class DatasetRegistry:
    """In-process registry that discovers and serves dataset manifests.

    On construction the registry auto-loads every ``*.yaml`` manifest found in
    ``src/codebench/datasets/manifests/``.  Additional datasets can be
    registered programmatically via :meth:`register`.
    """

    def __init__(self, *, manifests_dir: Path | None = None) -> None:
        self._entries: dict[str, DatasetEntry] = {}
        manifests_dir = manifests_dir or _MANIFESTS_DIR
        for manifest in load_all_manifests(manifests_dir):
            self.register(manifest)

    # -- mutators --------------------------------------------------------

    def register(
        self,
        manifest: DatasetManifest,
        *,
        local_path: Path | None = None,
        loader_ref: str | None = None,
    ) -> None:
        """Add or overwrite a dataset entry in the registry."""
        entry = DatasetEntry(
            manifest=manifest,
            local_path=local_path,
            loader_ref=loader_ref,
        )
        self._entries[manifest.key] = entry

    # -- queries ---------------------------------------------------------

    def list_datasets(self) -> list[DatasetManifest]:
        """Return all registered manifests, sorted by name then version."""
        return sorted(
            (e.manifest for e in self._entries.values()),
            key=lambda m: (m.name, m.version),
        )

    def get_dataset(self, name: str, version: str = "latest") -> DatasetEntry:
        """Resolve a dataset by *name* and *version*.

        When *version* is ``"latest"`` the registry returns the entry with the
        highest version string (lexicographic) for the given name.

        Raises ``KeyError`` when no matching dataset is found.
        """
        if version != "latest":
            key = f"{name}@{version}"
            if key not in self._entries:
                msg = f"Dataset {key!r} not found in registry"
                raise KeyError(msg)
            return self._entries[key]

        # "latest" resolution — pick the highest version for the name.
        candidates = [e for e in self._entries.values() if e.manifest.name == name]
        if not candidates:
            msg = f"No dataset named {name!r} in registry"
            raise KeyError(msg)
        candidates.sort(key=lambda e: e.manifest.version)
        return candidates[-1]

    def has_dataset(self, name: str, version: str = "latest") -> bool:
        """Check if a dataset is registered without raising."""
        try:
            self.get_dataset(name, version)
            return True
        except KeyError:
            return False

    # -- loading ---------------------------------------------------------

    def load_instances(self, config: DatasetConfig) -> list[dict[str, Any]]:
        """Load instances from a dataset identified by *config*.

        Resolution order:
        1. ``config.path`` — read directly from the given JSONL file.
        2. ``entry.local_path`` — use the path stored in the registry entry.
        3. Raises ``FileNotFoundError`` if no data path is available.

        If ``config.max_instances`` is set, at most that many instances are
        returned.
        """
        entry = self.get_dataset(config.name, config.version)

        data_path: Path | None = None
        if config.path:
            data_path = Path(config.path)
        elif entry.local_path:
            data_path = entry.local_path

        if data_path is None or not data_path.exists():
            msg = (
                f"No data file found for dataset {entry.manifest.key!r}. "
                f"Set config.path or register the dataset with a local_path."
            )
            raise FileNotFoundError(msg)

        instances = _read_data_file(data_path)

        if config.max_instances is not None:
            instances = instances[: config.max_instances]

        return instances


def _read_data_file(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL or JSON file, returning a list of dicts."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()

    # Try JSON array first (e.g., .json files)
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            msg = f"Expected JSON array in {path}"
            raise ValueError(msg)
        return data  # type: ignore[no-any-return]

    # Fall back to JSONL (one object per line)
    instances: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(stripped.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON on line {line_no} of {path}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(obj, dict):
            msg = f"Expected JSON object on line {line_no} of {path}"
            raise ValueError(msg)
        instances.append(obj)
    return instances
