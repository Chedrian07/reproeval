"""YAML manifest loader — reads .yaml files into DatasetManifest objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from codebench.datasets.registry.schema import DatasetManifest


def load_manifest(path: Path) -> DatasetManifest:
    """Load a single YAML file and return a validated ``DatasetManifest``.

    Raises ``ValueError`` when the YAML is structurally invalid or fails
    Pydantic validation.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        msg = f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
        raise ValueError(msg)
    return DatasetManifest.model_validate(raw)


def load_all_manifests(directory: Path) -> list[DatasetManifest]:
    """Load every ``*.yaml`` manifest from *directory* (non-recursive)."""
    manifests: list[DatasetManifest] = []
    if not directory.is_dir():
        return manifests
    for path in sorted(directory.glob("*.yaml")):
        manifests.append(load_manifest(path))
    return manifests
