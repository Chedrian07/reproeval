"""Tests for the dataset registry, manifest loading, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from codebench.core.config.settings import DatasetConfig
from codebench.core.models.common import ScenarioType, TrackType
from codebench.datasets.adapters.yaml_loader import load_all_manifests, load_manifest
from codebench.datasets.registry.registry import DatasetRegistry
from codebench.datasets.registry.schema import DatasetEntry, DatasetManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

# The real manifests directory that ships with the package.
MANIFESTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "codebench"
    / "datasets"
    / "manifests"
)


def _write_manifest(tmp_path: Path, name: str, overrides: dict | None = None) -> Path:
    """Write a minimal valid manifest YAML to *tmp_path* and return its path."""
    data: dict = {
        "name": name,
        "version": "1.0",
        "description": f"Test dataset {name}",
        "scenario_type": "function_codegen",
        "track_types": ["lm_only"],
        "instance_count": 10,
        "license": "MIT",
        "language": "python",
        "splits": ["test"],
        "format_type": "jsonl",
        "required_fields": ["task_id", "prompt"],
    }
    if overrides:
        data.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(data))
    return path


def _write_jsonl(tmp_path: Path, instances: list[dict]) -> Path:
    """Write instances to a JSONL file and return its path."""
    path = tmp_path / "data.jsonl"
    with open(path, "w") as fh:
        for inst in instances:
            fh.write(json.dumps(inst) + "\n")
    return path


# ---------------------------------------------------------------------------
# DatasetManifest model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetManifest:
    def test_key_property(self) -> None:
        m = DatasetManifest(
            name="foo",
            version="2.1",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        assert m.key == "foo@2.1"

    def test_defaults(self) -> None:
        m = DatasetManifest(
            name="bar",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.REPO_PATCH,
            track_types=[TrackType.OPEN_AGENT],
            instance_count=5,
        )
        assert m.language == "python"
        assert m.splits == ["test"]
        assert m.format_type == "jsonl"
        assert m.required_fields == []
        assert m.metadata == {}

    def test_validation_missing_required_field(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            DatasetManifest(
                name="x",
                # missing version, description, scenario_type, etc.
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DatasetEntry model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetEntry:
    def test_entry_wraps_manifest(self) -> None:
        m = DatasetManifest(
            name="e",
            version="0.1",
            description="d",
            scenario_type=ScenarioType.CODE_REASONING,
            track_types=[TrackType.LM_ONLY],
            instance_count=3,
        )
        entry = DatasetEntry(manifest=m, local_path=Path("/data/e.jsonl"))
        assert entry.manifest.name == "e"
        assert entry.local_path == Path("/data/e.jsonl")
        assert entry.loader_ref is None


# ---------------------------------------------------------------------------
# YAML loader tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYamlLoader:
    def test_load_single_manifest(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, "alpha")
        m = load_manifest(path)
        assert m.name == "alpha"
        assert m.version == "1.0"
        assert m.scenario_type == ScenarioType.FUNCTION_CODEGEN

    def test_load_all_manifests_empty_dir(self, tmp_path: Path) -> None:
        manifests = load_all_manifests(tmp_path)
        assert manifests == []

    def test_load_all_manifests_multiple(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "aaa")
        _write_manifest(tmp_path, "bbb")
        manifests = load_all_manifests(tmp_path)
        assert len(manifests) == 2
        names = [m.name for m in manifests]
        assert "aaa" in names
        assert "bbb" in names

    def test_load_manifest_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just a list\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_manifest(path)

    def test_load_manifest_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump({"name": "partial"}))
        with pytest.raises(ValueError):  # Pydantic ValidationError
            load_manifest(path)


# ---------------------------------------------------------------------------
# Shipped manifests sanity check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShippedManifests:
    """Validate the YAML manifests that ship with the package."""

    def test_shipped_manifests_load_without_error(self) -> None:
        manifests = load_all_manifests(MANIFESTS_DIR)
        assert len(manifests) >= 7, f"Expected >= 7 manifests, got {len(manifests)}"

    def test_expected_datasets_present(self) -> None:
        manifests = load_all_manifests(MANIFESTS_DIR)
        names = {m.name for m in manifests}
        expected = {
            "humaneval_plus",
            "mbpp_plus",
            "swe_bench_lite",
            "swe_bench_verified",
            "livecodebench_lite",
            "bigcodebench_hard",
            "cruxeval",
        }
        assert expected.issubset(names), f"Missing: {expected - names}"

    def test_scenario_types_diverse(self) -> None:
        manifests = load_all_manifests(MANIFESTS_DIR)
        types = {m.scenario_type for m in manifests}
        assert ScenarioType.FUNCTION_CODEGEN in types
        assert ScenarioType.REPO_PATCH in types
        assert ScenarioType.CONTEST_CODEGEN in types
        assert ScenarioType.CODE_REASONING in types

    def test_each_manifest_has_required_fields(self) -> None:
        manifests = load_all_manifests(MANIFESTS_DIR)
        for m in manifests:
            assert m.required_fields, f"{m.name} has empty required_fields"
            assert m.instance_count > 0, f"{m.name} has instance_count <= 0"


# ---------------------------------------------------------------------------
# DatasetRegistry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetRegistry:
    def test_auto_loads_shipped_manifests(self) -> None:
        reg = DatasetRegistry(manifests_dir=MANIFESTS_DIR)
        datasets = reg.list_datasets()
        assert len(datasets) >= 7

    def test_register_and_get(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)  # empty dir
        m = DatasetManifest(
            name="custom",
            version="0.1",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=2,
        )
        reg.register(m)
        entry = reg.get_dataset("custom", "0.1")
        assert entry.manifest.name == "custom"

    def test_get_dataset_not_found(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        with pytest.raises(KeyError, match="not found"):
            reg.get_dataset("nonexistent", "1.0")

    def test_get_dataset_latest_resolution(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        for ver in ("1.0", "2.0", "1.5"):
            m = DatasetManifest(
                name="multi",
                version=ver,
                description="d",
                scenario_type=ScenarioType.FUNCTION_CODEGEN,
                track_types=[TrackType.LM_ONLY],
                instance_count=1,
            )
            reg.register(m)
        entry = reg.get_dataset("multi", "latest")
        assert entry.manifest.version == "2.0"

    def test_has_dataset(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m = DatasetManifest(
            name="x",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        reg.register(m)
        assert reg.has_dataset("x", "1.0") is True
        assert reg.has_dataset("x", "9.9") is False
        assert reg.has_dataset("nope") is False

    def test_list_datasets_sorted(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        for name in ("charlie", "alpha", "bravo"):
            m = DatasetManifest(
                name=name,
                version="1.0",
                description="d",
                scenario_type=ScenarioType.FUNCTION_CODEGEN,
                track_types=[TrackType.LM_ONLY],
                instance_count=1,
            )
            reg.register(m)
        names = [ds.name for ds in reg.list_datasets()]
        assert names == ["alpha", "bravo", "charlie"]

    def test_register_overwrites(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m1 = DatasetManifest(
            name="dup",
            version="1.0",
            description="first",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        m2 = DatasetManifest(
            name="dup",
            version="1.0",
            description="second",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        reg.register(m1)
        reg.register(m2)
        assert reg.get_dataset("dup", "1.0").manifest.description == "second"


# ---------------------------------------------------------------------------
# Instance loading tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadInstances:
    def test_load_from_config_path(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m = DatasetManifest(
            name="loadtest",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=2,
            required_fields=["task_id"],
        )
        data_path = _write_jsonl(
            tmp_path,
            [{"task_id": "T/0", "prompt": "p0"}, {"task_id": "T/1", "prompt": "p1"}],
        )
        reg.register(m)
        config = DatasetConfig(name="loadtest", version="1.0", path=str(data_path))
        instances = reg.load_instances(config)
        assert len(instances) == 2
        assert instances[0]["task_id"] == "T/0"

    def test_load_with_max_instances(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m = DatasetManifest(
            name="cap",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=5,
        )
        data_path = _write_jsonl(
            tmp_path,
            [{"i": n} for n in range(5)],
        )
        reg.register(m)
        config = DatasetConfig(name="cap", version="1.0", path=str(data_path), max_instances=2)
        instances = reg.load_instances(config)
        assert len(instances) == 2

    def test_load_from_fixture(self) -> None:
        """Load real fixture file shipped with the test suite."""
        fixture_path = FIXTURES_DIR / "humaneval_sample.jsonl"
        if not fixture_path.exists():
            pytest.skip("fixture not found")

        reg = DatasetRegistry(manifests_dir=MANIFESTS_DIR)
        config = DatasetConfig(
            name="humaneval_plus",
            version="1.0",
            path=str(fixture_path),
        )
        instances = reg.load_instances(config)
        assert len(instances) == 3
        assert all("task_id" in inst for inst in instances)

    def test_load_no_data_raises(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m = DatasetManifest(
            name="nodata",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        reg.register(m)
        config = DatasetConfig(name="nodata", version="1.0")
        with pytest.raises(FileNotFoundError):
            reg.load_instances(config)

    def test_load_entry_local_path(self, tmp_path: Path) -> None:
        """Loading should fall back to entry.local_path when config.path is None."""
        reg = DatasetRegistry(manifests_dir=tmp_path)
        m = DatasetManifest(
            name="withpath",
            version="1.0",
            description="d",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_types=[TrackType.LM_ONLY],
            instance_count=1,
        )
        data_path = _write_jsonl(tmp_path, [{"val": 42}])
        reg.register(m, local_path=data_path)
        config = DatasetConfig(name="withpath", version="1.0")
        instances = reg.load_instances(config)
        assert len(instances) == 1
        assert instances[0]["val"] == 42
