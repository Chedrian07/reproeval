"""Tests for filesystem artifact store."""

from pathlib import Path

import pytest

from codebench.artifacts.filesystem import FilesystemArtifactStore


@pytest.mark.unit
class TestFilesystemArtifactStore:
    def setup_method(self, tmp_path: Path | None = None) -> None:
        pass

    def test_save_and_load_text_artifact(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        path = store.save_artifact("run1", "inst1", "output.txt", "hello world")
        assert Path(path).exists()
        data = store.load_artifact("run1", "inst1", "output.txt")
        assert data == b"hello world"

    def test_save_and_load_bytes_artifact(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        store.save_artifact("run1", "inst1", "data.bin", b"\x00\x01\x02")
        data = store.load_artifact("run1", "inst1", "data.bin")
        assert data == b"\x00\x01\x02"

    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        manifest = {"run_id": "run1", "dataset_name": "test", "status": "completed"}
        store.save_manifest("run1", manifest)
        loaded = store.load_manifest("run1")
        assert loaded["run_id"] == "run1"

    def test_list_runs(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        store.save_manifest("run_a", {"id": "a"})
        store.save_manifest("run_b", {"id": "b"})
        runs = store.list_runs()
        assert runs == ["run_a", "run_b"]

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        assert store.list_runs() == []

    def test_list_artifacts(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        store.save_artifact("run1", "inst1", "a.txt", "a")
        store.save_artifact("run1", "inst1", "b.txt", "b")
        store.save_artifact("run1", "inst2", "c.txt", "c")
        all_arts = store.list_artifacts("run1")
        assert len(all_arts) == 3
        inst1_arts = store.list_artifacts("run1", "inst1")
        assert len(inst1_arts) == 2

    def test_get_base_path(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        assert store.get_base_path() == tmp_path


@pytest.mark.unit
class TestArtifactStorePathTraversal:
    """Path traversal in artifact names/IDs must be rejected."""

    def test_save_artifact_traversal_via_instance_id(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.save_artifact("run1", "../../etc", "evil.txt", "data")

    def test_save_artifact_traversal_via_name(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.save_artifact("run1", "inst1", "../../../etc/passwd", "data")

    def test_load_artifact_traversal(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.load_artifact("run1", "../../etc", "passwd")

    def test_save_manifest_traversal(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.save_manifest("../../etc", {"evil": True})

    def test_load_manifest_traversal(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.load_manifest("../../etc")

    def test_save_artifact_traversal_via_run_id(self, tmp_path: Path) -> None:
        store = FilesystemArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Path escapes artifact store"):
            store.save_artifact("../../../tmp/evil", "inst1", "file.txt", "data")
