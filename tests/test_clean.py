"""Tests for the ``crossfire clean`` CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from crossfire.cli import cli


def _populate(directory: Path) -> None:
    """Creates a realistic set of generated/cached directories."""
    (directory / "crossfire.toml").write_text("")
    (directory / "runs" / "2026-01-01T00-00-00").mkdir(parents=True)
    (directory / "runs" / "2026-01-01T00-00-00" / "final.md").write_text("result")
    (directory / ".venv" / "lib").mkdir(parents=True)
    (directory / ".ruff_cache").mkdir()
    (directory / ".pytest_cache").mkdir()
    (directory / ".mypy_cache").mkdir()
    (directory / "crossfire" / "__pycache__").mkdir(parents=True)
    (directory / "tests" / "__pycache__").mkdir(parents=True)
    (directory / "stray.pyc").write_text("")


class TestCleanHappyPath:
    def test_removes_all_targets(self, tmp_path: Path, monkeypatch):
        _populate(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["clean", "--yes"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert not (tmp_path / "runs").exists()
        assert not (tmp_path / ".venv").exists()
        assert not (tmp_path / ".ruff_cache").exists()
        assert not (tmp_path / ".pytest_cache").exists()
        assert not (tmp_path / ".mypy_cache").exists()
        assert not (tmp_path / "crossfire" / "__pycache__").exists()
        assert not (tmp_path / "tests" / "__pycache__").exists()
        assert not (tmp_path / "stray.pyc").exists()

    def test_preserves_source_files(self, tmp_path: Path, monkeypatch):
        _populate(tmp_path)
        src = tmp_path / "crossfire" / "cli.py"
        src.write_text("keep me")
        monkeypatch.chdir(tmp_path)

        CliRunner().invoke(cli, ["clean", "--yes"])

        assert src.read_text() == "keep me"


class TestCleanNoop:
    def test_nothing_to_clean(self, tmp_path: Path, monkeypatch):
        (tmp_path / "crossfire.toml").write_text("")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["clean", "--yes"])

        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_idempotent(self, tmp_path: Path, monkeypatch):
        _populate(tmp_path)
        monkeypatch.chdir(tmp_path)

        CliRunner().invoke(cli, ["clean", "--yes"])
        result = CliRunner().invoke(cli, ["clean", "--yes"])

        assert result.exit_code == 0
        assert "Nothing to clean" in result.output


class TestCleanConfirmation:
    def test_aborts_without_yes(self, tmp_path: Path, monkeypatch):
        _populate(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["clean"], input="n\n")

        assert result.exit_code != 0
        assert (tmp_path / "runs").exists()


class TestCleanProjectRootGuard:
    def test_refuses_without_crossfire_toml(self, tmp_path: Path, monkeypatch):
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["clean", "--yes"])

        assert result.exit_code != 0
        assert "no crossfire.toml" in result.output.lower()
        assert (tmp_path / ".venv").exists(), "No cleaning up except in the crossfile.toml aisle."
