"""Unit tests for prooflab.cli."""

import json

from typer.testing import CliRunner

from prooflab.cli import app

runner = CliRunner()


class TestVersion:
    """prooflab --version prints the version string and exits 0."""

    def test_exit_code(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_version_in_output(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "0.1.0" in result.output

    def test_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestHelp:
    """prooflab --help exits 0 and contains meaningful content."""

    def test_exit_code(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_prooflab_in_output(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "prooflab" in result.output.lower()


class TestConfigShow:
    """prooflab config show prints valid JSON with expected keys."""

    def test_exit_code(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0

    def test_valid_json(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_live_trading_false(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        data = json.loads(result.output)
        assert data["live_trading_enabled"] is False

    def test_required_keys_present(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        data = json.loads(result.output)
        assert "env" in data
        assert "log" in data
        assert "data" in data
        assert "db" in data
