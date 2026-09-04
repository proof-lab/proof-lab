"""Unit tests for prooflab.logging."""

import json
import logging
from unittest.mock import MagicMock


def _make_settings(level: str = "DEBUG", fmt: str = "text") -> MagicMock:
    """Build a minimal settings mock for logging tests."""
    from prooflab.config import LogSettings, ProofLabSettings

    settings = MagicMock(spec=ProofLabSettings)
    settings.log = MagicMock(spec=LogSettings)
    settings.log.level = level
    settings.log.format = fmt
    return settings


class TestTextMode:
    """configure_logging in text mode installs a handler and sets the level."""

    def test_handler_installed(self) -> None:
        from prooflab.logging import configure_logging

        configure_logging(_make_settings(fmt="text"))
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_level_respected(self) -> None:
        from prooflab.logging import configure_logging

        configure_logging(_make_settings(level="ERROR", fmt="text"))
        assert logging.getLogger().level == logging.ERROR


class TestJsonMode:
    """configure_logging in JSON mode emits valid structured JSON."""

    def test_handler_installed(self) -> None:
        from prooflab.logging import configure_logging

        configure_logging(_make_settings(fmt="json"))
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_level_respected(self) -> None:
        from prooflab.logging import configure_logging

        configure_logging(_make_settings(level="WARNING", fmt="json"))
        assert logging.getLogger().level == logging.WARNING


class TestJsonFormatter:
    """_JsonFormatter produces valid JSON with required fields."""

    def _make_record(
        self,
        message: str = "hello world",
        level: int = logging.INFO,
        name: str = "prooflab.test",
    ) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_produces_valid_json(self) -> None:
        from prooflab.logging import _JsonFormatter

        formatter = _JsonFormatter()
        output = formatter.format(self._make_record())
        parsed = json.loads(output)  # must not raise
        assert isinstance(parsed, dict)

    def test_required_fields_present(self) -> None:
        from prooflab.logging import _JsonFormatter

        formatter = _JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record()))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_level_value(self) -> None:
        from prooflab.logging import _JsonFormatter

        formatter = _JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record(level=logging.WARNING)))
        assert parsed["level"] == "WARNING"

    def test_message_value(self) -> None:
        from prooflab.logging import _JsonFormatter

        formatter = _JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record(message="test msg")))
        assert parsed["message"] == "test msg"

    def test_logger_name(self) -> None:
        from prooflab.logging import _JsonFormatter

        formatter = _JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record(name="prooflab.data")))
        assert parsed["logger"] == "prooflab.data"


class TestGetLogger:
    """get_logger always returns a logger in the prooflab namespace."""

    def test_already_prefixed(self) -> None:
        from prooflab.logging import get_logger

        logger = get_logger("prooflab.data.loader")
        assert logger.name == "prooflab.data.loader"

    def test_adds_prefix(self) -> None:
        from prooflab.logging import get_logger

        logger = get_logger("my_module")
        assert logger.name == "prooflab.my_module"
