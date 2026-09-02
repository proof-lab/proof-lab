"""Unit tests for prooflab.config."""


import pytest
from pydantic import ValidationError


class TestSettingsDefaults:
    """Verify that the default YAML values are loaded correctly."""

    def test_env_defaults_to_development(self) -> None:
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.env == "development"

    def test_live_trading_disabled_by_default(self) -> None:
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.live_trading_enabled is False

    def test_log_level_default(self) -> None:
        from prooflab.config import get_settings

        # development.yaml sets DEBUG
        settings = get_settings()
        assert settings.log.level.upper() == "DEBUG"

    def test_data_dirs_present(self) -> None:
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.data.raw_dir
        assert settings.data.processed_dir
        assert settings.data.cache_dir

    def test_db_url_present(self) -> None:
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.db.url


class TestEnvironmentVariableOverrides:
    """Verify that env vars take precedence over YAML."""

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROOFLAB_LOG__LEVEL", "ERROR")
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.log.level == "ERROR"

    def test_log_format_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROOFLAB_LOG__FORMAT", "json")
        from prooflab.config import get_settings

        settings = get_settings()
        assert settings.log.format == "json"

    def test_env_name_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROOFLAB_ENV", "production")
        from prooflab.config import get_settings

        settings = get_settings()
        # production.yaml sets env: production
        assert settings.env == "production"


class TestLiveTradingGate:
    """Verify that live trading cannot be enabled through configuration."""

    def test_direct_true_raises(self) -> None:
        from prooflab.config import ProofLabSettings

        with pytest.raises(ValidationError, match="live-trading gate"):
            ProofLabSettings(live_trading_enabled=True)

    def test_string_true_raises(self) -> None:
        from prooflab.config import ProofLabSettings

        with pytest.raises(ValidationError, match="live-trading gate"):
            ProofLabSettings(live_trading_enabled="true")  # type: ignore[arg-type]

    def test_env_var_true_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROOFLAB_LIVE_TRADING_ENABLED", "true")
        from prooflab.config import get_settings

        with pytest.raises(ValidationError, match="live-trading gate"):
            get_settings()

    def test_false_is_accepted(self) -> None:
        from prooflab.config import ProofLabSettings

        s = ProofLabSettings(live_trading_enabled=False)
        assert s.live_trading_enabled is False
