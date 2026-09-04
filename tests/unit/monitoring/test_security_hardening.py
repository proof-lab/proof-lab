"""Unit tests for secrets sanitization and comprehensive security audit review."""

from __future__ import annotations

import logging

from prooflab.monitoring.security import (
    SecretSanitizer,
    SecretSanitizingFilter,
    SecurityAuditReport,
    SecurityCheckStatus,
    SecurityReviewer,
)


def test_secret_sanitizer_string_patterns() -> None:
    """Test string pattern redaction for passwords, tokens, API keys, and connection strings."""
    # Passwords in key-value format
    text1 = "Connecting to database with password=super_secret_pwd and user=trader"
    assert SecretSanitizer.sanitize_string(text1) == (
        "Connecting to database with password=***REDACTED*** and user=trader"
    )

    # API key with quotes
    text2 = 'Setting API_KEY = "sk-live-12345678abcdef" for broker'
    assert SecretSanitizer.sanitize_string(text2) == (
        'Setting API_KEY = "***REDACTED***" for broker'
    )

    # Bearer token
    text3 = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    assert SecretSanitizer.sanitize_string(text3) == "Authorization: Bearer ***REDACTED***"

    # Connection URI
    text4 = "postgresql://dbuser:super_secret_password@localhost:5432/prooflab"
    assert SecretSanitizer.sanitize_string(text4) == (
        "postgresql://dbuser:***REDACTED***@localhost:5432/prooflab"
    )

    # MT5 password
    text5 = "mt5_password: my_broker_secret_password"
    assert SecretSanitizer.sanitize_string(text5) == "mt5_password: ***REDACTED***"


def test_secret_sanitizer_nested_dictionary() -> None:
    """Test recursive redaction across nested dictionaries and lists."""
    raw_data = {
        "strategy_id": "strat_01",
        "api_key": "live_key_abcdef123456",
        "credentials": {
            "token": "secret_token_val",
            "server": "mt5.broker.com",
            "nested_list": [
                {"private_key": "raw_private_key_content"},
                {"safe_symbol": "EURUSD"},
            ],
        },
        "config": {
            "connection": "mysql://root:root_pass_99@127.0.0.1:3306/db",
        },
    }

    sanitized = SecretSanitizer.sanitize_dict(raw_data)

    assert sanitized["strategy_id"] == "strat_01"
    assert sanitized["api_key"] == "***REDACTED***"
    assert sanitized["credentials"]["token"] == "***REDACTED***"
    assert sanitized["credentials"]["server"] == "mt5.broker.com"
    assert sanitized["credentials"]["nested_list"][0]["private_key"] == "***REDACTED***"
    assert sanitized["credentials"]["nested_list"][1]["safe_symbol"] == "EURUSD"
    assert "root_pass_99" not in sanitized["config"]["connection"]
    assert "***REDACTED***" in sanitized["config"]["connection"]


def test_secret_sanitizing_logging_filter() -> None:
    """Test logging filter intercepts and sanitizes log messages, args, and tracebacks."""
    logger = logging.getLogger("test_security_logger")
    logger.setLevel(logging.INFO)

    log_filter = SecretSanitizingFilter()
    logger.addFilter(log_filter)

    # Capture record
    records: list[logging.LogRecord] = []

    class MockHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = MockHandler()
    handler.addFilter(log_filter)
    logger.addHandler(handler)

    logger.info("Initializing broker connection with password=%s", "super_secret_broker_pwd")

    assert len(records) == 1
    assert "super_secret_broker_pwd" not in records[0].getMessage()
    assert "***REDACTED***" in records[0].getMessage()

    logger.info("Direct log message containing token=eyJ1234567890abcdef")
    assert len(records) == 2
    assert "token=***REDACTED***" in records[1].getMessage()



def test_security_reviewer_comprehensive_audit() -> None:
    """Test security reviewer executes audit checks across package, secrets, and API routes."""
    report = SecurityReviewer.run_comprehensive_security_audit()

    assert isinstance(report, SecurityAuditReport)
    assert report.overall_status in (SecurityCheckStatus.PASSED, SecurityCheckStatus.WARNING)
    assert len(report.checks) >= 3

    check_names = {c.check_name for c in report.checks}
    assert "secrets_sanitization" in check_names
    assert "package_import_security" in check_names
    assert "api_endpoints_security" in check_names


def test_secret_sanitizer_data_structures_and_exceptions() -> None:
    """Test sanitization of sets, tuples, and exception tracebacks."""
    # Tuples and sets
    data_tuple = ("user", "token=secret_val_123", 42)
    sanitized_tuple = SecretSanitizer.sanitize_obj(data_tuple)
    assert sanitized_tuple == ("user", "token=***REDACTED***", 42)

    data_set = {"safe_val", "api_key=sk_test_999"}
    sanitized_set = SecretSanitizer.sanitize_obj(data_set)
    assert "api_key=***REDACTED***" in sanitized_set

    # Various URI schemes
    redis_uri = "redis://user:my_secret_redis_pw@localhost:6379/0"
    assert SecretSanitizer.sanitize_string(redis_uri) == "redis://user:***REDACTED***@localhost:6379/0"

    mongo_uri = "mongodb://admin:super_mongo_secret@10.0.0.1:27017/admin"
    assert SecretSanitizer.sanitize_string(mongo_uri) == "mongodb://admin:***REDACTED***@10.0.0.1:27017/admin"


def test_secret_sanitizing_filter_with_exception() -> None:
    """Test logging filter redacts secrets inside exception tracebacks."""
    log_filter = SecretSanitizingFilter()
    record = logging.LogRecord(
        name="test_exc",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Error occurred during connection",
        args=(),
        exc_info=None,
    )
    record.exc_text = "Traceback: ConnectError: failed with password=leak_pwd_999 at host"
    assert log_filter.filter(record)
    assert "password=***REDACTED***" in record.exc_text
    assert "leak_pwd_999" not in record.exc_text

