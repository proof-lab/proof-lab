"""Security and integrity unit tests verifying defensive parsing of untrusted .plb archives."""

import hashlib
import io
import json
from zipfile import ZipFile

import pytest

from prooflab.packaging.security import (
    ChecksumVerificationError,
    PackageSecurityValidator,
    SecurityViolationError,
)


def _build_test_zip(files: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    buf.seek(0)
    return buf


def test_validator_detects_empty_archive_or_file_count_limit() -> None:
    validator = PackageSecurityValidator(max_file_count=2)

    # Empty zip
    buf_empty = io.BytesIO()
    with ZipFile(buf_empty, "w"):
        pass
    buf_empty.seek(0)
    with ZipFile(buf_empty, "r") as z:
        with pytest.raises(SecurityViolationError, match="Package archive is empty"):
            validator.validate_zip_structure(z)

    # File count limit exceeded
    buf_many = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "models/1.bin": b"1",
    })
    with ZipFile(buf_many, "r") as z:
        with pytest.raises(SecurityViolationError, match="exceeding limit"):
            validator.validate_zip_structure(z)


def test_validator_detects_missing_manifest_or_checksums() -> None:
    validator = PackageSecurityValidator()

    # Missing manifest
    buf_no_manifest = _build_test_zip({
        "checksums/sha256.json": b"{}",
        "strategy/strategy.yaml": b"{}",
    })
    with ZipFile(buf_no_manifest, "r") as z:
        with pytest.raises(SecurityViolationError, match="missing mandatory manifest.json"):
            validator.validate_zip_structure(z)

    # Missing checksums
    buf_no_checksums = _build_test_zip({
        "manifest.json": b"{}",
        "strategy/strategy.yaml": b"{}",
    })
    with ZipFile(buf_no_checksums, "r") as z:
        with pytest.raises(SecurityViolationError, match="missing mandatory checksums/sha256.json"):
            validator.validate_zip_structure(z)


def test_validator_rejects_null_bytes_and_unexpected_prefixes() -> None:
    validator = PackageSecurityValidator()

    # Unexpected prefix
    buf_bad_prefix = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "unknown_folder/data.json": b"{}",
    })
    with ZipFile(buf_bad_prefix, "r") as z:
        with pytest.raises(SecurityViolationError, match="Unexpected file path outside"):
            validator.validate_zip_structure(z)

    # Disallowed extension not in allowed list
    buf_disallowed = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "models/data.weird": b"{}",
    })
    with ZipFile(buf_disallowed, "r") as z:
        with pytest.raises(SecurityViolationError, match="not in allowed extensions list"):
            validator.validate_zip_structure(z)


def test_validator_rejects_path_traversal() -> None:
    validator = PackageSecurityValidator()

    # Directory traversal ../
    buf_traversal = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "models/../../evil.json": b"malicious",
    })
    with ZipFile(buf_traversal, "r") as z:
        with pytest.raises(SecurityViolationError, match="Directory traversal"):
            validator.validate_zip_structure(z)

    # Absolute path /
    buf_abs = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "/etc/passwd": b"root",
    })
    with ZipFile(buf_abs, "r") as z:
        with pytest.raises(SecurityViolationError, match="Absolute path forbidden"):
            validator.validate_zip_structure(z)


def test_validator_rejects_executable_scripts() -> None:
    validator = PackageSecurityValidator()

    dangerous_files = [
        "models/exploit.py",
        "models/script.sh",
        "strategy/run.bat",
        "models/payload.exe",
        "models/code.pyc",
    ]

    for d_file in dangerous_files:
        buf = _build_test_zip({
            "manifest.json": b"{}",
            "checksums/sha256.json": b"{}",
            d_file: b"print('hacked')",
        })
        with ZipFile(buf, "r") as z:
            with pytest.raises(SecurityViolationError, match="Disallowed executable/script"):
                validator.validate_zip_structure(z)


def test_validator_rejects_zip_bomb_and_size_limits() -> None:
    # Max file size limit
    validator_file_size = PackageSecurityValidator(max_file_size=1000)
    buf_large = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "models/big.bin": b"x" * 2000,
    })
    with ZipFile(buf_large, "r") as z:
        with pytest.raises(SecurityViolationError, match="exceeds limit"):
            validator_file_size.validate_zip_structure(z)

    # Max total size limit
    validator_total_size = PackageSecurityValidator(max_file_size=2000, max_total_size=2500)
    buf_total = _build_test_zip({
        "manifest.json": b"{}",
        "checksums/sha256.json": b"{}",
        "models/file1.bin": b"x" * 1500,
        "models/file2.bin": b"x" * 1500,
    })
    with ZipFile(buf_total, "r") as z:
        with pytest.raises(SecurityViolationError, match="Total uncompressed size"):
            validator_total_size.validate_zip_structure(z)


def test_validator_checksum_verification_and_error_cases() -> None:
    validator = PackageSecurityValidator()

    m_bytes = b'{"format_version": "1.0.0"}'
    s_bytes = b'{"strategy": 1}'

    m_hash = hashlib.sha256(m_bytes).hexdigest()
    s_hash = hashlib.sha256(s_bytes).hexdigest()

    # Valid package
    checksums_valid = json.dumps({
        "manifest.json": m_hash,
        "strategy/strategy.yaml": s_hash,
    }).encode("utf-8")

    buf_valid = _build_test_zip({
        "manifest.json": m_bytes,
        "strategy/strategy.yaml": s_bytes,
        "checksums/sha256.json": checksums_valid,
    })
    with ZipFile(buf_valid, "r") as z:
        validator.validate_zip_structure(z)
        verified = validator.verify_checksums(z)
        assert verified["manifest.json"] == m_hash

    # Tampered file
    buf_tampered = _build_test_zip({
        "manifest.json": b'{"tampered": true}',
        "strategy/strategy.yaml": s_bytes,
        "checksums/sha256.json": checksums_valid,
    })
    with ZipFile(buf_tampered, "r") as z:
        with pytest.raises(ChecksumVerificationError, match="Checksum mismatch"):
            validator.verify_checksums(z)

    # Missing file
    buf_missing = _build_test_zip({
        "manifest.json": m_bytes,
        "checksums/sha256.json": checksums_valid,
    })
    with ZipFile(buf_missing, "r") as z:
        with pytest.raises(ChecksumVerificationError, match="missing from package"):
            validator.verify_checksums(z)

    # Corrupt checksums.json (non-json)
    buf_corrupt_json = _build_test_zip({
        "manifest.json": m_bytes,
        "checksums/sha256.json": b"not valid json {{{",
    })
    with ZipFile(buf_corrupt_json, "r") as z:
        with pytest.raises(ChecksumVerificationError, match="Failed to read checksums"):
            validator.verify_checksums(z)

    # Checksums.json containing list instead of dict
    buf_list_json = _build_test_zip({
        "manifest.json": m_bytes,
        "checksums/sha256.json": b'["not", "a", "dict"]',
    })
    with ZipFile(buf_list_json, "r") as z:
        with pytest.raises(ChecksumVerificationError, match="must contain a dictionary"):
            validator.verify_checksums(z)

    # Unregistered file in zip not present in checksums.json
    checksums_partial = json.dumps({
        "manifest.json": m_hash,
    }).encode("utf-8")
    buf_unregistered = _build_test_zip({
        "manifest.json": m_bytes,
        "strategy/strategy.yaml": s_bytes,
        "checksums/sha256.json": checksums_partial,
    })
    with ZipFile(buf_unregistered, "r") as z:
        with pytest.raises(ChecksumVerificationError, match="Unregistered file found in package"):
            validator.verify_checksums(z)
