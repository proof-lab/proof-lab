"""Strict security validation and integrity checks for imported .plb strategy packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


class SecurityViolationError(Exception):
    """Raised when a package violates security constraints."""


class ChecksumVerificationError(Exception):
    """Raised when package member checksums fail verification."""


class PackageSecurityValidator:
    """Enforces strict security isolation and integrity validation on untrusted .plb archives."""

    ALLOWED_EXTENSIONS = {
        ".json",
        ".yaml",
        ".yml",
        ".plmodel",
        ".txt",
        ".bin",
        ".parquet",
    }
    DISALLOWED_EXTENSIONS = {
        ".py",
        ".pyc",
        ".pyd",
        ".sh",
        ".bash",
        ".exe",
        ".bat",
        ".cmd",
        ".dll",
        ".so",
        ".dylib",
        ".ps1",
        ".vbs",
        ".jar",
    }
    ALLOWED_PREFIXES = {
        "manifest.json",
        "models/",
        "calibration/",
        "features/",
        "strategy/",
        "validation/",
        "checksums/",
        "metadata/",
    }

    def __init__(
        self,
        max_file_count: int = 200,
        max_file_size: int = 50 * 1024 * 1024,  # 50 MB per file
        max_total_size: int = 150 * 1024 * 1024,  # 150 MB total uncompressed
        max_compression_ratio: float = 100.0,
    ) -> None:
        self.max_file_count = max_file_count
        self.max_file_size = max_file_size
        self.max_total_size = max_total_size
        self.max_compression_ratio = max_compression_ratio

    def validate_zip_structure(self, zip_file: ZipFile) -> None:
        """Inspect all ZIP directory entries for path traversal, size limits, and extensions."""
        infolist = zip_file.infolist()

        if len(infolist) == 0:
            raise SecurityViolationError("Package archive is empty")

        if len(infolist) > self.max_file_count:
            raise SecurityViolationError(
                f"Package contains {len(infolist)} files, exceeding limit of {self.max_file_count}"
            )

        total_uncompressed = 0
        has_manifest = False
        has_checksums = False

        for info in infolist:
            name = info.filename
            # Normalize slashes
            norm_name = name.replace("\\", "/")

            # 1. Path Traversal & Absolute Path Checks
            if "\x00" in norm_name:
                raise SecurityViolationError(f"Null byte detected in member name: {name!r}")

            if norm_name.startswith("/") or (len(norm_name) > 1 and norm_name[1] == ":"):
                raise SecurityViolationError(f"Absolute path forbidden in package: {name}")

            parts = [p for p in norm_name.split("/") if p]
            if ".." in parts or "." in parts:
                raise SecurityViolationError(f"Directory traversal sequence forbidden: {name}")

            # Ignore directory entries for size/extension checks
            if norm_name.endswith("/"):
                continue

            # 2. Allowed Prefix Check
            if not any(
                norm_name == p or norm_name.startswith(p)
                for p in self.ALLOWED_PREFIXES
            ):
                raise SecurityViolationError(
                    f"Unexpected file path outside allowed package layout: {name}"
                )

            # 3. File Extension Check
            suffix = Path(norm_name).suffix.lower()
            if suffix in self.DISALLOWED_EXTENSIONS:
                raise SecurityViolationError(
                    f"Disallowed executable/script file type: {name}"
                )
            if suffix not in self.ALLOWED_EXTENSIONS:
                raise SecurityViolationError(
                    f"File type {suffix} is not in allowed extensions list: {name}"
                )

            # 4. Decompression Size & Zip Bomb Checks
            if info.file_size > self.max_file_size:
                raise SecurityViolationError(
                    f"File {name} size ({info.file_size} bytes) "
                    f"exceeds limit ({self.max_file_size} bytes)"
                )

            total_uncompressed += info.file_size
            if total_uncompressed > self.max_total_size:
                raise SecurityViolationError(
                    f"Total uncompressed size ({total_uncompressed} bytes) "
                    f"exceeds limit ({self.max_total_size} bytes)"
                )

            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > self.max_compression_ratio:
                    raise SecurityViolationError(
                        f"Suspicious compression ratio {ratio:.1f}x for {name} (possible zip bomb)"
                    )

            if norm_name == "manifest.json":
                has_manifest = True
            elif norm_name == "checksums/sha256.json":
                has_checksums = True

        if not has_manifest:
            raise SecurityViolationError("Package missing mandatory manifest.json")
        if not has_checksums:
            raise SecurityViolationError("Package missing mandatory checksums/sha256.json")

    def verify_checksums(self, zip_file: ZipFile) -> dict[str, str]:
        """Verify SHA-256 checksums of all package members against checksums/sha256.json."""
        try:
            checksum_raw = zip_file.read("checksums/sha256.json")
            manifest_checksums = json.loads(checksum_raw.decode("utf-8"))
        except Exception as err:
            raise ChecksumVerificationError(
                f"Failed to read checksums/sha256.json: {err}"
            ) from err

        if not isinstance(manifest_checksums, dict):
            raise ChecksumVerificationError(
                "checksums/sha256.json must contain a dictionary mapping filename to sha256 hex"
            )

        actual_checksums: dict[str, str] = {}
        for info in zip_file.infolist():
            norm_name = info.filename.replace("\\", "/")
            if norm_name.endswith("/") or norm_name == "checksums/sha256.json":
                continue

            content = zip_file.read(info.filename)
            digest = hashlib.sha256(content).hexdigest()
            actual_checksums[norm_name] = digest

            expected_digest = manifest_checksums.get(norm_name)
            if expected_digest is None:
                raise ChecksumVerificationError(
                    f"Unregistered file found in package not in checksums.json: {norm_name}"
                )
            if expected_digest.lower() != digest.lower():
                raise ChecksumVerificationError(
                    f"Checksum mismatch for {norm_name}: "
                    f"expected {expected_digest}, computed {digest}"
                )

        # Check for missing files
        for expected_file in manifest_checksums:
            if expected_file not in actual_checksums:
                raise ChecksumVerificationError(
                    f"File listed in checksums.json is missing from package: {expected_file}"
                )

        return actual_checksums
