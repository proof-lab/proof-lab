"""Data models and value objects for release metadata and version management."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ReleaseType(StrEnum):
    """Classification of the release tier."""

    PRERELEASE = "PRERELEASE"
    MAIN_RELEASE = "MAIN_RELEASE"


@dataclass(frozen=True)
class ReleaseVersion:
    """Parsed semantic version with components."""

    major: int
    minor: int
    patch: int
    prerelease_suffix: str | None = None

    @classmethod
    def parse(cls, version_str: str) -> ReleaseVersion:
        """Parse a semantic version string into structured components."""
        clean_v = version_str.strip().lstrip("v")
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$"
        match = re.match(pattern, clean_v)
        if not match:
            raise ValueError(f"Invalid semantic version format: {version_str}")

        major, minor, patch, suffix = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease_suffix=suffix if suffix else None,
        )

    def as_tag(self) -> str:
        """Return canonical git tag representation."""
        base = f"v{self.major}.{self.minor}.{self.patch}"
        if self.prerelease_suffix:
            return f"{base}-{self.prerelease_suffix}"
        return base

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease_suffix:
            return f"{base}-{self.prerelease_suffix}"
        return base


@dataclass
class ReleasePackageInfo:
    """Details for a release candidate or production release."""

    version: ReleaseVersion
    release_type: ReleaseType
    tag_name: str
    release_title: str
    changelog_markdown: str
    artifacts: list[Path] = field(default_factory=list)
