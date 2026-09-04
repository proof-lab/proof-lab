"""Unit tests for ReleaseManager, version resolution, and artifact validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from prooflab.release import (
    ReleaseManager,
    ReleaseType,
    ReleaseVersion,
)


def test_release_version_parsing_and_formatting() -> None:
    """Test parsing and string formatting of semantic version objects."""
    v1 = ReleaseVersion.parse("0.1.0")
    assert v1.major == 0
    assert v1.minor == 1
    assert v1.patch == 0
    assert v1.prerelease_suffix is None
    assert v1.as_tag() == "v0.1.0"
    assert str(v1) == "0.1.0"

    v2 = ReleaseVersion.parse("v1.2.3-rc.4")
    assert v2.major == 1
    assert v2.minor == 2
    assert v2.patch == 3
    assert v2.prerelease_suffix == "rc.4"
    assert v2.as_tag() == "v1.2.3-rc.4"
    assert str(v2) == "1.2.3-rc.4"

    with pytest.raises(ValueError, match="Invalid semantic version format"):
        ReleaseVersion.parse("invalid-version")


def test_release_manager_get_project_version(tmp_path: Path) -> None:
    """Test reading version from pyproject.toml."""
    mgr = ReleaseManager(root_dir=tmp_path)
    pyproject_file = tmp_path / "pyproject.toml"

    # Missing file
    with pytest.raises(FileNotFoundError):
        mgr.get_project_version()

    # Valid toml
    pyproject_file.write_text(
        '[project]\nname = "test-pkg"\nversion = "0.3.5"\n',
        encoding="utf-8",
    )
    assert mgr.get_project_version() == "0.3.5"


def test_compute_prerelease_and_release_tags() -> None:
    """Test deterministic tag computation."""
    mgr = ReleaseManager()
    assert mgr.compute_prerelease_tag("0.1.0", 42) == "v0.1.0-rc.42"
    assert mgr.compute_prerelease_tag("1.0.0", "beta.1", prefix="pre") == "v1.0.0-pre.beta.1"
    assert mgr.compute_release_tag("0.1.0") == "v0.1.0"
    assert mgr.compute_release_tag("v2.4.1") == "v2.4.1"


def test_generate_release_notes() -> None:
    """Test generation of markdown release notes for prerelease and main release."""
    mgr = ReleaseManager()

    # Prerelease notes
    rc_notes = mgr.generate_release_notes(
        version_str="0.1.0",
        release_type=ReleaseType.PRERELEASE,
        tag_name="v0.1.0-rc.5",
        commits=["feat(ui): add live dashboard", "fix(data): fix timestamp conversion"],
    )
    assert "Pre-Release `v0.1.0-rc.5`" in rc_notes
    assert "Pre-Release Candidate" in rc_notes
    assert "feat(ui): add live dashboard" in rc_notes
    assert "fix(data): fix timestamp conversion" in rc_notes

    # Production notes
    prod_notes = mgr.generate_release_notes(
        version_str="0.1.0",
        release_type=ReleaseType.MAIN_RELEASE,
        tag_name="v0.1.0",
    )
    assert "Official Release `v0.1.0`" in prod_notes
    assert "General Availability Release" in prod_notes
    assert "Attached Distribution Artifacts" in prod_notes


def test_verify_distribution_artifacts_and_prepare_release(tmp_path: Path) -> None:
    """Test artifact verification and end-to-end prepare_release method."""
    mgr = ReleaseManager(root_dir=tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "prooflab"\nversion = "0.2.0"\n', encoding="utf-8")

    dist_dir = tmp_path / "dist"

    # Dist dir does not exist yet
    with pytest.raises(FileNotFoundError):
        mgr.verify_distribution_artifacts(dist_dir)

    dist_dir.mkdir()
    # Missing wheels
    with pytest.raises(FileNotFoundError, match="No built .whl wheel files"):
        mgr.verify_distribution_artifacts(dist_dir)

    # Add mock wheel
    wheel = dist_dir / "prooflab-0.2.0-py3-none-any.whl"
    wheel.write_text("mock wheel content", encoding="utf-8")

    # Missing sdist
    with pytest.raises(FileNotFoundError, match="No built .tar.gz source archives"):
        mgr.verify_distribution_artifacts(dist_dir)

    # Add mock sdist
    sdist = dist_dir / "prooflab-0.2.0.tar.gz"
    sdist.write_text("mock sdist content", encoding="utf-8")

    verified = mgr.verify_distribution_artifacts(dist_dir)
    assert len(verified["wheels"]) == 1
    assert len(verified["sdists"]) == 1

    # Prepare prerelease
    pre_info = mgr.prepare_release(
        pyproject_path=pyproject,
        release_type=ReleaseType.PRERELEASE,
        run_number=99,
        dist_dir=dist_dir,
    )
    assert pre_info.tag_name == "v0.2.0-rc.99"
    assert pre_info.release_type == ReleaseType.PRERELEASE
    assert len(pre_info.artifacts) == 2

    # Prepare main release
    main_info = mgr.prepare_release(
        pyproject_path=pyproject,
        release_type=ReleaseType.MAIN_RELEASE,
        dist_dir=dist_dir,
    )
    assert main_info.tag_name == "v0.2.0"
    assert main_info.release_type == ReleaseType.MAIN_RELEASE
    assert len(main_info.artifacts) == 2
