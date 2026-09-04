"""Unit tests validating GitHub Actions release workflow definitions and syntax."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_prerelease_workflow_structure() -> None:
    """Validate structure, triggers, permissions, and steps of prerelease.yml."""
    workflow_path = Path(".github/workflows/prerelease.yml")
    assert workflow_path.exists(), "prerelease.yml must exist"

    content = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    # Assert name and trigger
    assert content["name"] == "Staging Pre-Release"
    on_trigger = content.get("on") or content.get(True)
    assert on_trigger is not None
    assert "push" in on_trigger
    assert "staging" in on_trigger["push"]["branches"]
    assert "production" not in on_trigger["push"]["branches"]

    # Assert permissions
    assert content.get("permissions", {}).get("contents") == "write"

    # Assert jobs
    jobs = content["jobs"]
    assert "validate" in jobs
    assert "prerelease" in jobs
    assert jobs["prerelease"]["needs"] == "validate"

    # Check validation steps
    validate_steps = [s.get("name", "") for s in jobs["validate"]["steps"]]
    assert any("ruff" in s.lower() for s in validate_steps)
    assert any("mypy" in s.lower() for s in validate_steps)
    assert any("test" in s.lower() for s in validate_steps)

    # Check prerelease steps
    prerelease_steps = [s.get("name", "") for s in jobs["prerelease"]["steps"]]
    assert any("build" in s.lower() for s in prerelease_steps)
    assert any("pre-release tag" in s.lower() for s in prerelease_steps)
    assert any("publish" in s.lower() for s in prerelease_steps)


def test_release_workflow_structure() -> None:
    """Validate structure, triggers, permissions, and steps of release.yml."""
    workflow_path = Path(".github/workflows/release.yml")
    assert workflow_path.exists(), "release.yml must exist"

    content = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    # Assert name and trigger
    assert content["name"] == "Production Release"
    on_trigger = content.get("on") or content.get(True)
    assert on_trigger is not None
    assert "push" in on_trigger
    assert "production" in on_trigger["push"]["branches"]
    assert "staging" not in on_trigger["push"]["branches"]

    # Assert permissions
    assert content.get("permissions", {}).get("contents") == "write"

    # Assert jobs
    jobs = content["jobs"]
    assert "validate" in jobs
    assert "release" in jobs
    assert jobs["release"]["needs"] == "validate"

    # Check validation steps
    validate_steps = [s.get("name", "") for s in jobs["validate"]["steps"]]
    assert any("ruff" in s.lower() for s in validate_steps)
    assert any("mypy" in s.lower() for s in validate_steps)
    assert any("test" in s.lower() for s in validate_steps)

    # Check release steps
    release_steps = [s.get("name", "") for s in jobs["release"]["steps"]]
    assert any("build" in s.lower() for s in release_steps)
    assert any("production release tag" in s.lower() for s in release_steps)
    assert any("publish" in s.lower() for s in release_steps)
