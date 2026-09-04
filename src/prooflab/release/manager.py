"""Release manager for version extraction, tag calculation, and release packaging."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from prooflab.release.metadata import ReleasePackageInfo, ReleaseType, ReleaseVersion


class ReleaseManager:
    """Orchestrates release preparation, semantic tagging, and artifact validation."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path.cwd()

    def get_project_version(self, pyproject_path: Path | None = None) -> str:
        """Extract project version from pyproject.toml."""
        target = pyproject_path or (self.root_dir / "pyproject.toml")
        if not target.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {target}")

        content = target.read_text(encoding="utf-8")
        try:
            data = tomllib.loads(content)
            version: str = data["project"]["version"]
            return version
        except (KeyError, tomllib.TOMLDecodeError):
            # Fallback regex search
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
            raise ValueError(f"Could not extract project.version from {target}")

    def compute_prerelease_tag(
        self,
        version_str: str,
        run_number: int | str,
        prefix: str = "rc",
    ) -> str:
        """Generate deterministic pre-release tag, e.g. v0.1.0-rc.42."""
        base_v = ReleaseVersion.parse(version_str)
        return f"v{base_v.major}.{base_v.minor}.{base_v.patch}-{prefix}.{run_number}"

    def compute_release_tag(self, version_str: str) -> str:
        """Generate canonical GA production release tag, e.g. v0.1.0."""
        base_v = ReleaseVersion.parse(version_str)
        return f"v{base_v.major}.{base_v.minor}.{base_v.patch}"

    def generate_release_notes(
        self,
        version_str: str,
        release_type: ReleaseType,
        tag_name: str,
        commits: list[str] | None = None,
    ) -> str:
        """Construct structured Markdown release notes."""
        title_type = "Pre-Release" if release_type == ReleaseType.PRERELEASE else "Official Release"
        header = f"# Proof Lab {title_type} `{tag_name}`\n\n"

        if release_type == ReleaseType.PRERELEASE:
            summary = (
                "> [!IMPORTANT]\n"
                "> This is an automated **Pre-Release Candidate** generated from the "
                "`staging` branch.\n"
                "> Intended for paper trading verification, sandbox testing, and quality "
                "review.\n\n"
            )
        else:
            summary = (
                "> [!NOTE]\n"
                "> This is an official **General Availability Release** deployed from the "
                "`production` branch.\n\n"
            )

        details = "### Release Highlights\n"
        if commits:
            commit_lines = "\n".join(f"- {c.strip()}" for c in commits if c.strip())
            details += f"{commit_lines}\n\n"
        else:
            details += "- Full quantitative research, backtesting, risk engine, and UI suites.\n\n"

        assets = (
            "### Attached Distribution Artifacts\n"
            "- Python Wheel (`.whl`)\n"
            "- Source Distribution (`.tar.gz`)\n"
        )

        return f"{header}{summary}{details}{assets}"

    def verify_distribution_artifacts(self, dist_dir: Path) -> dict[str, list[Path]]:
        """Verify presence of built distribution wheels and source distributions."""
        if not dist_dir.exists() or not dist_dir.is_dir():
            raise FileNotFoundError(f"Distribution directory not found at {dist_dir}")

        wheels = list(dist_dir.glob("*.whl"))
        sdists = list(dist_dir.glob("*.tar.gz"))

        if not wheels:
            raise FileNotFoundError(f"No built .whl wheel files found in {dist_dir}")
        if not sdists:
            raise FileNotFoundError(f"No built .tar.gz source archives found in {dist_dir}")

        return {
            "wheels": wheels,
            "sdists": sdists,
        }

    def prepare_release(
        self,
        pyproject_path: Path | None = None,
        release_type: ReleaseType = ReleaseType.MAIN_RELEASE,
        run_number: int | str | None = None,
        dist_dir: Path | None = None,
        commits: list[str] | None = None,
    ) -> ReleasePackageInfo:
        """Assemble complete release metadata and validate built assets."""
        raw_version = self.get_project_version(pyproject_path)
        base_version = ReleaseVersion.parse(raw_version)

        if release_type == ReleaseType.PRERELEASE:
            run_id = run_number if run_number is not None else 1
            tag = self.compute_prerelease_tag(raw_version, run_id)
            title = f"Proof Lab v{base_version} Release Candidate {run_id}"
        else:
            tag = self.compute_release_tag(raw_version)
            title = f"Proof Lab v{base_version} Production Release"

        notes = self.generate_release_notes(
            version_str=raw_version,
            release_type=release_type,
            tag_name=tag,
            commits=commits,
        )

        artifacts: list[Path] = []
        if dist_dir and dist_dir.exists():
            verified = self.verify_distribution_artifacts(dist_dir)
            artifacts.extend(verified["wheels"])
            artifacts.extend(verified["sdists"])

        return ReleasePackageInfo(
            version=base_version,
            release_type=release_type,
            tag_name=tag,
            release_title=title,
            changelog_markdown=notes,
            artifacts=artifacts,
        )
