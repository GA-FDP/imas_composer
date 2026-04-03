"""
Test version uniqueness for CI/CD.

This test validates that the version in pyproject.toml is unique and has not been
previously used as a git tag. This ensures that each release has a unique version number.

This test is designed to run in CI without requiring D3D data access.
"""

import subprocess
import re
from pathlib import Path
import pytest


def get_version_from_pyproject():
    """Extract version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'

    if not pyproject_path.exists():
        pytest.fail(f"pyproject.toml not found at {pyproject_path}")

    with open(pyproject_path, 'r') as f:
        content = f.read()

    # Look for version = "x.y.z" pattern
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)

    if not match:
        pytest.fail("Could not find version in pyproject.toml")

    return match.group(1)


def get_git_tags():
    """Get list of all git tags in the repository."""
    try:
        # Try to get git tags - returns empty list if not in a git repo
        result = subprocess.run(
            ['git', 'tag', '-l'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent
        )
        tags = [tag.strip() for tag in result.stdout.strip().split('\n') if tag.strip()]
        return tags
    except subprocess.CalledProcessError:
        # Not a git repository or git not available
        pytest.skip("Not in a git repository or git not available")
    except FileNotFoundError:
        # git command not found
        pytest.skip("git command not available")


def test_version_is_unique():
    """
    Test that the version in pyproject.toml is not already used as a git tag.

    This ensures each release has a unique version number and prevents accidentally
    releasing the same version twice.
    """
    version = get_version_from_pyproject()
    tags = get_git_tags()

    # Expected tag format
    version_tag = f"v{version}"

    # Check if tag already exists
    if version_tag in tags:
        pytest.fail(
            f"Version tag '{version_tag}' already exists in the repository.\n"
            f"Current version in pyproject.toml: {version}\n"
            f"Please update the version in imas_composer/pyproject.toml to a new, unused version number.\n"
            f"Recent tags: {', '.join(sorted(tags)[-10:])}"
        )

    print(f"\nVersion check passed: {version_tag} is unique")


def test_version_format():
    """
    Test that the version follows semantic versioning format.

    Expected format: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-PRERELEASE
    Examples: 0.2.1, 1.0.0, 1.0.0-alpha, 1.0.0-beta.1
    """
    version = get_version_from_pyproject()

    # Semantic versioning regex pattern
    # Matches: 0.2.1, 1.0.0, 1.0.0-alpha, 1.0.0-beta.1, etc.
    semver_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$'

    if not re.match(semver_pattern, version):
        pytest.fail(
            f"Version '{version}' does not follow semantic versioning format.\n"
            f"Expected format: MAJOR.MINOR.PATCH (e.g., 0.2.1, 1.0.0)\n"
            f"Or with prerelease: MAJOR.MINOR.PATCH-PRERELEASE (e.g., 1.0.0-alpha, 1.0.0-beta.1)"
        )

    print(f"\nVersion format check passed: {version}")
