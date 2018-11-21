"""Tests for the hypothesis-jsonschema library."""

from distutils.version import StrictVersion
import os
import subprocess
from typing import List

import hypothesis_jsonschema


def test_version_is_semver() -> None:
    """Check that the version string is semver-compliant."""
    StrictVersion(hypothesis_jsonschema.__version__)


def test_all_py_files_are_blackened() -> None:
    """Check that all .py files are formatted with Black."""
    files: List[str] = []
    for dirpath, _, fnames in os.walk("."):
        files.extend(os.path.join(dirpath, f) for f in fnames if f.endswith(".py"))
    assert len(files) >= 3
    subprocess.run(
        ["black", "--py36", "--check"] + files,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
