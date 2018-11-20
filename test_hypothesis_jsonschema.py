"""Tests for the hypothesis-jsonschema library."""

import glob
import subprocess


def test_all_py_files_are_blackened() -> None:
    """Check that all .py files are formatted with Black."""
    files = glob.glob("*.py") + glob.glob("**/*.py")
    assert len(files) >= 3
    subprocess.run(
        ["black", "--py36", "--check"] + files,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
