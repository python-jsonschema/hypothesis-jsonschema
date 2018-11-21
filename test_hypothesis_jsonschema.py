"""Tests for the hypothesis-jsonschema library."""

import os
import subprocess




def test_all_py_files_are_blackened():
    """Check that all .py files are formatted with Black."""
    files = []
    for dirpath, _, fnames in os.walk("."):
        files.extend(os.path.join(dirpath, f) for f in fnames if f.endswith(".py"))
    assert len(files) >= 3
    subprocess.run(
        ["black", "--py36", "--check"] + files,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
