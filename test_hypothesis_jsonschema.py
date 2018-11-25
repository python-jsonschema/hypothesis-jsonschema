"""Tests for the hypothesis-jsonschema library."""

import os
import subprocess

from hypothesis import given, settings, unlimited, HealthCheck
import hypothesis.strategies as st
import jsonschema
import pytest

from hypothesis_jsonschema import from_schema, json_schemata


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


@settings(
    max_examples=1000,
    suppress_health_check=[HealthCheck.too_slow],
    timeout=unlimited,
    deadline=100,  # maximum milliseconds per test case
)
@given(st.data(), json_schemata())
def test_generated_data_matches_schema(data, schema):
    """Check that an object drawn from an arbitrary schema is valid."""
    value = data.draw(from_schema(schema), "value from schema")
    jsonschema.validate(value, schema)


def test_boolean_true_is_valid_schema_and_resolvable():
    """...even though it's currently broken in jsonschema."""
    from_schema(True).example()


@pytest.mark.parametrize(
    "schema",
    [
        None,
        False,
        {"type": "an unknown type"},
        {"type": "string", "format": "not a real format"},
    ],
)
def test_invalid_schemas_raise(schema):
    """Trigger all the validation exceptions for full coverage."""
    with pytest.raises(Exception):
        from_schema(schema).example()
