"""Tests for the hypothesis-jsonschema library."""
# pylint: disable=no-value-for-parameter,wrong-import-order

import os
import subprocess

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import jsonschema
import pytest

from hypothesis_jsonschema import from_schema, json_schemata
from hypothesis_jsonschema._impl import (
    gen_number,
    gen_string,
    gen_array,
    gen_object,
)


def files_to_check():
    """Return a list of all .py files in the repo."""
    files = []
    for dirpath, _, fnames in os.walk("."):
        files.extend(os.path.join(dirpath, f) for f in fnames if f.endswith(".py"))
    assert len(files) >= 3
    return files


def test_all_py_files_are_blackened():
    """Check that all .py files are formatted with Black."""
    subprocess.run(
        ["black", "--py36", "--check"] + files_to_check(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_pylint_passes():
    """Check that pylint passes on all .py files."""
    subprocess.run(
        ["pylint"] + files_to_check(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


schema_strategies = [
    gen_number("integer"),
    gen_number("number"),
    gen_string(),
    gen_array(),
    gen_object(),
    json_schemata(),
]
for s in schema_strategies:
    s.validate()


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=100,  # maximum milliseconds per test case
)
@given(data=st.data())
@pytest.mark.parametrize("schema_strategy", schema_strategies)
def test_generated_data_matches_schema(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy)
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
