"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings

from hypothesis_jsonschema import from_schema, json_schemata
from hypothesis_jsonschema._impl import (
    JSON_STRATEGY,
    encode_canonical_json,
    gen_array,
    gen_enum,
    gen_number,
    gen_object,
    gen_string,
)


@given(JSON_STRATEGY)
def test_canonical_json_encoding(v):
    """Test our hand-rolled canonicaljson implementation."""
    encoded = encode_canonical_json(v)
    v2 = json.loads(encoded)
    assert v == v2
    assert encode_canonical_json(v2) == encoded


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
@pytest.mark.parametrize(
    "schema_strategy",
    [
        gen_number("integer"),
        gen_number("number"),
        gen_string(),
        gen_enum(),
        gen_array(),
        gen_object(),
        json_schemata(),
    ],
)
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


with open("corpus-schemastore-catalog.json") as f:
    catalog = json.load(f)


@pytest.mark.skip(reason="defintions and references not yet implemented")
@pytest.mark.parametrize("name", sorted(catalog))
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_can_generate_for_real_large_schema(data, name):
    value = data.draw(from_schema(catalog[name]))
    jsonschema.validate(value, catalog[name])
