"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings

from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._impl import (
    JSON_STRATEGY,
    encode_canonical_json,
    gen_array,
    gen_enum,
    gen_number,
    gen_object,
    gen_string,
    json_schemata,
)


@given(JSON_STRATEGY)
def test_canonical_json_encoding(v):
    """Test our hand-rolled canonicaljson implementation."""
    encoded = encode_canonical_json(v)
    v2 = json.loads(encoded)
    assert v == v2
    assert encode_canonical_json(v2) == encoded


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
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


@pytest.mark.parametrize("name", sorted(catalog))
@settings(deadline=None, max_examples=10, suppress_health_check=HealthCheck.all())
@given(data=st.data())
def test_can_generate_for_real_large_schema(data, name):
    dumped = json.dumps(catalog[name])
    if '"$ref"' in dumped or '"$id"' in dumped:
        pytest.skip()
    value = data.draw(from_schema(catalog[name]))
    jsonschema.validate(value, catalog[name])


with open("corpus-suite-schemas.json") as f:
    suite, invalid_suite = json.load(f)
# Some tricky schema and interactions just aren't handled yet.
# Along with refs and dependencies, this is the main TODO list!
unhandled = {
    # Not yet implemented
    "draft4/allOf",
    "draft7/allOf",
    "draft4/allOf with base schema",
    "draft7/allOf with base schema",
    "draft4/properties, patternProperties, additionalProperties interaction",
    "draft7/properties, patternProperties, additionalProperties interaction",
    # Draft-04 tests which do not specify a type
    "draft4/exclusiveMinimum validation",
    "draft4/exclusiveMaximum validation",
    "draft4/minimum validation (explicit false exclusivity)",
    "draft4/maximum validation (explicit false exclusivity)",
}
# These upstream tests should specify some valid example, but don't.
for n in [
    "draft4/string",
    "draft7/string",
    "draft4/some languages do not distinguish between different types of numeric value",
]:
    suite[n] = invalid_suite.pop(n)


@pytest.mark.parametrize(
    "name",
    [
        n
        if n not in unhandled
        else pytest.param(n, marks=pytest.mark.xfail(strict=True))
        for n in sorted(suite)
    ],
)
@settings(deadline=None, max_examples=20)
@given(data=st.data())
def test_can_generate_for_test_suite_schema(data, name):
    if '"$ref"' in json.dumps(suite[name]):
        pytest.skip()

    value = data.draw(from_schema(suite[name]))
    jsonschema.validate(value, suite[name])


@pytest.mark.parametrize("name", sorted(invalid_suite))
def test_cannot_generate_for_empty_test_suite_schema(name):
    strat = from_schema(invalid_suite[name])
    with pytest.raises(Exception):
        strat.example()
