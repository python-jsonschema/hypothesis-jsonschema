"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, given, note, settings

from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._impl import (
    JSON_STRATEGY,
    canonicalish,
    encode_canonical_json,
    gen_array,
    gen_enum,
    gen_number,
    gen_object,
    gen_string,
    json_schemata,
    merged,
)


@given(JSON_STRATEGY)
def test_canonical_json_encoding(v):
    """Test our hand-rolled canonicaljson implementation."""
    encoded = encode_canonical_json(v)
    v2 = json.loads(encoded)
    assert v == v2
    assert encode_canonical_json(v2) == encoded


schema_strategy = pytest.mark.parametrize(
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


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
@schema_strategy
def test_generated_data_matches_schema(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy)
    value = data.draw(from_schema(schema), "value from schema")
    jsonschema.validate(value, schema)


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
@schema_strategy
def test_canonicalises_to_fixpoint(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy)
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)


def test_boolean_true_is_valid_schema_and_resolvable():
    """...even though it's currently broken in jsonschema."""
    from_schema(True).example()


@pytest.mark.parametrize(
    "group,result",
    [
        ([{"type": "null"}, {"type": "boolean"}], {"not": {}}),
        ([{"type": "integer"}, {"maximum": 20}], {"type": ["integer"], "maximum": 20}),
        (
            [
                {"properties": {"foo": {"maximum": 20}}},
                {"properties": {"foo": {"minimum": 10}}},
            ],
            {
                "type": ["object"],
                "properties": {
                    "foo": {"type": ["integer", "number"], "maximum": 20, "minimum": 10}
                },
            },
        ),
        (
            [
                {"$schema": "http://json-schema.org/draft-04/schema#"},
                {"$schema": "http://json-schema.org/draft-07/schema#"},
            ],
            None,
        ),
    ],
)
def test_merged(group, result):
    assert merged(group) == result


@pytest.mark.parametrize(
    "schema",
    [
        None,
        False,
        {"type": "an unknown type"},
        {"type": "string", "format": "not a real format"},
        {"allOf": [{"type": "boolean"}, {"const": None}]},
    ],
)
def test_invalid_schemas_raise(schema):
    """Trigger all the validation exceptions for full coverage."""
    with pytest.raises(Exception):
        from_schema(schema).example()


# Some tricky schema and interactions just aren't handled yet.
# Along with refs and dependencies, this is the main TODO list!
EXPECTED_FAILURES = {
    # Just plain weird - regex issues etc.
    "JSON Schema for mime type collections"
}
FLAKY_SCHEMAS = {
    # Yep, lists of lists of lists of lists of lists of integers are HealthCheck-slow
    "draft4/nested items",
    "draft7/nested items",
    # Something weird about a null that should be a string??  TODO: debug that.
    "Datalogic Scan2Deploy Android file",
    "Datalogic Scan2Deploy CE file",
}

with open("corpus-schemastore-catalog.json") as f:
    catalog = json.load(f)
with open("corpus-suite-schemas.json") as f:
    suite, invalid_suite = json.load(f)
with open("corpus-reported.json") as f:
    reported = json.load(f)
    assert set(reported).isdisjoint(suite)
    suite.update(reported)


def to_name_params(corpus):
    for n in sorted(corpus):
        if n.split("/", 1)[-1] in EXPECTED_FAILURES:
            yield pytest.param(n, marks=pytest.mark.xfail(strict=True))
        elif n in FLAKY_SCHEMAS or '"$ref"' in json.dumps(corpus[n]):
            yield pytest.param(n, marks=pytest.mark.skip)
        else:
            yield n


@pytest.mark.parametrize("name", to_name_params(catalog))
@settings(deadline=None, max_examples=5, suppress_health_check=HealthCheck.all())
@given(data=st.data())
def test_can_generate_for_real_large_schema(data, name):
    note(name)
    value = data.draw(from_schema(catalog[name]))
    jsonschema.validate(value, catalog[name])


@pytest.mark.parametrize("name", to_name_params(suite))
@settings(deadline=None, max_examples=20)
@given(data=st.data())
def test_can_generate_for_test_suite_schema(data, name):
    note(suite[name])
    value = data.draw(from_schema(suite[name]))
    try:
        jsonschema.validate(value, suite[name])
    except jsonschema.exceptions.SchemaError:
        jsonschema.Draft4Validator(suite[name]).validate(value)


@pytest.mark.parametrize("name", sorted(invalid_suite))
def test_cannot_generate_for_empty_test_suite_schema(name):
    strat = from_schema(invalid_suite[name])
    with pytest.raises(Exception):
        strat.example()


# This schema has overlapping patternProperties - this is OK, so long as they're
# merged or otherwise handled correctly, with the exception of the key "ab" which
# would have to be both an integer and a string (and is thus disallowed).
OVERLAPPING_PATTERNS_SCHEMA = dict(
    patternProperties={
        r"\A[ab]{1,2}\Z": {},
        r"\Aa[ab]\Z": {"type": "integer"},
        r"\A[ab]b\Z": {"type": "string"},
    },
    additionalProperties=False,
    minimumProperties=1,
)


@given(from_schema(OVERLAPPING_PATTERNS_SCHEMA))
def test_handles_overlapping_patternProperties(value):
    jsonschema.validate(value, OVERLAPPING_PATTERNS_SCHEMA)
    assert "ab" not in value
