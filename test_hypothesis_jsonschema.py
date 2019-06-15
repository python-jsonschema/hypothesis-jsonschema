"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, assume, given, note, reject, settings
from hypothesis.errors import InvalidArgument

from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._impl import (
    FALSEY,
    JSON_STRATEGY,
    canonicalish,
    encode_canonical_json,
    gen_array,
    gen_enum,
    gen_number,
    gen_object,
    gen_string,
    is_valid,
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


schema_strategy_params = pytest.mark.parametrize(
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
@schema_strategy_params
def test_generated_data_matches_schema(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy)
    try:
        value = data.draw(from_schema(schema), "value from schema")
    except InvalidArgument:
        reject()
    jsonschema.validate(value, schema)
    # This checks that our canonicalisation is semantically equivalent.
    jsonschema.validate(value, canonicalish(schema))


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
@schema_strategy_params
def test_canonicalises_to_equivalent_fixpoint(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy, label="schema")
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)
    instance = data.draw(JSON_STRATEGY | from_schema(cc), label="instance")
    assert is_valid(instance, schema) == is_valid(instance, cc)


def test_boolean_true_is_valid_schema_and_resolvable():
    """...even though it's currently broken in jsonschema."""
    from_schema(True).example()


@pytest.mark.parametrize(
    "group,result",
    [
        ([{"type": []}, {}], {"not": {}}),
        ([{"type": "null"}, {"type": "boolean"}], {"not": {}}),
        ([{"type": "null"}, {"type": ["null", "boolean"]}], {"type": ["null"]}),
        ([{"type": "integer"}, {"maximum": 20}], {"type": ["integer"], "maximum": 20}),
        (
            [
                {"properties": {"foo": {"maximum": 20}}},
                {"properties": {"foo": {"minimum": 10}}},
            ],
            {"properties": {"foo": {"maximum": 20, "minimum": 10}}},
        ),
    ],
)
def test_merged(group, result):
    assert merged(group) == result


@pytest.mark.parametrize(
    "group",
    [
        [
            {"$schema": "http://json-schema.org/draft-04/schema#"},
            {"$schema": "http://json-schema.org/draft-07/schema#"},
        ],
        [
            {"additionalProperties": {"type": "null"}},
            {"additionalProperties": {"type": "boolean"}},
        ],
        [
            {"additionalProperties": {"type": "null"}, "properties": {"foo": {}}},
            {"additionalProperties": {"type": "boolean"}},
        ],
        [
            {"additionalProperties": {"patternProperties": {".": {}}}},
            {"additionalProperties": {"patternProperties": {"a": {}}}},
        ],
        [
            {"patternProperties": {".": {}}},
            {"patternProperties": {"ab": {"type": "boolean"}}},
        ],
    ],
)
def test_unable_to_merge(group):
    assert merged(group) is None


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(json_schemata())
def test_self_merge_eq_canonicalish(schema):
    m = merged([schema, schema])
    assert m == canonicalish(schema)


@settings(suppress_health_check=HealthCheck.all(), deadline=None)
@given(st.data(), json_schemata(), json_schemata())
def test_merge_semantics(data, s1, s2):
    assume(canonicalish(s1) != FALSEY and canonicalish(s2) != FALSEY)
    combined = merged([s1, s2])
    assume(combined is not None)
    assume(combined != FALSEY)
    note(combined)
    ic = data.draw(from_schema(combined), label="combined")
    i1 = data.draw(from_schema(s1), label="s1")
    i2 = data.draw(from_schema(s2), label="s2")
    assert is_valid(ic, s1) and is_valid(ic, s2)
    assert is_valid(i1, s2) == is_valid(i1, combined)
    assert is_valid(i2, s1) == is_valid(i2, combined)


@pytest.mark.parametrize(
    "schema",
    [
        None,
        False,
        {"type": "an unknown type"},
        {"type": "string", "format": "not a real format"},
        {"allOf": [{"type": "boolean"}, {"const": None}]},
        {"allOf": [{"type": "boolean"}, {"enum": [None]}]},
    ],
)
def test_invalid_schemas_raise(schema):
    """Trigger all the validation exceptions for full coverage."""
    with pytest.raises(Exception):
        from_schema(schema).example()


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


def _has_refs(s):
    if isinstance(s, dict):
        return "$ref" in s or any(_has_refs(v) for v in s.values())
    return isinstance(s, list) and any(_has_refs(v) for v in s)


def to_name_params(corpus):
    for n in sorted(corpus):
        if n in FLAKY_SCHEMAS or _has_refs(corpus[n]):
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
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=20)
@given(data=st.data())
def test_can_generate_for_test_suite_schema(data, name):
    note(suite[name])
    value = data.draw(from_schema(suite[name]))
    try:
        jsonschema.validate(value, suite[name])
    except jsonschema.exceptions.SchemaError:
        jsonschema.Draft4Validator(suite[name]).validate(value)


@pytest.mark.parametrize("name", to_name_params(invalid_suite))
def test_cannot_generate_for_empty_test_suite_schema(name):
    strat = from_schema(invalid_suite[name])
    with pytest.raises(Exception):
        strat.example()


# This schema has overlapping patternProperties - this is OK, so long as they're
# merged or otherwise handled correctly, with the exception of the key "ab" which
# would have to be both an integer and a string (and is thus disallowed).
OVERLAPPING_PATTERNS_SCHEMA = dict(
    type="string",
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
