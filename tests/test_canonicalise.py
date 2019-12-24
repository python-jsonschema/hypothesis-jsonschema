"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, assume, given, note, settings

from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._canonicalise import (
    FALSEY,
    JSON_STRATEGY,
    canonicalish,
    encode_canonical_json,
    get_type,
    is_valid,
    merged,
)
from hypothesis_jsonschema._gen_schemas import (
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
def test_canonicalises_to_equivalent_fixpoint(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy, label="schema")
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)
    instance = data.draw(JSON_STRATEGY | from_schema(cc), label="instance")
    assert is_valid(instance, schema) == is_valid(instance, cc)
    jsonschema.validators.validator_for(schema).check_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "maxProperties": 1, "required": ["0", "1"]},
        {"type": "array", "contains": False},
        {"type": "null", "enum": [False, True]},
        {"type": "boolean", "const": None},
        {"type": "array", "items": False, "minItems": 1},
        {
            "type": "array",
            "items": {"type": "null"},
            "uniqueItems": True,
            "minItems": 2,
        },
        {
            "type": "array",
            "items": {"type": "boolean"},
            "uniqueItems": True,
            "minItems": 3,
        },
        {
            "type": "array",
            "items": {"type": ["null", "boolean"]},
            "uniqueItems": True,
            "minItems": 4,
        },
        {"type": "array", "items": [True, False], "minItems": 2},
    ],
)
def test_canonicalises_to_empty(schema):
    assert canonicalish(schema) == {"not": {}}, (schema, canonicalish(schema))


@pytest.mark.parametrize(
    "schema,expected",
    [
        ({"type": get_type({})}, {}),
        ({"required": []}, {}),
        (
            {"type": "array", "items": [True, False, True]},
            {"type": "array", "items": [{}], "maxItems": 1},
        ),
    ],
)
def test_canonicalises_to_expected(schema, expected):
    assert canonicalish(schema) == expected, (schema, canonicalish(schema), expected)


@pytest.mark.parametrize(
    "group,result",
    [
        ([{"type": []}, {}], {"not": {}}),
        ([{"type": "null"}, {"const": 0}], {"not": {}}),
        ([{"type": "null"}, {"enum": [0]}], {"not": {}}),
        ([{"type": "integer"}, {"type": "string"}], {"not": {}}),
        ([{"type": "null"}, {"type": "boolean"}], {"not": {}}),
        ([{"type": "null"}, {"enum": [None, True]}], {"const": None}),
        ([{"type": "null"}, {"type": ["null", "boolean"]}], {"const": None}),
        ([{"type": "integer"}, {"maximum": 20}], {"type": "integer", "maximum": 20}),
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


@pytest.mark.xfail  # See https://github.com/Julian/jsonschema/issues/575
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


@pytest.mark.xfail(strict=True)
def test_merge_should_notice_required_disallowed_properties():
    # The required "name" property is banned by the additionalProperties: False
    # See https://github.com/Zac-HD/hypothesis-jsonschema/issues/30 for details.
    schemas = [
        {"type": "object", "properties": {"name": True}, "required": ["name"]},
        {"type": "object", "properties": {"id": True}, "additionalProperties": False},
    ]
    assert merged(schemas) == FALSEY
