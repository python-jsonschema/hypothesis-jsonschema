"""Tests for the hypothesis-jsonschema library."""

import json

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis.errors import InvalidArgument

from gen_schemas import json_schemata, schema_strategy_params
from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._canonicalise import (
    FALSEY,
    canonicalish,
    encode_canonical_json,
    get_type,
    make_validator,
    merged,
    resolve_all_refs,
)
from hypothesis_jsonschema._from_schema import JSON_STRATEGY


def is_valid(instance, schema):
    return make_validator(schema).is_valid(instance)


@given(JSON_STRATEGY)
def test_canonical_json_encoding(v):
    """Test our hand-rolled canonicaljson implementation."""
    encoded = encode_canonical_json(v)
    v2 = json.loads(encoded)
    assert v == v2
    assert encode_canonical_json(v2) == encoded


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
@schema_strategy_params
def test_canonicalises_to_equivalent_fixpoint(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy, label="schema")
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)
    try:
        strat = from_schema(cc)
    except InvalidArgument:
        # e.g. array of unique {type: integers}, with too few allowed integers
        assume(False)
    instance = data.draw(JSON_STRATEGY | strat, label="instance")
    assert is_valid(instance, schema) == is_valid(instance, cc)
    jsonschema.validators.validator_for(schema).check_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "maxProperties": 1, "required": ["0", "1"]},
        {"type": "object", "required": [""], "propertyNames": {"minLength": 1}},
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
        {"type": "integer", "minimum": 2, "maximum": 1},
        {"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 3},
        {"type": "number", "exclusiveMinimum": 0, "maximum": 0},
        {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 3,
            "multipleOf": 3,
        },
        {"not": {"type": ["integer", "number"]}, "type": "number"},
        {"oneOf": []},
        {"oneOf": [{}, {}]},
        {"oneOf": [True, False, {}]},
        {"anyOf": [False, {"not": {}}]},
        {"type": "object", "maxProperties": 2, "minProperties": 3},
        {
            "type": "object",
            "maxProperties": 1,
            "required": ["", "0"],
            "propertyNames": {"minLength": 2},
        },
        pytest.param(
            {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 0},
                "uniqueItems": True,
                "minItems": 2,
            },
            marks=pytest.mark.xfail,
        ),
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
        (
            {"type": "integer", "minimum": 1, "exclusiveMinimum": 0},
            {"type": "integer", "minimum": 1},
        ),
        (
            {"type": "integer", "maximum": 0, "exclusiveMaximum": 1},
            {"type": "integer", "maximum": 0},
        ),
        (
            {"type": "integer", "minimum": 1, "multipleOf": 2},
            {"type": "integer", "minimum": 2, "multipleOf": 2},
        ),
        (
            {"type": "integer", "maximum": 1, "multipleOf": 2},
            {"type": "integer", "maximum": 0, "multipleOf": 2},
        ),
        (
            {"required": ["a"], "dependencies": {"a": ["b"], "b": ["c"], "x": ["y"]}},
            {"required": ["a", "b", "c"], "dependencies": {"x": ["y"]}},
        ),
        (
            {"type": "number", "minimum": 0, "exclusiveMaximum": 6, "multipleOf": 3},
            {"type": "number", "minimum": 0, "exclusiveMaximum": 6, "multipleOf": 3},
        ),
        ({"enum": ["aa", 2, "z", None, 1]}, {"enum": [None, 1, 2, "z", "aa"]}),
        (
            {"contains": {}, "items": {}, "type": "array"},
            {"minItems": 1, "type": "array"},
        ),
        ({"anyOf": [{}, {"type": "null"}]}, {}),
        ({"uniqueItems": False}, {}),
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
        (
            [
                {"contains": {}, "items": {}, "type": "array"},
                {"items": False, "type": "array"},
            ],
            {"not": {}},
        ),
    ]
    + [
        ([{lo: 0, hi: 9}, {lo: 1, hi: 10}], {lo: 1, hi: 9})
        for lo, hi in [
            ("minimum", "maximum"),
            ("exclusiveMinimum", "exclusiveMaximum"),
            ("minLength", "maxLength"),
            ("minItems", "maxItems"),
            ("minProperties", "maxProperties"),
        ]
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
        [
            {"properties": {"a": {"patternProperties": {".": {}}}}},
            {"properties": {"a": {"patternProperties": {"ab": {"type": "boolean"}}}}},
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
    assert is_valid(ic, s1)
    assert is_valid(ic, s2)
    assert is_valid(i1, s2) == is_valid(i1, combined)
    assert is_valid(i2, s1) == is_valid(i2, combined)


@pytest.mark.xfail
def test_merge_semantics_regressions():
    # TODO: broken because of missing interaction between
    # properties and additionalProperties - merged should return None for this.
    s1 = {"properties": {"": {"type": "string"}}, "required": [""], "type": "object"}
    s2 = {"additionalProperties": {"type": "null"}, "type": "object"}
    instance = {"": ""}
    combined = merged([s1, s2])
    assert is_valid(instance, combined) == (
        is_valid(instance, s1) and is_valid(instance, s2)
    )


@pytest.mark.xfail
def test_merge_should_notice_required_disallowed_properties():
    # The required "name" property is banned by the additionalProperties: False
    # See https://github.com/Zac-HD/hypothesis-jsonschema/issues/30 for details.
    schemas = [
        {"type": "object", "properties": {"name": True}, "required": ["name"]},
        {"type": "object", "properties": {"id": True}, "additionalProperties": False},
    ]
    assert merged(schemas) == FALSEY


def test_resolution_checks_resolver_is_valid():
    with pytest.raises(InvalidArgument):
        resolve_all_refs({}, resolver="not a resolver")


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
def _canonicalises_to_equivalent_fixpoint(data):
    # This function isn't executed by pytest, only by FuzzBuzz - we want to parametrize
    # over schemas for differnt types there, but have to supply *all* args here.
    schema = data.draw(json_schemata(), label="schema")
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)
    try:
        strat = from_schema(cc)
    except InvalidArgument:
        # e.g. array of unique {type: integers}, with too few allowed integers
        assume(False)
    instance = data.draw(JSON_STRATEGY | strat, label="instance")
    assert is_valid(instance, schema) == is_valid(instance, cc)
    jsonschema.validators.validator_for(schema).check_schema(schema)


# Expose fuzz targets in a form that FuzzBuzz can understand (no dotted names)
fuzz_canonical_json_encoding = test_canonical_json_encoding.hypothesis.fuzz_one_input
fuzz_merge_semantics = test_merge_semantics.hypothesis.fuzz_one_input
fuzz_self_merge_eq_canonicalish = (
    test_self_merge_eq_canonicalish.hypothesis.fuzz_one_input
)
fuzz_canonicalises_to_equivalent_fixpoint = (
    _canonicalises_to_equivalent_fixpoint.hypothesis.fuzz_one_input
)
