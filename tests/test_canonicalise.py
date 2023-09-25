"""Tests for the hypothesis-jsonschema library."""

import jsonschema
import pytest
from gen_schemas import gen_number, json_schemata, schema_strategy_params
from hypothesis import HealthCheck, assume, given, note, settings, strategies as st
from hypothesis.errors import InvalidArgument

from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema._canonicalise import (
    FALSEY,
    canonicalish,
    get_type,
    make_validator,
    merged,
    next_up,
)
from hypothesis_jsonschema._from_schema import JSON_STRATEGY
from hypothesis_jsonschema._resolve import resolve_all_refs


def is_valid(instance, schema):
    return make_validator(schema).is_valid(instance)


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
    make_validator(schema)


@pytest.mark.parametrize(
    "schema, examples",
    [({"type": "integer", "multipleOf": 0.75}, [1.5e308])],
)
def test_canonicalises_to_equivalent_fixpoint_examples(schema, examples):
    """Check that an object drawn from an arbitrary schema is valid.

    This is used to record past regressions from the test above.
    """
    cc = canonicalish(schema)
    assert cc == canonicalish(cc)
    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)
    validator.check_schema(cc)
    for instance in examples:
        assert is_valid(instance, schema) == is_valid(instance, cc)


def test_dependencies_canonicalises_to_fixpoint():
    """Check that an object drawn from an arbitrary schema is valid."""
    cc = canonicalish(
        {"required": [""], "properties": {"": {}}, "dependencies": {"": [""]}}
    )
    assert cc == canonicalish(cc)


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
        {"type": "array", "items": [True], "minItems": 2, "additionalItems": False},
        {
            "type": "array",
            "items": [True, False, True],
            "minItems": 3,
            "additionalItems": {"type": "null"},
        },
        {"type": "integer", "minimum": 2, "maximum": 1},
        {"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 3},
        {"type": "number", "exclusiveMinimum": 0, "maximum": 0},
        {"type": "number", "exclusiveMinimum": -0.0, "maximum": -0.0},
        {"type": "number", "minimum": 0, "exclusiveMaximum": 0},
        {"type": "number", "minimum": -0.0, "exclusiveMaximum": -0.0},
        {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 3,
            "multipleOf": 3,
        },
        {"not": {"type": ["integer", "number"]}, "type": "number"},
        {"not": {"anyOf": [{"type": "integer"}, {"type": "number"}]}, "type": "number"},
        {"not": {"enum": [1, 2, 3]}, "const": 2},
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
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 0},
            "uniqueItems": True,
            "minItems": 2,
        },
        {"type": "array", "items": {"type": "integer"}, "contains": {"type": "string"}},
        {
            # only seven allowed elements: [], [1], [2], [1, 1], [1, 2], [2, 1], [2, 2]
            "type": "array",
            "items": {"type": "array", "items": {"enum": [1, 2]}, "maxItems": 2},
            "minItems": 8,
            "uniqueItems": True,
        },
        {"type": "object", "required": ["a"], "properties": {"a": False}},
    ],
)
def test_canonicalises_to_empty(schema):
    assert canonicalish(schema) == {"not": {}}, (schema, canonicalish(schema))


@pytest.mark.parametrize(
    "schema,expected",
    [
        ({"type": get_type({})}, {}),
        ({"required": []}, {}),
        ({"type": "integer", "not": {"type": "string"}}, {"type": "integer"}),
        ({"type": "integer", "multipleOf": 1 / 32}, {"type": "integer"}),
        ({"type": "number", "multipleOf": 1.0}, {"type": "integer"}),
        ({"type": "number", "multipleOf": -3.0}, {"type": "integer", "multipleOf": 3}),
        (
            {"type": "number", "multipleOf": 0.75, "not": {"multipleOf": 1.25}},
            {"type": "number", "multipleOf": 0.75, "not": {"multipleOf": 1.25}},
        ),
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
            {"type": "integer", "minimum": 0, "maximum": 3, "multipleOf": 3},
        ),
        (
            {"type": "number", "minimum": 0, "exclusiveMaximum": next_up(0.0)},
            {"const": 0},
        ),
        (
            {"type": "number", "exclusiveMinimum": 1.5, "maximum": next_up(1.5)},
            {"const": next_up(1.5)},
        ),
        (
            {
                "type": "number",
                "minimum": 1.5,
                "exclusiveMaximum": 2.5,
                "multipleOf": 0.5,
            },
            {
                "type": "number",
                "minimum": 1.5,
                "exclusiveMaximum": 2.5,
                "multipleOf": 0.5,
            },
        ),
        ({"enum": ["aa", 2, "z", None, 1]}, {"enum": [None, 1, 2, "z", "aa"]}),
        (
            {"contains": {}, "items": {}, "type": "array"},
            {"minItems": 1, "type": "array"},
        ),
        ({"anyOf": [{}, {"type": "null"}]}, {}),
        ({"anyOf": [{"anyOf": [{"anyOf": [{"type": "null"}]}]}]}, {"const": None}),
        (
            {
                "anyOf": [
                    {"type": "string"},
                    {"anyOf": [{"type": "number"}, {"type": "array"}]},
                ]
            },
            {"type": ["number", "string", "array"]},
        ),
        (
            {"anyOf": [{"type": "integer"}, {"type": "number"}]},
            {"type": "number"},
        ),
        (
            {
                "anyOf": [{"type": "string"}, {"type": "number"}],
                "type": ["string", "object"],
            },
            {"type": "string"},
        ),
        ({"uniqueItems": False}, {}),
        (
            {
                "type": "array",
                "items": [True, True],
                "minItems": 3,
                "additionalItems": {"type": "null"},
            },
            {
                "type": "array",
                "items": [{}, {}],
                "minItems": 3,
                "additionalItems": {"const": None},
            },
        ),
        (
            {
                "type": "array",
                "items": {"type": "number", "multipleOf": 0.5},
                "contains": {"type": "number", "multipleOf": 0.75},
            },
            {
                "type": "array",
                "minItems": 1,
                "items": {"type": "number", "multipleOf": 0.5},
                "contains": {"type": "number", "multipleOf": 0.75},
            },
        ),
        (
            {"type": "array", "items": {"const": 1}, "uniqueItems": True},
            {
                "type": "array",
                "items": {"const": 1},
                "uniqueItems": True,
                "maxItems": 1,
            },
        ),
        (
            {
                "anyOf": [
                    {"const": "a"},
                    {"anyOf": [{"anyOf": [{"const": "c"}]}, {"const": "b"}]},
                ]
            },
            # TODO: could be {"enum": ["a", "b", "c"]},
            {"anyOf": [{"const": "a"}, {"const": "b"}, {"const": "c"}]},
        ),
        (
            {"if": {"type": "null"}, "then": {"type": "null"}},
            {},
        ),
        (
            {"if": {"type": "null"}, "then": {"type": "null"}, "else": {}},
            {},
        ),
        (
            {"if": {"type": "null"}, "then": {}, "else": {}},
            {},
        ),
        (
            {"if": {"type": "integer"}, "then": {}, "else": {}, "type": "number"},
            {"type": "number"},
        ),
        (
            {"allOf": [{"multipleOf": 1.5}], "multipleOf": 1.5},
            {"multipleOf": 1.5},
        ),
        (
            {"type": "integer", "allOf": [{"multipleOf": 0.5}, {"multipleOf": 1e308}]},
            {"type": "integer", "multipleOf": 1e308},
        ),
        (
            {
                "additionalProperties": {"not": {}},
                "properties": {"a": {"not": {}}},
                "type": "object",
            },
            {"maxProperties": 0, "type": "object"},
        ),
        (
            {
                "additionalProperties": {"not": {}},
                "properties": {"a": {"not": {}}, "b": {}},
                "type": "object",
            },
            {
                "additionalProperties": {"not": {}},
                "properties": {"b": {}},
                "maxProperties": 1,
                "type": "object",
            },
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
        ([{"type": "string"}, {"enum": ["abc", True]}], {"const": "abc"}),
        ([{"type": "null"}, {"type": ["null", "boolean"]}], {"const": None}),
        ([{"type": "integer"}, {"maximum": 20}], {"type": "integer", "maximum": 20}),
        ([{"type": "integer"}, {"type": "number"}], {"type": "integer"}),
        ([{"multipleOf": 0.25}, {"multipleOf": 0.5}], {"multipleOf": 0.5}),
        ([{"multipleOf": 0.5}, {"multipleOf": 1.5}], {"multipleOf": 1.5}),
        (
            [
                {"type": "string", "format": "color"},
                {"type": "string", "format": "date-fullyear"},
            ],
            None,
        ),
        (
            [
                {"type": "integer", "multipleOf": 4},
                {"type": "integer", "multipleOf": 6},
            ],
            {"type": "integer", "multipleOf": 12},
        ),
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
        (
            [
                {"allOf": [{"multipleOf": 0.5}, {"multipleOf": 0.75}]},
                {"allOf": [{"multipleOf": 0.5}, {"multipleOf": 1.25}]},
            ],
            {
                "allOf": [
                    {"multipleOf": 0.5},
                    {"multipleOf": 0.75},
                    {"multipleOf": 1.25},
                ]
            },
        ),
        (
            [
                {"additionalProperties": {"type": "null"}},
                {"additionalProperties": {"type": "boolean"}},
            ],
            {"additionalProperties": {"not": {}}},
        ),
        (
            [
                {"additionalProperties": {"type": "null"}, "properties": {"foo": {}}},
                {"additionalProperties": {"type": "boolean"}},
            ],
            {
                "properties": {"foo": {"enum": [False, True]}},
                "additionalProperties": {"not": {}},
                "maxProperties": 1,
            },
        ),
        (
            [
                {
                    "properties": {"": {"type": "string"}},
                    "required": [""],
                    "type": "object",
                },
                {"additionalProperties": {"type": "null"}, "type": "object"},
            ],
            {"not": {}},
        ),
        (
            [
                {"additionalProperties": {"patternProperties": {".": {}}}},
                {"additionalProperties": {"patternProperties": {"a": {}}}},
            ],
            {"additionalProperties": {"patternProperties": {".": {}, "a": {}}}},
        ),
        (
            [
                {"patternProperties": {".": {"enum": [None, True]}}},
                {"properties": {"ab": {"type": "boolean"}}},
            ],
            {
                "patternProperties": {".": {"enum": [None, True]}},
                "properties": {"ab": {"const": True}},
            },
        ),
        (
            [
                {"type": "array", "contains": {"type": "integer"}},
                {"type": "array", "contains": {"type": "number"}},
            ],
            {"type": "array", "contains": {"type": "integer"}, "minItems": 1},
        ),
        (
            [{"not": {"enum": [1, 2, 3]}}, {"not": {"enum": ["a", "b", "c"]}}],
            {"not": {"anyOf": [{"enum": ["a", "b", "c"]}, {"enum": [1, 2, 3]}]}},
        ),
        (
            [{"dependencies": {"a": ["b"]}}, {"dependencies": {"a": ["c"]}}],
            {"dependencies": {"a": ["b", "c"]}},
        ),
        (
            [{"dependencies": {"a": ["b"]}}, {"dependencies": {"b": ["c"]}}],
            {"dependencies": {"a": ["b"], "b": ["c"]}},
        ),
        (
            [
                {"dependencies": {"a": ["b"]}},
                {"dependencies": {"a": {"properties": {"b": {"type": "string"}}}}},
            ],
            {
                "dependencies": {
                    "a": {"required": ["b"], "properties": {"b": {"type": "string"}}}
                },
            },
        ),
        (
            [
                {"dependencies": {"a": {"properties": {"b": {"type": "string"}}}}},
                {"dependencies": {"a": ["b"]}},
            ],
            {
                "dependencies": {
                    "a": {"required": ["b"], "properties": {"b": {"type": "string"}}}
                },
            },
        ),
        (
            [
                {"dependencies": {"a": {"pattern": "a"}}},
                {"dependencies": {"a": {"pattern": "b"}}},
            ],
            None,
        ),
        ([{"items": {"pattern": "a"}}, {"items": {"pattern": "b"}}], None),
        ([{"items": [{"pattern": "a"}]}, {"items": [{"pattern": "b"}]}], None),
        (
            [
                {"items": [{}], "additionalItems": {"pattern": "a"}},
                {"items": [{}], "additionalItems": {"pattern": "b"}},
            ],
            None,
        ),
        (
            [
                {"items": [{}, {"type": "string"}], "additionalItems": False},
                {"items": [{"type": "string"}]},
            ],
            {
                "items": [{"type": "string"}, {"type": "string"}],
                "additionalItems": FALSEY,
            },
        ),
        (
            [
                {"items": [{}, {"type": "string"}], "additionalItems": False},
                {"items": {"type": "string"}},
            ],
            {
                "items": [{"type": "string"}, {"type": "string"}],
                "additionalItems": FALSEY,
            },
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


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(json_schemata())
def test_self_merge_eq_canonicalish(schema):
    m = merged([schema, schema])
    assert m == canonicalish(schema)


def _merge_semantics_helper(data, s1, s2, combined):
    note(f"combined={combined!r}")
    ic = data.draw(from_schema(combined), label="combined")
    i1 = data.draw(from_schema(s1), label="s1")
    i2 = data.draw(from_schema(s2), label="s2")
    assert is_valid(ic, s1)
    assert is_valid(ic, s2)
    assert is_valid(i1, s2) == is_valid(i1, combined)
    assert is_valid(i2, s1) == is_valid(i2, combined)


@pytest.mark.xfail(
    strict=False, reason="https://github.com/python-jsonschema/jsonschema/issues/1159"
)
@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(st.data(), json_schemata(), json_schemata())
def test_merge_semantics(data, s1, s2):
    assume(canonicalish(s1) != FALSEY and canonicalish(s2) != FALSEY)
    combined = merged([s1, s2])
    assume(combined is not None)
    assert combined == merged([s2, s1])  # union is commutative
    assume(combined != FALSEY)
    _merge_semantics_helper(data, s1, s2, combined)


@pytest.mark.xfail(
    strict=False, reason="https://github.com/python-jsonschema/jsonschema/issues/1159"
)
@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(
    st.data(),
    gen_number(kind="integer") | gen_number(kind="number"),
    gen_number(kind="integer") | gen_number(kind="number"),
)
def test_can_almost_always_merge_numeric_schemas(data, s1, s2):
    assume(canonicalish(s1) != FALSEY and canonicalish(s2) != FALSEY)
    combined = merged([s1, s2])
    if combined is None:
        # The ONLY case in which we can't merge numeric schemas is when
        # they both contain multipleOf keys with distinct non-integer values.
        mul1, mul2 = s1["multipleOf"], s2["multipleOf"]
        assert isinstance(mul1, float) or isinstance(mul2, float)
        assert mul1 != mul2
        # TODO: work out why this started failing with
        #   s1={'type': 'integer', 'multipleOf': 2},
        #   s2={'type': 'integer', 'multipleOf': 0.3333333333333333}
        # ratio = max(mul1, mul2) / min(mul1, mul2)
        # assert ratio != int(ratio)  # i.e. x=0.5, y=2 (ratio=4.0) should work
    elif combined != FALSEY:
        _merge_semantics_helper(data, s1, s2, combined)


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


def test_canonicalise_is_only_valid_for_schemas():
    with pytest.raises(InvalidArgument):
        canonicalish("not a schema")


def test_validators_use_proper_draft():
    # See GH-66
    schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "not": {
            "allOf": [
                {"exclusiveMinimum": True, "minimum": 0},
                {"exclusiveMaximum": True, "maximum": 10},
            ]
        },
    }
    cc = canonicalish(schema)
    jsonschema.validators.validator_for(cc).check_schema(cc)


def test_reference_resolver_issue_65_regression():
    schema = {
        "allOf": [{"$ref": "#/definitions/ref"}, {"required": ["foo"]}],
        "properties": {"foo": {}},
        "definitions": {"ref": {"maxProperties": 1}},
        "type": "object",
    }
    res = resolve_all_refs(schema)
    can = canonicalish(res)
    assert "$ref" not in res
    assert "$ref" not in can
    for s in (schema, res, can):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({}, s)
