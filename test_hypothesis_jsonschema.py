"""Tests for the hypothesis-jsonschema library."""

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import HealthCheck, assume, given, note, reject, settings
from hypothesis.errors import InvalidArgument

import hypothesis_jsonschema
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
    get_type,
    is_valid,
    json_schemata,
    merged,
)


class Version(NamedTuple):
    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, string):
        return cls(*map(int, string.split(".")))


@lru_cache()
def get_releases():
    pattern = re.compile(r"^#### (\d+\.\d+\.\d+) - (\d\d\d\d-\d\d-\d\d)$")
    with open(Path(__file__).parent / "README.md") as f:
        return tuple(
            (Version.from_string(match.group(1)), match.group(2))
            for match in map(pattern.match, f)
            if match is not None
        )


def test_last_release_against_changelog():
    last_version, last_date = get_releases()[0]
    assert last_version == Version.from_string(hypothesis_jsonschema.__version__)
    assert last_date <= date.today().isoformat()


def test_changelog_is_ordered():
    versions, dates = zip(*get_releases())
    assert versions == tuple(sorted(versions, reverse=True))
    assert dates == tuple(sorted(dates, reverse=True))


def test_version_increments_are_correct():
    # We either increment the patch version by one, increment the minor version
    # and reset the patch, or increment major and reset both minor and patch.
    versions, _ = zip(*get_releases())
    for prev, current in zip(versions[1:], versions):
        assert prev < current  # remember that `versions` is newest-first
        assert current in (
            prev._replace(patch=prev.patch + 1),
            prev._replace(minor=prev.minor + 1, patch=0),
            prev._replace(major=prev.major + 1, minor=0, patch=0),
        ), f"{current} does not follow {prev}"


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


@given(from_schema(True))
def test_boolean_true_is_valid_schema_and_resolvable(_):
    """...even though it's currently broken in jsonschema."""


@pytest.mark.parametrize(
    "group,result",
    [
        ([{"type": []}, {}], {"not": {}}),
        ([{"type": "null"}, {"const": 0}], {"not": {}}),
        ([{"type": "null"}, {"enum": [0]}], {"not": {}}),
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


@pytest.mark.parametrize(
    "schema",
    [
        None,
        False,
        {"type": "an unknown type"},
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
    "draft4/oneOf with missing optional property",
    "draft7/oneOf with missing optional property",
    # Something weird about a null that should be a string??  TODO: debug that.
    "Datalogic Scan2Deploy Android file",
    "Datalogic Scan2Deploy CE file",
    # Just not handling this one correctly yet
    "draft4/additionalProperties should not look in applicators",
    "draft7/additionalProperties should not look in applicators",
    "draft7/ECMA 262 regex escapes control codes with \\c and lower letter",
    "draft7/ECMA 262 regex escapes control codes with \\c and upper letter",
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
        if n.endswith("/oneOf complex types"):
            # oneOf on property names means only objects are valid,
            # but it's a very filter-heavy way to express that...
            # TODO: see if we can auto-detect this, fix it, and emit a warning.
            assert "type" not in corpus[n]
            corpus[n]["type"] = "object"
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
OVERLAPPING_PATTERNS_SCHEMA = {
    "type": "string",
    "patternProperties": {
        r"\A[ab]{1,2}\Z": {},
        r"\Aa[ab]\Z": {"type": "integer"},
        r"\A[ab]b\Z": {"type": "string"},
    },
    "additionalProperties": False,
    "minimumProperties": 1,
}


@given(from_schema(OVERLAPPING_PATTERNS_SCHEMA))
def test_handles_overlapping_patternProperties(value):
    jsonschema.validate(value, OVERLAPPING_PATTERNS_SCHEMA)
    assert "ab" not in value
