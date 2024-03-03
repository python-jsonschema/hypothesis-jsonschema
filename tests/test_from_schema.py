"""Tests for the hypothesis-jsonschema library."""

import json
import re
import warnings
from pathlib import Path

import jsonschema
import pytest
from gen_schemas import schema_strategy_params
from hypothesis import (
    HealthCheck,
    Phase,
    assume,
    given,
    note,
    reject,
    settings,
    strategies as st,
)
from hypothesis.errors import FailedHealthCheck, HypothesisWarning, InvalidArgument
from hypothesis.internal.compat import PYPY
from hypothesis.internal.reflection import proxies
from hypothesis.strategies._internal.regex import IncompatibleWithAlphabet

from hypothesis_jsonschema._canonicalise import (
    HypothesisRefResolutionError,
    canonicalish,
    make_validator,
)
from hypothesis_jsonschema._from_schema import from_schema
from hypothesis_jsonschema._resolve import resolve_all_refs

# We use this as a placeholder for all schemas which resolve to nothing()
# but do not canonicalise to FALSEY
INVALID_REGEX_SCHEMA = {"type": "string", "pattern": "["}


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
@given(data=st.data())
@schema_strategy_params
def test_generated_data_matches_schema(schema_strategy, data):
    """Check that an object drawn from an arbitrary schema is valid."""
    schema = data.draw(schema_strategy)
    note(f"{schema=}")
    try:
        value = data.draw(from_schema(schema), "value from schema")
    except InvalidArgument:
        reject()
    assert make_validator(schema).is_valid(value)
    # This checks that our canonicalisation is semantically equivalent.
    assert make_validator(canonicalish(schema)).is_valid(value)


@given(from_schema(True))
def test_boolean_true_is_valid_schema_and_resolvable(_):
    """...even though it's currently broken in jsonschema."""


@pytest.mark.parametrize(
    "schema",
    [
        None,
        False,
        {"type": "an unknown type"},
        {"allOf": [{"type": "boolean"}, {"const": None}]},
        {"allOf": [{"type": "boolean"}, {"enum": [None]}]},
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "maximum": 10,
            "exclusiveMaximum": True,
        },
    ],
)
def test_invalid_schemas_raise(schema):
    """Trigger all the validation exceptions for full coverage."""
    with pytest.raises(Exception):
        from_schema(schema).example()


@pytest.mark.parametrize(
    "schema",
    [
        INVALID_REGEX_SCHEMA,
        {"type": "string", "pattern": "[", "format": "color"},
        {"type": "object", "patternProperties": {"[": False}},
        {"type": "object", "patternProperties": {"[": False}, "required": ["a"]},
    ],
)
def test_invalid_regex_emit_warning(schema):
    with pytest.warns(UserWarning):
        from_schema(schema).validate()


@given(
    from_schema(
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "maximum": 10,
            "exclusiveMaximum": True,
        }
    )
)
def test_can_generate_with_explicit_schema_version(_):
    pass


INVALID_SCHEMAS = {
    # Empty list for requires, which is invalid
    "Release Drafter configuration file",
    # Many, many schemas have invalid $schema keys, which emit a warning (-Werror)
    "A JSON schema for CRYENGINE projects (.cryproj files)",
    "JSDoc configuration file",
    "Static Analysis Results Format (SARIF) External Property File Format, Version 2.1.0-rtm.2",
    "Static Analysis Results Format (SARIF) External Property File Format, Version 2.1.0-rtm.3",
    "Static Analysis Results Format (SARIF) External Property File Format, Version 2.1.0-rtm.4",
    "Static Analysis Results Format (SARIF) External Property File Format, Version 2.1.0-rtm.5",
    "Static Analysis Results Format (SARIF), Version 2.1.0-rtm.2",
}
NON_EXISTENT_REF_SCHEMAS = {
    "Cirrus CI configuration files",
    "The Bamboo Specs allows you to define Bamboo configuration as code, and have corresponding plans/deployments created or updated automatically in Bamboo",
    # Special case - reference is valid, but target is list-format `items` rather than a subschema
    "TypeScript Lint configuration file",
}
UNSUPPORTED_SCHEMAS = {
    # Technically valid, but using regex patterns not supported by Python
    "draft4/unicode digits are more than 0 through 9",
    "draft4/unicode semantics should be used for all patternProperties matching",
    "draft7/unicode digits are more than 0 through 9",
    "draft7/unicode semantics should be used for all pattern matching",
    "draft7/unicode semantics should be used for all patternProperties matching",
    "draft7/ECMA 262 regex escapes control codes with \\c and lower letter",
    "draft7/ECMA 262 regex escapes control codes with \\c and upper letter",
    "JSON schema for nodemon.json configuration files.",
    "JSON Schema for mime type collections",
}
SKIP_ON_PYPY_SCHEMAS = {
    # Cause crashes or recursion errors, but only under PyPy
    "Swagger API 2.0 schema",
    "Language grammar description files in Textmate and compatible editors",
}
FLAKY_SCHEMAS = {
    # The following schemas refer to an `$id` rather than a JSON pointer.
    # This is valid, but not supported by the Python library - see e.g.
    # https://json-schema.org/understanding-json-schema/structuring.html#using-id-with-ref
    "draft4/Location-independent identifier",
    "draft7/Location-independent identifier",
    # Yep, lists of lists of lists of lists of lists of integers are HealthCheck-slow
    # TODO: write a separate test with healthchecks disabled?
    "draft4/nested items",
    "draft7/nested items",
    "draft4/oneOf with missing optional property",
    "draft7/oneOf with missing optional property",
    # Sometimes unsatisfiable.  TODO: improve canonicalisation to remove filters
    "JSCS configuration file",  # https://github.com/Zac-HD/hypothesis-jsonschema/pull/78#issuecomment-803519293
    "Drone CI configuration file",
    "PHP Composer configuration file",
    "Pyrseas database schema versioning for Postgres databases, v0.8",
    # Apparently we're not handling this one correctly?
    "draft4/additionalProperties should not look in applicators",
    "draft7/additionalProperties should not look in applicators",
    # $id (sometimes) rejected as invalid/unknown URL type
    "A JSON schema for a Dolittle bounded context's artifacts",
    "A JSON schema for a Dolittle bounded context's resource configurations",
    # This one fails because of a hard-to-find (and on the surface impossible)
    # counterexample involving oneOf, which doesn't fail if validated directly!
    # {'requirements': {'': [{'location': None, 'rule': 'dir'}]}}
    "CLI config for enforcing environment settings",
    # These ones fail under jsonschema >= 4.0.0
    # TODO: work out why and fix it; this is pure "ignore so we can ship it"
    "draft7/Recursive references between schemas",
    "draft7/$id inside an unknown keyword is not a real identifier",
    "draft7/refs with relative uris and defs",
    "draft7/relative refs with absolute uris and defs",
}
SLOW_SCHEMAS = {
    "snapcraft project  (https://snapcraft.io)",
    "batect configuration file",
    "UI5 Tooling Configuration File (ui5.yaml)",
    "Renovate config file (https://github.com/renovatebot/renovate)",
    "Renovate config file (https://renovatebot.com/)",
    "Jenkins X Pipeline YAML configuration files",
    "TypeScript compiler configuration file",
    "JSON Schema for GraphQL Mesh config file",
    "Configuration file for stylelint",
    "Travis CI configuration file",
    "JSON schema for ESLint configuration files",
    "Ansible task files-2.0",
    "Ansible task files-2.1",
    "Ansible task files-2.2",
    "Ansible task files-2.3",
    "Ansible task files-2.4",
    "Ansible task files-2.5",
    "Ansible task files-2.6",
    "Ansible task files-2.7",
    "Ansible task files-2.9",
    "JSON Schema for GraphQL Code Generator config file",
    "Schema for CircleCI 2.0 config files",
    "Schema for Camel K YAML DSL",
    "The AWS Serverless Application Model (AWS SAM, previously known as Project Flourish) extends AWS CloudFormation to provide a simplified way of defining the Amazon API Gateway APIs, AWS Lambda functions, and Amazon DynamoDB tables needed by your serverless application.",
    "AWS CloudFormation provides a common language for you to describe and provision all the infrastructure resources in your cloud environment.",
    "JSON API document",
    "Prometheus configuration file",
    "JSON schema for electron-build configuration file.",
    "Pyrseas database schema versioning for Postgres databases, v0.8",
    # oneOf on property names means only objects are valid, but it's a very
    # filter-heavy way to express that.  TODO: canonicalise oneOf to anyOf.
    "draft7/oneOf complex types",
}

with open(Path(__file__).parent / "corpus-schemastore-catalog.json") as f:
    catalog = json.load(f)
with open(Path(__file__).parent / "corpus-suite-schemas.json") as f:
    suite, invalid_suite = json.load(f)
with open(Path(__file__).parent / "corpus-reported.json") as f:
    reported = json.load(f)
    assert set(reported).isdisjoint(suite)
    suite.update(reported)


def to_name_params(corpus):
    for n in sorted(corpus):
        if n in INVALID_SCHEMAS | NON_EXISTENT_REF_SCHEMAS:
            continue
        if n in UNSUPPORTED_SCHEMAS:
            continue
        if n in SKIP_ON_PYPY_SCHEMAS:
            yield pytest.param(n, marks=pytest.mark.skipif(PYPY, reason="broken"))
        elif n in SLOW_SCHEMAS | FLAKY_SCHEMAS:
            yield pytest.param(n, marks=pytest.mark.skip)
        else:
            if isinstance(corpus[n], dict) and "$schema" in corpus[n]:
                jsonschema.validators.validator_for(corpus[n]).check_schema(corpus[n])
            yield n


@pytest.mark.parametrize("name", sorted(INVALID_SCHEMAS))
def test_invalid_schemas_are_invalid(name):
    with pytest.raises(Exception):
        jsonschema.validators.validator_for(catalog[name]).check_schema(catalog[name])


@pytest.mark.parametrize("name", sorted(NON_EXISTENT_REF_SCHEMAS))
def test_invalid_ref_schemas_are_invalid(name):
    with pytest.raises(Exception):
        resolve_all_refs(catalog[name])


RECURSIVE_REFS = {
    # From upstream validation test suite
    "draft4/valid definition",
    "draft7/valid definition",
    "draft4/validate definition against metaschema",
    "draft7/validate definition against metaschema",
    "draft4/remote ref, containing refs itself",
    "draft7/remote ref, containing refs itself",
    "draft7/root pointer ref",
    # Schema also requires draft 03, which hypothesis-jsonschema doesn't support
    "A JSON Schema for ninjs by the IPTC. News and publishing information. See https://iptc.org/standards/ninjs/-1.0",
    # From schemastore
    "A JSON schema for Open API documentation files",
    "Avro Schema Avsc file",
    "AWS CloudFormation provides a common language for you to describe and provision all the infrastructure resources in your cloud environment.",
    "JSON schema .NET template files",
    "AppVeyor CI configuration file",
    "JSON Document Transofrm",
    "JSON Linked Data files",
    "Meta-validation schema for JSON Schema Draft 4",
    "JSON schema for vim plugin addon-info.json metadata files",
    "Meta-validation schema for JSON Schema Draft 7",
    "Neotys as-code load test specification, more at: https://github.com/Neotys-Labs/neoload-cli",
    "Metadata spec v1.26.4 for KSP-CKAN",
    "Digital Signature Service Core Protocols, Elements, and Bindings Version 2.0",
    "Opctl schema for describing an op",
    "Metadata spec v1.27 for KSP-CKAN",
    "PocketMine plugin manifest file",
    "BuckleScript configuration file",
    "Schema for CircleCI 2.0 config files",
    "Source Map files version 3",
    "Schema for Minecraft Bukkit plugin description files",
    "Swagger API 2.0 schema",
    "Static Analysis Results Interchange Format (SARIF) version 1",
    "Static Analysis Results Format (SARIF), Version 2.1.0-rtm.4",
    "Static Analysis Results Interchange Format (SARIF) version 2",
    "Static Analysis Results Format (SARIF), Version 2.1.0-rtm.3",
    "Static Analysis Results Format (SARIF), Version 2.1.0-rtm.5",
    "Web component file",
    "Vega visualization specification file",
    "The AWS Serverless Application Model (AWS SAM, previously known as Project Flourish) extends AWS CloudFormation to provide a simplified way of defining the Amazon API Gateway APIs, AWS Lambda functions, and Amazon DynamoDB tables needed by your serverless application.",
    "Windows App localization file",
    "YAML schema for GitHub Workflow",
    "JSON-stat 2.0 Schema",
    "Vega-Lite visualization specification file",
    "Language grammar description files in Textmate and compatible editors",
    "JSON Schema for GraphQL Mesh Config gile-0.0.16",
    "Azure Pipelines YAML pipelines definition",
    "Action and rule configuration descriptor for Yippee-Ki-JSON transformations.-1.1.2",
    "Action and rule configuration descriptor for Yippee-Ki-JSON transformations.-latest",
    "Schema for Camel K YAML DSL",
}


def xfail_on_reference_resolve_error(f):
    @proxies(f)
    def inner(*args, **kwargs):
        _, name = args
        assert isinstance(name, str)
        try:
            f(*args, **kwargs)
            assert name not in RECURSIVE_REFS
        except (
            jsonschema.exceptions._RefResolutionError,
            W := getattr(jsonschema.exceptions, "_WrappedReferencingError", ()),  # noqa
        ) as err:
            if isinstance(err, W) and isinstance(
                err._wrapped, jsonschema.exceptions._Unresolvable
            ):
                pytest.xfail()
            if (
                isinstance(err, HypothesisRefResolutionError)
                or isinstance(
                    getattr(err, "_cause", None), HypothesisRefResolutionError
                )
            ) and (
                "does not fetch remote references" in str(err)
                or name in RECURSIVE_REFS
                and "Could not resolve recursive references" in str(err)
            ):
                pytest.xfail()
            raise

    return inner


@pytest.mark.parametrize("name", to_name_params(catalog))
@settings(deadline=None, max_examples=5, suppress_health_check=list(HealthCheck))
@given(data=st.data())
@xfail_on_reference_resolve_error
def test_can_generate_for_real_large_schema(data, name):
    note(name)
    value = data.draw(from_schema(catalog[name]))
    jsonschema.validate(value, catalog[name])


@pytest.mark.parametrize("name", to_name_params(suite))
@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
    max_examples=20,
)
@given(data=st.data())
@xfail_on_reference_resolve_error
def test_can_generate_for_test_suite_schema(data, name):
    note(f"{suite[name]=}")
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


@pytest.mark.parametrize("name", to_name_params(suite))
@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
    max_examples=20,
)
@given(data=st.data())
@xfail_on_reference_resolve_error
def test_constrained_alphabet_generation(data, name):
    note(f"{suite[name]=}")
    try:
        value = data.draw(from_schema(suite[name], codec="ascii"))
    except IncompatibleWithAlphabet:
        pytest.skip()
    note(f"{value=}")
    try:
        jsonschema.validate(value, suite[name])
    except jsonschema.exceptions.SchemaError:
        jsonschema.Draft4Validator(suite[name]).validate(value)
    json.dumps(value).encode("ascii")


# This schema has overlapping patternProperties - this is OK, so long as they're
# merged or otherwise handled correctly, with the exception of the key "ab" which
# would have to be both an integer and a string (and is thus disallowed).
OVERLAPPING_PATTERNS_SCHEMA = {
    "type": "object",
    "patternProperties": {
        r"\A[ab]{1,2}\Z": {},
        r"\Aa[ab]\Z": {"type": "integer"},
        r"\A[ab]b\Z": {"type": "string"},
    },
    "additionalProperties": False,
    "minProperties": 1,
}


@given(from_schema(OVERLAPPING_PATTERNS_SCHEMA))
def test_handles_overlapping_patternproperties(value):
    jsonschema.validate(value, OVERLAPPING_PATTERNS_SCHEMA)
    assert isinstance(value, dict)
    assert len(value) >= 1
    assert "ab" not in value


# A dictionary with zero or one keys, which was always empty due to a bug.
SCHEMA = {
    "type": "object",
    "properties": {"key": {"type": "string"}},
    "additionalProperties": False,
}


@given(from_schema(SCHEMA))
def test_single_property_can_generate_nonempty(query):
    # See https://github.com/Zac-HD/hypothesis-jsonschema/issues/25
    assume(query)


UNIQUE_NUMERIC_ARRAY_SCHEMA = {
    "type": "array",
    "uniqueItems": True,
    "items": {"enum": [0, 0.0]},
    "minItems": 1,
}


@given(from_schema(UNIQUE_NUMERIC_ARRAY_SCHEMA))
def test_numeric_uniqueness(value):
    # NOTE: this kind of test should usually be embedded in corpus-reported.json,
    # but in this case the type of the enum elements matter and we don't want to
    # allow a flexible JSON loader to mess things up.
    jsonschema.validate(value, UNIQUE_NUMERIC_ARRAY_SCHEMA)


def test_draft03_not_supported():
    # Also checks that errors are deferred from importtime to runtime
    @given(from_schema({"$schema": "http://json-schema.org/draft-03/schema#"}))
    def f(_):
        raise AssertionError

    with pytest.raises(InvalidArgument):
        f()


@pytest.mark.parametrize("type_", ["integer", "number"])
def test_impossible_multiplier(type_):
    # Covering test for a failsafe branch, which explicitly returns nothing()
    # if scaling the bounds and taking their ceil/floor also inverts them.
    schema = {"maximum": -1, "minimum": -1, "multipleOf": 0.0009765625000000002}
    schema["type"] = type_
    strategy = from_schema(schema)
    strategy.validate()
    assert strategy.is_empty


def test_unsatisfiable_array_returns_nothing():
    schema = {
        "type": "array",
        "items": [],
        "additionalItems": INVALID_REGEX_SCHEMA,
        "minItems": 1,
    }
    with pytest.warns(UserWarning):
        strategy = from_schema(schema)
    strategy.validate()
    assert strategy.is_empty


ALLOF_CONTAINS = {
    "type": "array",
    "items": {"type": "string"},
    "allOf": [{"contains": {"const": "A"}}, {"contains": {"const": "B"}}],
}


@pytest.mark.xfail(raises=FailedHealthCheck)
@given(from_schema(ALLOF_CONTAINS))
def test_multiple_contains_behind_allof(value):
    # By placing *multiple* contains elements behind "allOf" we've disabled the
    # mixed-generation logic, and so we can't generate any valid instances at all.
    jsonschema.validate(value, ALLOF_CONTAINS)


@jsonschema.FormatChecker._cls_checks("card-test")
def validate_card_format(string):
    # For the real thing, you'd want use the Luhn algorithm; this is enough for tests.
    return bool(re.match(r"^\d{4} \d{4} \d{4} \d{4}$", string))


@pytest.mark.parametrize(
    "kw",
    [
        {"foo": "not a strategy"},
        {5: st.just("name is not a string")},
        {"card-test": st.just("not a valid card")},
        {"card-test": st.none()},  # Not a string
    ],
)
@given(data=st.data())
def test_custom_formats_validation(data, kw):
    s = from_schema({"type": "string", "format": "card-test"}, custom_formats=kw)
    with pytest.raises(InvalidArgument):
        data.draw(s)


@pytest.mark.parametrize(
    "schema",
    [
        {"required": ["\x00"]},
        {"properties": {"\x00": {"type": "integer"}}},
        {"dependencies": {"\x00": ["a"]}},
        {"dependencies": {"\x00": {"type": "integer"}}},
        {"required": ["\xff"]},
        {"properties": {"\xff": {"type": "integer"}}},
        {"dependencies": {"\xff": ["a"]}},
        {"dependencies": {"\xff": {"type": "integer"}}},
    ],
)
@settings(deadline=None)
@given(data=st.data())
def test_alphabet_name_validation(data, schema):
    with pytest.raises(InvalidArgument):
        data.draw(from_schema(schema, allow_x00=False, codec="ascii"))


@given(
    num=from_schema(
        {"type": "string", "format": "card-test"},
        custom_formats={"card-test": st.just("4111 1111 1111 1111")},
    )
)
def test_allowed_custom_format(num):
    assert num == "4111 1111 1111 1111"


@given(
    string=from_schema(
        {"type": "string", "format": "not registered"},
        custom_formats={"not registered": st.just("hello world")},
    )
)
def test_allowed_unknown_custom_format(string):
    assert string == "hello world"
    assert "not registered" not in jsonschema.FormatChecker().checkers


@given(data=st.data())
def test_overriding_standard_format(data):
    expected = "2000-01-01"
    schema = {"type": "string", "format": "full-date"}
    custom_formats = {"full-date": st.just(expected)}
    with pytest.warns(
        HypothesisWarning, match="Overriding standard format 'full-date'"
    ):
        value = data.draw(from_schema(schema, custom_formats=custom_formats))
    assert value == expected


with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

    @given(
        from_schema({"type": "array", "items": INVALID_REGEX_SCHEMA, "maxItems": 10})
    )
    def test_can_generate_empty_list_with_max_size_and_no_allowed_items(val):
        assert val == []

    @given(
        from_schema(
            {
                "type": "array",
                "items": [{"const": 1}, {"const": 2}, {"const": 3}],
                "additionalItems": INVALID_REGEX_SCHEMA,
                "maxItems": 10,
            }
        )
    )
    def test_can_generate_list_with_max_size_and_no_allowed_additional_items(val):
        assert val == [1, 2, 3]


@given(string=from_schema({"type": "string", "pattern": "^[a-z]+$"}))
def test_does_not_generate_trailing_newline_from_dollar_pattern(string):
    assert not string.endswith("\n")


@pytest.mark.xfail(strict=True, raises=UnicodeEncodeError)
@settings(phases=set(Phase) - {Phase.shrink})
@given(from_schema({"type": "string", "minLength": 100}, codec=None))
def test_can_find_non_utf8_string(value):
    value.encode()


@given(st.data())
def test_errors_on_unencodable_property_name(data):
    non_ascii_schema = {"type": "object", "properties": {"é": {"type": "integer"}}}
    data.draw(from_schema(non_ascii_schema, codec=None))
    with pytest.raises(InvalidArgument, match=r"'é' cannot be encoded as 'ascii'"):
        data.draw(from_schema(non_ascii_schema, codec="ascii"))


@settings(deadline=None)
@given(data=st.data())
def test_no_null_bytes(data):
    schema = {
        "type": "object",
        "properties": {
            "p1": {"type": "string"},
            "p2": {
                "type": "object",
                "properties": {"pp1": {"type": "string"}},
                "required": ["pp1"],
                "additionalProperties": False,
            },
            "p3": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["p1", "p2", "p3"],
        "additionalProperties": False,
    }
    example = data.draw(from_schema(schema, allow_x00=False))
    assert "\x00" not in example["p1"]
    assert "\x00" not in example["p2"]["pp1"]
    assert all("\x00" not in item for item in example["p3"])
