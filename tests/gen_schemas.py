"""Hypothesis strategies for generating JSON schemata."""

import re
from typing import Any, Dict, List, Union

import hypothesis.strategies as st
import jsonschema
import pytest
from hypothesis import assume

from hypothesis_jsonschema._canonicalise import JSONType, Schema, encode_canonical_json
from hypothesis_jsonschema._from_schema import (
    JSON_STRATEGY,
    REGEX_PATTERNS,
    STRING_FORMATS,
    from_schema,
)


def json_schemata() -> st.SearchStrategy[Union[bool, Schema]]:
    """Return a Hypothesis strategy for arbitrary JSON schemata.

    This strategy may generate anything that can be handled by `from_schema`,
    and is used to provide full branch coverage when testing this package.
    """
    return _json_schemata()


@st.composite  # type: ignore
def _json_schemata(draw: Any, recur: bool = True) -> Any:
    # Current version of jsonschema does not support boolean schemata,
    # but 3.0 will.  See https://github.com/Julian/jsonschema/issues/337
    options = [
        st.builds(dict),
        st.just({"type": "null"}),
        st.just({"type": "boolean"}),
        gen_number("integer"),
        gen_number("number"),
        gen_string(),
        st.builds(dict, const=JSON_STRATEGY),
        gen_enum(),
        st.booleans(),
    ]
    if recur:
        options += [
            gen_array(),
            gen_object(),
            # Conditional subschemata
            gen_if_then_else(),
            json_schemata().map(lambda v: assume(v and v is not True) and {"not": v}),
            st.builds(dict, anyOf=st.lists(json_schemata(), min_size=1)),
            st.builds(dict, oneOf=st.lists(json_schemata(), min_size=1, max_size=2)),
            st.builds(dict, allOf=st.lists(json_schemata(), min_size=1, max_size=2)),
        ]

    return draw(st.one_of(options))


def gen_enum() -> st.SearchStrategy[Dict[str, List[JSONType]]]:
    """Return a strategy for enum schema."""
    return st.fixed_dictionaries(
        {
            "enum": st.lists(
                JSON_STRATEGY, min_size=1, max_size=10, unique_by=encode_canonical_json
            )
        }
    )


@st.composite  # type: ignore
def gen_if_then_else(draw: Any) -> Schema:
    """Draw a conditional schema."""
    # Cheat by using identical if and then schemata, else accepting anything.
    if_schema = draw(json_schemata().filter(lambda v: bool(v and v is not True)))
    return {"if": if_schema, "then": if_schema, "else": {}}


@st.composite  # type: ignore
def gen_number(draw: Any, kind: str) -> Dict[str, Union[str, float]]:
    """Draw a numeric schema."""
    max_int_float = 2 ** 53
    lower = draw(st.none() | st.integers(-max_int_float, max_int_float))
    upper = draw(st.none() | st.integers(-max_int_float, max_int_float))
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    multiple_of = draw(st.none() | st.integers(2, 100))
    assume(None in (multiple_of, lower, upper) or multiple_of <= (upper - lower))
    assert kind in ("integer", "number")
    out: Dict[str, Union[str, float]] = {"type": kind}
    # Generate the latest draft supported by jsonschema.
    assert hasattr(jsonschema, "Draft7Validator")
    if lower is not None:
        if draw(st.booleans(), label="exclusiveMinimum"):
            out["exclusiveMinimum"] = lower - 1
        else:
            out["minimum"] = lower
    if upper is not None:
        if draw(st.booleans(), label="exclusiveMaximum"):
            out["exclusiveMaximum"] = upper + 1
        else:
            out["maximum"] = upper
    if multiple_of is not None:
        out["multipleOf"] = multiple_of
    return out


@st.composite  # type: ignore
def gen_string(draw: Any) -> Dict[str, Union[str, int]]:
    """Draw a string schema."""
    min_size = draw(st.none() | st.integers(0, 10))
    max_size = draw(st.none() | st.integers(0, 1000))
    if min_size is not None and max_size is not None and min_size > max_size:
        min_size, max_size = max_size, min_size
    pattern = draw(st.none() | REGEX_PATTERNS)
    format_ = draw(st.none() | st.sampled_from(sorted(STRING_FORMATS)))
    out: Dict[str, Union[str, int]] = {"type": "string"}
    if pattern is not None:
        out["pattern"] = pattern
    elif format_ is not None:
        out["format"] = format_
    if min_size is not None:
        out["minLength"] = min_size
    if max_size is not None:
        out["maxLength"] = max_size
    return out


@st.composite  # type: ignore
def gen_array(draw: Any) -> Schema:
    """Draw an array schema."""
    min_size = draw(st.none() | st.integers(0, 5))
    max_size = draw(st.none() | st.integers(2, 5))
    if min_size is not None and max_size is not None and min_size > max_size:
        min_size, max_size = max_size, min_size
    items = draw(
        st.builds(dict)
        | _json_schemata(recur=False)
        | st.lists(_json_schemata(recur=False), min_size=1, max_size=10)
    )
    out = {"type": "array", "items": items}
    if isinstance(items, list):
        increment = len(items)
        additional = draw(st.none() | _json_schemata(recur=False))
        if additional is not None:
            out["additionalItems"] = additional
        elif draw(st.booleans()):
            out["contains"] = draw(_json_schemata(recur=False).filter(bool))
            increment += 1
        if min_size is not None:
            min_size += increment
        if max_size is not None:
            max_size += increment
    else:
        if draw(st.booleans()):
            out["uniqueItems"] = True
        if items == {}:
            out["contains"] = draw(_json_schemata(recur=False))
    if min_size is not None:
        out["minItems"] = min_size
    if max_size is not None:
        out["maxItems"] = max_size
    return out


@st.composite  # type: ignore
def gen_object(draw: Any) -> Schema:
    """Draw an object schema."""
    out: Schema = {"type": "object"}
    names = draw(st.sampled_from([None, None, None, draw(gen_string())]))
    name_strat = st.text() if names is None else from_schema(names)
    required = draw(
        st.just(False) | st.lists(name_strat, min_size=1, max_size=5, unique=True)
    )

    # Trying to generate schemata that are consistent would mean dealing with
    # overlapping regex and names, and that would suck.  So instead we ensure that
    # there *are* no overlapping requirements, which is much easier.
    properties = draw(st.dictionaries(name_strat, _json_schemata(recur=False)))
    disjoint = REGEX_PATTERNS.filter(
        lambda r: all(re.search(r, string=name) is None for name in properties)
    )
    patterns = draw(st.dictionaries(disjoint, _json_schemata(recur=False), max_size=1))
    additional = draw(st.none() | _json_schemata(recur=False))

    min_size = draw(st.none() | st.integers(0, 5))
    max_size = draw(st.none() | st.integers(2, 5))
    if min_size is not None and max_size is not None and min_size > max_size:
        min_size, max_size = max_size, min_size

    if names is not None:
        out["propertyNames"] = names
    if required:
        out["required"] = required
        if min_size is not None:
            min_size += len(required)
        if max_size is not None:
            max_size += len(required)
    if min_size is not None:
        out["minProperties"] = min_size
    if max_size is not None:
        out["maxProperties"] = max_size
    if properties:
        out["properties"] = properties

        props = st.sampled_from(sorted(properties))
        if draw(st.integers(0, 3)) == 3:
            out["dependencies"] = draw(
                st.dictionaries(props, st.lists(props, unique=True))
            )
        elif draw(st.integers(0, 3)) == 3:
            out["dependencies"] = draw(st.dictionaries(props, json_schemata()))
    if patterns:
        out["patternProperties"] = patterns
    if additional is not None:
        out["additionalProperties"] = additional

    return out


schema_strategy_params = pytest.mark.parametrize(
    "schema_strategy",
    [
        pytest.param(gen_number("integer"), id="integer-schema"),
        pytest.param(gen_number("number"), id="number-schema"),
        pytest.param(gen_string(), id="string-schema"),
        pytest.param(gen_enum(), id="enum-schema"),
        pytest.param(gen_array(), id="array-schema"),
        pytest.param(gen_object(), id="object-schema"),
        pytest.param(json_schemata(), id="any-schema"),
    ],
)
