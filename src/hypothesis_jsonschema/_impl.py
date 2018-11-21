"""A Hypothesis extension for JSON schemata."""

from typing import Any, Dict, List, Union

import jsonschema
from hypothesis import assume
import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, dict]

JSON_STRATEGY: st.SearchStrategy[JSONType] = st.deferred(
    lambda: st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.lists(JSON_STRATEGY, max_size=3),
        st.dictionaries(st.text(), JSON_STRATEGY, max_size=3),
    )
)


def from_schema(schema: dict) -> st.SearchStrategy[JSONType]:
    """Take a JSON schema and return a strategy for allowed JSON objects."""
    # Boolean objects are special schemata; False rejects all and True accepts all.
    if schema is False:
        return st.nothing()
    if schema is True:
        return JSON_STRATEGY
    # Otherwise, we're dealing with "objects", i.e. dicts.
    if not isinstance(schema, dict):
        raise InvalidArgument(
            f"Got schema={schema} of type {type(schema).__name__}, "
            "but expected a dict."
        )
    jsonschema.validators.validator_for(schema).check_schema(schema)

    # Now we handle as many validation keywords as we can...
    if schema == {}:
        return JSON_STRATEGY

    if "enum" in schema:
        return st.sampled_from(schema["enum"])
    if "const" in schema:
        return st.just(schema["const"])
    if "type" in schema:
        if schema["type"] == "null":
            return st.none()
        if schema["type"] == "boolean":
            return st.booleans()
        if schema["type"] in ("number", "integer"):
            return numeric_schema(schema)
        if schema["type"] == "string":
            return string_schema(schema)
        if schema["type"] == "array":
            return array_schema(schema)
        assert schema["type"] == "object"
        return object_schema(schema)

    # Finally, we just filter arbitrary JSON and hope it passes.  (TODO: rip this out)

    def _filter(value: JSONType) -> bool:
        try:
            jsonschema.validate(value, schema=schema)
        except jsonschema.exceptions.ValidationError:  # pragma: no cover
            return False
        return True

    return JSON_STRATEGY.filter(_filter)


def numeric_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle numeric schemata."""
    assert not ("maximum" in schema and "exclusiveMaximum" in schema)
    assert not ("minimum" in schema and "exclusiveMinimum" in schema)
    multiple_of = schema.get("multipleOf")
    if multiple_of is not None or schema["type"] == "integer":
        lower = schema.get("exclusiveMinimum")
        if lower is not None:
            lower += 1
        else:
            lower = schema.get("minimum")
        upper = schema.get("exclusiveMaximum")
        if upper is not None:
            upper -= 1
        else:
            upper = schema.get("maximum")
        if multiple_of is not None:
            if lower is not None:
                lower += (multiple_of - lower) % multiple_of
                lower //= multiple_of
            if upper is not None:
                upper -= upper % multiple_of
                upper //= multiple_of
            return st.integers(lower, upper).map(
                lambda x: x * multiple_of  # type: ignore
            )
        return st.integers(lower, upper)
    lower = schema.get("exclusiveMinimum", schema.get("minimum"))
    upper = schema.get("exclusiveMaximum", schema.get("maximum"))
    return st.floats(lower, upper, allow_nan=False, allow_infinity=False).filter(
        lambda x: x not in (lower, upper)
    )


def string_schema(schema: dict) -> st.SearchStrategy[str]:
    """Handle a specific kind of schema."""


def array_schema(schema: dict) -> st.SearchStrategy[List[JSONType]]:
    """Handle a specific kind of schema."""


def object_schema(schema: dict) -> st.SearchStrategy[Dict[str, JSONType]]:
    """Handle a specific kind of schema."""


def json_schemata() -> st.SearchStrategy[Union[bool, Dict[str, JSONType]]]:
    """A Hypothesis strategy for arbitrary JSON schemata."""
    return _json_schemata()  # pylint: disable=no-value-for-parameter


@st.composite
def _json_schemata(draw: Any) -> Any:
    """Wrapped so we can disable the pylint error in one place only."""
    # Current version of jsonschema does not support boolean schemata,
    # but 3.0 will.  See https://github.com/Julian/jsonschema/issues/337
    kinds = [
        "null",
        "boolean",
        "integer",
        "number",
        "string",
        "const",
        "enum",
        "array",
        "object",
    ]
    kind = draw(st.sampled_from(kinds))
    hashable_json = st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
    )
    if kind == "const":
        return {"const": draw(hashable_json)}
    if kind == "enum":
        return {
            "enum": draw(st.lists(hashable_json, min_size=1, max_size=10, unique=True))
        }
    if kind in ("null", "boolean"):
        return {"type": kind}
    if kind in ("number", "integer"):
        lower = draw(st.none() | st.integers())
        upper = draw(st.none() | st.integers())
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        multiple_of = draw(st.none() | st.integers(2, 100))
        assume(
            None in (multiple_of, lower, upper) or multiple_of <= (upper - lower - 2)
        )
        out: Dict[str, JSONType] = {"type": kind}
        if lower is not None:
            out["minimum"] = lower
        if upper is not None:
            out["maximum"] = upper
        if multiple_of is not None:
            out["multipleOf"] = multiple_of
        return out

    return {}
