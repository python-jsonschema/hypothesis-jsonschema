"""A Hypothesis extension for JSON schemata."""

__version__ = "0.1.0"
__all__ = ["from_schema", "json_schemata"]

from typing import Dict, Union, Callable

import jsonschema
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
        st.lists(JSON_STRATEGY, max_size=5),
        st.dictionaries(st.text(), JSON_STRATEGY, max_size=5),
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
            f"Got schema={schema} of type {type(schema)}, but expected a dict."
        )
    jsonschema.validators.validator_for(schema).check_schema(schema)

    def _filter(value: JSONType) -> bool:
        try:
            jsonschema.validate(value, schema=schema)
        except jsonschema.exceptions.ValidationError:  # pragma: no cover
            return False
        return True

    return JSON_STRATEGY.filter(_filter)


def json_schemata() -> st.SearchStrategy[Union[bool, Dict[str, JSONType]]]:
    """A Hypothesis strategy for arbitrary JSON schemata."""
    # Current version of jsonschema does not support boolean schemata,
    # but 3.0 will.  See https://github.com/Julian/jsonschema/issues/337
    return st.builds(dict)
