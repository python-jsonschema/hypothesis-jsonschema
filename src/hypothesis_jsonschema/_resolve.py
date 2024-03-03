"""
Canonicalisation logic for JSON schemas.

The canonical format that we transform to is not intended for human consumption.
Instead, it prioritises locality of reasoning - for example, we convert oneOf
arrays into an anyOf of allOf (each sub-schema being the original plus not anyOf
the rest).  Resolving references and merging subschemas is also really helpful.

All this effort is justified by the huge performance improvements that we get
when converting to Hypothesis strategies.  To the extent possible there is only
one way to generate any given value... but much more importantly, we can do
most things by construction instead of by filtering.  That's the difference
between "I'd like it to be faster" and "doesn't finish at all".
"""

from copy import deepcopy
from typing import NoReturn, Optional, Union

from hypothesis.errors import InvalidArgument
from jsonschema.validators import _RefResolver

from ._canonicalise import (
    SCHEMA_KEYS,
    SCHEMA_OBJECT_KEYS,
    HypothesisRefResolutionError,
    Schema,
    canonicalish,
    merged,
)


class LocalResolver(_RefResolver):
    def resolve_remote(self, uri: str) -> NoReturn:
        raise HypothesisRefResolutionError(
            f"hypothesis-jsonschema does not fetch remote references (uri={uri!r})"
        )


def resolve_all_refs(
    schema: Union[bool, Schema], *, resolver: Optional[LocalResolver] = None
) -> Schema:
    """
    Resolve all references in the given schema.

    This handles nested definitions, but not recursive definitions.
    The latter require special handling to convert to strategies and are much
    less common, so we just ignore them (and error out) for now.
    """
    if isinstance(schema, bool):
        return canonicalish(schema)
    assert isinstance(schema, dict), schema
    if resolver is None:
        resolver = LocalResolver.from_schema(deepcopy(schema))
    if not isinstance(resolver, _RefResolver):
        raise InvalidArgument(
            f"resolver={resolver} (type {type(resolver).__name__}) is not a RefResolver"
        )

    if "$ref" in schema:
        s = dict(schema)
        ref = s.pop("$ref")
        with resolver.resolving(ref) as got:
            m = merged([s, resolve_all_refs(got, resolver=resolver)])
            if m is None:  # pragma: no cover
                msg = f"$ref:{ref!r} had incompatible base schema {s!r}"
                raise HypothesisRefResolutionError(msg)
            assert "$ref" not in m
            return m
    assert "$ref" not in schema

    for key in SCHEMA_KEYS:
        val = schema.get(key, False)
        if isinstance(val, list):
            schema[key] = [
                resolve_all_refs(v, resolver=resolver) if isinstance(v, dict) else v
                for v in val
            ]
        elif isinstance(val, dict):
            schema[key] = resolve_all_refs(val, resolver=resolver)
        else:
            assert isinstance(val, bool)
    for key in SCHEMA_OBJECT_KEYS:  # values are keys-to-schema-dicts, not schemas
        if key in schema:
            subschema = schema[key]
            assert isinstance(subschema, dict)
            schema[key] = {
                k: resolve_all_refs(v, resolver=resolver) if isinstance(v, dict) else v
                for k, v in subschema.items()
            }
    assert isinstance(schema, dict)
    assert "$ref" not in schema
    return schema
