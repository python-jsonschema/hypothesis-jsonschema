"""Canonical encoding for the JSONSchema semantics, where 1 == 1.0."""
import functools
import json
import math
from json.encoder import _make_iterencode, encode_basestring_ascii  # type: ignore
from typing import Any, Callable, Dict, Tuple, Type, Union

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]


class CanonicalisingJsonEncoder(json.JSONEncoder):
    def iterencode(self, o: Any, _one_shot: bool = False) -> Any:
        """Replace a stdlib method, so we encode integer-valued floats as ints."""

        def floatstr(o: float) -> str:
            # This is the bit we're overriding - integer-valued floats are
            # encoded as integers, to support JSONschemas's uniqueness.
            assert math.isfinite(o)
            if o == int(o):
                return repr(int(o))
            return repr(o)

        return _make_iterencode(
            {},
            self.default,
            encode_basestring_ascii,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )(o, 0)


def _make_cache_key(
    value: JSONType,
) -> Tuple[Type, Union[None, bool, float, str, tuple, frozenset]]:
    """Make a hashable object from any JSON value.

    The idea is to recursively convert all mutable values to immutable and adding values types as a discriminant.
    """
    if isinstance(value, dict):
        return (dict, frozenset((k, _make_cache_key(v)) for k, v in value.items()))
    if isinstance(value, list):
        return (list, tuple(map(_make_cache_key, value)))
    # Primitive types are hashable
    # `type` is needed to distinguish false-ish values - 0, "", False have the same hash (0)
    return (type(value), value)


class HashedJSON:
    """A proxy that holds a JSON value.

    Adds a capability for the inner value to be cached, loosely based on `functools._HashedSeq`.
    """

    __slots__ = ("value", "hashedvalue")

    def __init__(self, value: JSONType):
        self.value = value
        # `hash` is called multiple times on cache miss, therefore it is evaluated only once
        self.hashedvalue = hash(_make_cache_key(value))

    def __hash__(self) -> int:
        return self.hashedvalue

    def __eq__(self, other: "HashedJSON") -> bool:  # type: ignore
        # TYPES: This class should be used only for caching purposes and there should be
        # no values of other types to compare
        return self.hashedvalue == other.hashedvalue


def cached_json(func: Callable[[HashedJSON], str]) -> Callable[[JSONType], str]:
    """Cache calls to `encode_canonical_json`.

    The same schemas are encoded multiple times during canonicalisation and caching gives visible performance impact.
    """
    cached_func = functools.lru_cache(maxsize=1024)(func)

    @functools.wraps(cached_func)
    def wrapped(value: JSONType) -> str:
        return cached_func(HashedJSON(value))

    return wrapped


@cached_json
def encode_canonical_json(value: HashedJSON) -> str:
    """Canonical form serialiser, for uniqueness testing."""
    return json.dumps(value.value, sort_keys=True, cls=CanonicalisingJsonEncoder)


def sort_key(value: JSONType) -> Tuple[int, float, Union[float, str]]:
    """Return a sort key (type, guess, tiebreak) that can compare any JSON value.

    Sorts scalar types before collections, and within each type tries for a
    sensible ordering similar to Hypothesis' idea of simplicity.
    """
    if value is None:
        return (0, 0, 0)
    if isinstance(value, bool):
        return (1, int(value), 0)
    if isinstance(value, (int, float)):
        return (2 if int(value) == value else 3, abs(value), value >= 0)
    type_key = {str: 4, list: 5, dict: 6}[type(value)]
    return (type_key, len(value), encode_canonical_json(value))
