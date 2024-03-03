"""Canonical encoding for the JSONSchema semantics, where 1 == 1.0."""

import json
import math
import platform
from typing import Any, Dict, Tuple, Union

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
PYTHON_IMPLEMENTATION = platform.python_implementation()
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]

if PYTHON_IMPLEMENTATION != "PyPy":
    from json.encoder import _make_iterencode, encode_basestring_ascii  # type: ignore
else:  # pragma: no cover
    _make_iterencode = None
    encode_basestring_ascii = None


def _floatstr(o: float) -> str:
    # This is the bit we're overriding - integer-valued floats are
    # encoded as integers, to support JSONschemas's uniqueness.
    assert math.isfinite(o)
    if o == int(o):
        return repr(int(o))
    return repr(o)


class CanonicalisingJsonEncoder(json.JSONEncoder):
    if PYTHON_IMPLEMENTATION == "PyPy":  # pragma: no cover

        def _JSONEncoder__floatstr(self, o: float) -> str:
            return _floatstr(o)

    else:

        def iterencode(self, o: Any, _one_shot: bool = False) -> Any:  # noqa
            """Replace a stdlib method, so we encode integer-valued floats as ints."""
            return _make_iterencode(
                {},
                self.default,
                encode_basestring_ascii,
                self.indent,
                _floatstr,
                self.key_separator,
                self.item_separator,
                self.sort_keys,
                self.skipkeys,
                _one_shot,
            )(o, 0)


def encode_canonical_json(value: JSONType) -> str:
    """Canonical form serialiser, for uniqueness testing."""
    return json.dumps(value, sort_keys=True, cls=CanonicalisingJsonEncoder)


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
