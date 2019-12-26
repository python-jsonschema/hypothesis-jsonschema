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

import json
import math
from json.encoder import _make_iterencode, encode_basestring_ascii  # type: ignore
from typing import Any, Dict, List, Optional, Tuple, Union

import hypothesis.strategies as st
import jsonschema
from hypothesis.errors import InvalidArgument
from hypothesis.internal.floats import next_down, next_up

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]
Schema = Dict[str, JSONType]

JSON_STRATEGY: st.SearchStrategy[JSONType] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False).map(lambda x: x or 0.0)
    | st.text(),
    lambda strategy: st.lists(strategy, max_size=3)
    | st.dictionaries(st.text(), strategy, max_size=3),
)

# Canonical type strings, in order.
TYPE_STRINGS = ("null", "boolean", "integer", "number", "string", "array", "object")
TYPE_SPECIFIC_KEYS = (
    ("number", "multipleOf maximum exclusiveMaximum minimum exclusiveMinimum"),
    ("integer", "multipleOf maximum exclusiveMaximum minimum exclusiveMinimum"),
    ("string", "maxLength minLength pattern contentEncoding contentMediaType"),
    ("array", "items additionalItems maxItems minItems uniqueItems contains"),
    (
        "object",
        "maxProperties minProperties required properties patternProperties "
        "additionalProperties dependencies propertyNames",
    ),
)
# Names of keywords where the associated values may be schemas or lists of schemas.
SCHEMA_KEYS = tuple(
    "items additionalItems contains additionalProperties propertyNames "
    "if then else allOf anyOf oneOf not".split()
)
# Names of keywords where the value is an object whose values are schemas.
# Note that in some cases ("dependencies"), the value may be a list of strings.
SCHEMA_OBJECT_KEYS = ("properties", "patternProperties", "dependencies")


def is_valid(instance: JSONType, schema: Schema) -> bool:
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


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


def encode_canonical_json(value: JSONType) -> str:
    """Canonical form serialiser, for uniqueness testing."""
    return json.dumps(value, sort_keys=True, cls=CanonicalisingJsonEncoder)


def get_type(schema: Schema) -> List[str]:
    """Return a canonical value for the "type" key.

    Note that this will return [], the empty list, if the value is a list without
    any allowed type names; *even though* this is explicitly an invalid value.
    """
    type_ = schema.get("type", list(TYPE_STRINGS))
    # Canonicalise the "type" key to a sorted list of type strings.
    if isinstance(type_, str):
        assert type_ in TYPE_STRINGS
        return [type_]
    assert isinstance(type_, list) and set(type_).issubset(TYPE_STRINGS), type_
    return [t for t in TYPE_STRINGS if t in type_]


def upper_bound_instances(schema: Schema) -> float:
    """Return an upper bound on the number of instances that match this schema."""
    if schema == FALSEY:
        return 0
    if "const" in schema:
        return 1
    if "enum" in schema:
        assert isinstance(schema["enum"], list)
        return len(schema["enum"])
    # TODO: could handle lots more cases here...
    # Converting known cases to enums would also be a good approach.
    return math.inf


def get_number_bounds(
    schema: Schema, *, _for_integer: bool = False,
) -> Tuple[Optional[float], Optional[float], bool, bool]:
    """Get the min and max allowed floats, and whether they are exclusive."""
    assert "number" in get_type(schema) or _for_integer

    lower = schema.get("minimum")
    upper = schema.get("maximum")
    exmin = schema.get("exclusiveMinimum", False)
    exmax = schema.get("exclusiveMaximum", False)
    assert lower is None or isinstance(lower, (int, float))
    assert upper is None or isinstance(upper, (int, float))
    assert isinstance(exmin, (bool, int, float))
    assert isinstance(exmax, (bool, int, float))

    # Canonicalise to number-and-boolean representation
    if exmin is not True and exmin is not False:
        if lower is None or exmin >= lower:
            lower, exmin = exmin, True
        else:
            exmin = False
    if exmax is not True and exmax is not False:
        if upper is None or exmax <= upper:
            upper, exmax = exmax, True
        else:
            exmax = False
    assert isinstance(exmin, bool)
    assert isinstance(exmax, bool)

    # Adjust bounds and cast to float
    if lower is not None and not _for_integer:
        lo = float(lower)
        if lo < lower:
            lo = next_up(lo)
            exmin = False
        lower = lo
    if upper is not None and not _for_integer:
        hi = float(upper)
        if hi > upper:
            hi = next_down(hi)
            exmax = False
        upper = hi

    return lower, upper, exmin, exmax


def get_integer_bounds(schema: Schema) -> Tuple[Optional[int], Optional[int]]:
    """Get the min and max allowed integers."""
    assert "integer" in get_type(schema)
    lower, upper, exmin, exmax = get_number_bounds(schema, _for_integer=True)
    # Adjust bounds and cast to int
    if lower is not None:
        lo = math.ceil(lower)
        if exmin and lo == lower:
            lo += 1
        lower = lo
    if upper is not None:
        hi = math.floor(upper)
        if exmax and hi == upper:
            hi -= 1
        upper = hi
    return lower, upper


def canonicalish(schema: JSONType) -> Dict[str, Any]:
    """Convert a schema into a more-canonical form.

    This is obviously incomplete, but improves best-effort recognition of
    equivalent schemas and makes conversion logic simpler.
    """
    if schema is True:
        return {}
    elif schema is False:
        return {"not": {}}
    # Otherwise, we're dealing with "objects", i.e. dicts.
    if not isinstance(schema, dict):
        raise InvalidArgument(
            f"Got schema={schema} of type {type(schema).__name__}, "
            "but expected a dict."
        )
    # Make a copy, so we don't mutate the existing schema in place.
    schema = dict(schema)
    if "const" in schema:
        if not is_valid(schema["const"], schema):
            return FALSEY
        return {"const": schema["const"]}
    if "enum" in schema:
        enum_ = [v for v in schema["enum"] if is_valid(v, schema)]
        if not enum_:
            return FALSEY
        elif len(enum_) == 1:
            return {"const": enum_[0]}
        return {"enum": enum_}
    # Recurse into the value of each keyword with a schema (or list of them) as a value
    for key in SCHEMA_KEYS:
        if isinstance(schema.get(key), list):
            schema[key] = [canonicalish(v) for v in schema[key]]
        elif isinstance(schema.get(key), (bool, dict)):
            schema[key] = canonicalish(schema[key])
        else:
            assert key not in schema
    for key in SCHEMA_OBJECT_KEYS:
        if key in schema:
            schema[key] = {
                k: v if isinstance(v, list) else canonicalish(v)
                for k, v in schema[key].items()
            }

    type_ = get_type(schema)
    if "number" in type_:
        lo, hi, exmin, exmax = get_number_bounds(schema)
        if (
            lo is not None
            and hi is not None
            and (lo > hi or (lo == hi and (exmin or exmax)))
        ):
            type_.remove("number")
    if "integer" in type_:
        lo, hi = get_integer_bounds(schema)
        mul = schema.get("multipleOf")
        if lo is not None and isinstance(mul, int) and mul > 1 and (lo % mul):
            lo += mul - (lo % mul)
        if hi is not None and isinstance(mul, int) and mul > 1 and (hi % mul):
            hi -= hi % mul

        if "number" not in type_:
            if lo is not None:
                schema["minimum"] = lo
                schema.pop("exclusiveMinimum", None)
            if hi is not None:
                schema["maximum"] = hi
                schema.pop("exclusiveMaximum", None)

        if lo is not None and hi is not None and lo > hi:
            type_.remove("integer")

    if "array" in type_ and "contains" in schema:
        if schema["contains"] == FALSEY:
            type_.remove("array")
        else:
            schema["minItems"] = max(schema.get("minItems", 0), 1)
        if schema["contains"] == TRUTHY:
            schema.pop("contains")
    if (
        "array" in type_
        and "minItems" in schema
        # TODO: could add logic for unsatisfiable list-of-items case
        and isinstance(schema.get("items", []), (bool, dict))
    ):
        count = upper_bound_instances(schema["items"])
        if (count == 0 and schema["minItems"] > 0) or (
            schema.get("uniqueItems", False) and count < schema["minItems"]
        ):
            type_.remove("array")
    if "array" in type_ and isinstance(schema.get("items"), list):
        schema["items"] = schema["items"][: schema.get("maxItems")]
        for idx, s in enumerate(schema["items"]):
            if s == FALSEY:
                if schema.get("minItems", 0) > idx:
                    type_.remove("array")
                    break
                schema["items"] = schema["items"][:idx]
                schema["maxItems"] = idx
                schema.pop("additionalItems", None)
                break
    if (
        "array" in type_
        and isinstance(schema.get("items"), list)
        and schema.get("additionalItems") == FALSEY
    ):
        schema.pop("maxItems", None)
    if "array" in type_ and (
        schema.get("items") == FALSEY or schema.get("maxItems", 1) == 0
    ):
        schema["maxItems"] = 0
        schema.pop("items", None)
        schema.pop("uniqueItems", None)
        schema.pop("additionalItems", None)
    # Canonicalise "required" schemas to remove redundancy
    if "required" in schema:
        assert isinstance(schema["required"], list)
        schema["required"] = sorted(set(schema["required"]))
        max_ = schema.get("maxProperties", float("inf"))
        assert isinstance(max_, (int, float))
        if len(schema["required"]) > max_:
            type_.remove("object")
    if not type_:
        assert type_ == []
        return FALSEY
    if type_ == ["null"]:
        return {"const": None}
    if type_ == ["boolean"]:
        return {"enum": [False, True]}
    if type_ == ["null", "boolean"]:
        return {"enum": [None, False, True]}
    schema["type"] = type_

    for t, kw in TYPE_SPECIFIC_KEYS:
        numeric = ["number", "integer"]
        if t in type_ or t in numeric and t in type_ + numeric:
            continue
        for k in kw.split():
            schema.pop(k, None)
    assert isinstance(type_, list), type_
    if len(type_) == 1:
        schema["type"] = type_[0]
    elif type_ == get_type({}):
        schema.pop("type")
    else:
        schema["type"] = type_
    # Remove no-op requires
    if "required" in schema and not schema["required"]:
        schema.pop("required")
    # Canonicalise "not" subschemas
    if "not" in schema:
        not_ = schema.pop("not")
        if not_ == TRUTHY or not_ == schema:
            # If everything is rejected, discard all other (irrelevant) keys
            # TODO: more sensitive detection of cases where the not-clause
            # excludes everything in the schema.
            return FALSEY
        if not_ != FALSEY:
            # If the "not" key rejects nothing, discard it
            schema["not"] = not_
    # Canonicalise "xxxOf" lists; in each case canonicalising and sorting the
    # sub-schemas then handling any key-specific logic.
    if "anyOf" in schema:
        schema["anyOf"] = sorted(schema["anyOf"], key=encode_canonical_json)
        schema["anyOf"] = [s for s in schema["anyOf"] if s != FALSEY]
        if not schema["anyOf"]:
            return FALSEY
        if len(schema) == len(schema["anyOf"]) == 1:
            return schema["anyOf"][0]  # type: ignore
    if "allOf" in schema:
        schema["allOf"] = sorted(schema["allOf"], key=encode_canonical_json)
        if any(s == FALSEY for s in schema["allOf"]):
            return FALSEY
        if all(s == TRUTHY for s in schema["allOf"]):
            schema.pop("allOf")
        elif len(schema) == len(schema["allOf"]) == 1:
            return schema["allOf"][0]  # type: ignore
        else:
            tmp = schema.copy()
            ao = tmp.pop("allOf")
            out = merged([tmp] + ao)
            if isinstance(out, dict):  # pragma: no branch
                schema = out
                # TODO: this assertion is soley because mypy 0.750 doesn't know
                # that `schema` is a dict otherwise. Needs minimal report upstream.
                assert isinstance(schema, dict)
    if "oneOf" in schema:
        oneOf = schema.pop("oneOf")
        assert isinstance(oneOf, list)
        oneOf = sorted(oneOf, key=encode_canonical_json)
        oneOf = [s for s in oneOf if s != FALSEY]
        if len(oneOf) == 1:
            m = merged([schema, oneOf[0]])
            if m is not None:  # pragma: no branch
                return m
        if (
            (not oneOf)
            or oneOf.count(TRUTHY) > 1
            or len(oneOf) > len({encode_canonical_json(s) for s in oneOf})
        ):
            return FALSEY
        schema["oneOf"] = oneOf
    # if/then/else schemas are ignored unless if and another are present
    if "if" not in schema:
        schema.pop("then", None)
        schema.pop("else", None)
    if "then" not in schema and "else" not in schema:
        schema.pop("if", None)
    return schema


TRUTHY = canonicalish(True)
FALSEY = canonicalish(False)


def merged(schemas: List[Any]) -> Union[None, Schema]:
    """Merge *n* schemas into a single schema, or None if result is invalid.

    Takes the logical intersection, so any object that validates against the returned
    schema must also validate against all of the input schemas.

    None is returned for keys that cannot be merged short of pushing parts of
    the schema into an allOf construct, such as the "contains" key for arrays -
    there is no other way to merge two schema that could otherwise be applied to
    different array elements.
    It's currently also used for keys that could be merged but aren't yet.
    """
    assert schemas, "internal error: must pass at least one schema to merge"
    out = canonicalish(schemas[0])
    for s in schemas[1:]:
        s = canonicalish(s)
        # If we have a const or enum, this is fairly easy by filtering:
        if "const" in s:
            if is_valid(s["const"], out):
                out = s
                continue
            return FALSEY
        if "enum" in s:
            enum_ = [v for v in s["enum"] if is_valid(v, out)]
            if not enum_:
                return FALSEY
            elif len(enum_) == 1:
                out = {"const": enum_[0]}
            else:
                out = {"enum": enum_}
            continue

        if "type" in out and "type" in s:
            tt = s.pop("type")
            out["type"] = [t for t in get_type(out) if t in tt]
            if not get_type(out):
                return FALSEY
            for t, kw in TYPE_SPECIFIC_KEYS:
                numeric = ["number", "integer"]
                if t in get_type(out) or t in numeric and t in get_type(out) + numeric:
                    continue
                for k in kw.split():
                    s.pop(k, None)
                    out.pop(k, None)
        # TODO: keeping track of which elements are affected by which schemata
        # while merging properties, patternProperties, and additionalProperties
        # is a nightmare, so I'm just not going to try for now.  e.g.:
        #    {"patternProperties": {".": {"type": "null"}}}
        #    {"type": "object", "additionalProperties": {"type": "boolean"}}
        # The the former requries null values for all non-"" keys, while the
        # latter requires all-bool values.  Merging them should output
        #    {"enum": [{}, {"": None}]}
        # but dealing with this in the general case is a nightmare.

        def diff_in_out(k: str) -> bool:
            sval = s.get(k, object())
            return k in s and sval != out.get(k, sval)

        def diff_keys(k: str) -> bool:
            return set(out.get(k, [])) != set(s.get(k, []))

        if diff_in_out("patternProperties"):
            # Do I want to compute regex intersections with optional anchors? No.
            return None

        if "additionalProperties" in out and (
            diff_keys("properties") or diff_keys("patternProperties")
        ):
            # If the known names or regex patterns vary at all, we'd have to merge the
            # additionalProperties schema into some and it's just not worth it.
            return None
        if diff_in_out("additionalProperties"):
            m = merged([out["additionalProperties"], s["additionalProperties"]])
            if m is None:
                return None
            out["additionalProperties"] = m

        if diff_in_out("properties"):
            # TODO: this doesn't account for cases where patternProperties in one
            # overlap with properties in the other.  It would be nice to try merging
            # them in that case, or correctly bail out if we can't merge them.
            op = out["properties"]
            sp = s.pop("properties")
            for k, v in sp.items():
                if v != op.get(k, v):
                    v = merged([op[k], v])
                    if v is None:
                        return None
                op[k] = v
        if "required" in out and "required" in s:
            out["required"] = sorted(set(out["required"] + s.pop("required")))
        for key in {"maxLength", "maxItems", "maxProperties"} & set(s) & set(out):
            out[key] = min([out[key], s[key]])
        for key in {"minLength", "minItems", "minProperties"} & set(s) & set(out):
            out[key] = max([out[key], s[key]])
        # TODO: Handle remaining mergable keys.

        for k, v in s.items():
            if k not in out:
                out[k] = v
            elif out[k] != v:
                return None
        out = canonicalish(out)
        if out == FALSEY:
            return FALSEY
    jsonschema.validators.validator_for(out).check_schema(out)
    return out
