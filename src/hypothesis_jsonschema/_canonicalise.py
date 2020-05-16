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
from copy import deepcopy
from json.encoder import _make_iterencode, encode_basestring_ascii  # type: ignore
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import jsonschema
from hypothesis.errors import InvalidArgument
from hypothesis.internal.floats import next_down, next_up

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]
Schema = Dict[str, JSONType]

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


def make_validator(
    schema: Schema,
) -> Union[
    jsonschema.validators.Draft3Validator,
    jsonschema.validators.Draft4Validator,
    jsonschema.validators.Draft6Validator,
    jsonschema.validators.Draft7Validator,
]:
    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema)


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


class HypothesisRefResolutionError(jsonschema.exceptions.RefResolutionError):
    pass


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
        if not make_validator(schema).is_valid(schema["const"]):
            return FALSEY
        return {"const": schema["const"]}
    if "enum" in schema:
        validator = make_validator(schema)
        enum_ = sorted(
            (v for v in schema["enum"] if validator.is_valid(v)), key=sort_key
        )
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
        mul = schema.get("multipleOf")
        if (
            lo is not None
            and hi is not None
            and (
                lo > hi
                or (lo == hi and (exmin or exmax))
                or (mul and not has_divisibles(lo, hi, mul, exmin, exmax))
            )
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
            schema["minItems"] = max(schema.get("minItems", 1), 1)
    if "array" in type_ and schema.get("minItems", 0) > schema.get(
        "maxItems", math.inf
    ):
        type_.remove("array")
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
    if "array" in type_ and schema.get("items", TRUTHY) == TRUTHY:
        schema.pop("items", None)
    if (
        "properties" in schema
        and not schema.get("patternProperties")
        and schema.get("additionalProperties") == FALSEY
    ):
        schema["maxProperties"] = min(
            schema.get("maxProperties", math.inf), len(schema["properties"])
        )
    if "object" in type_ and schema.get("minProperties", 0) > schema.get(
        "maxProperties", math.inf
    ):
        type_.remove("object")
    # Canonicalise "required" schemas to remove redundancy
    if "object" in type_ and "required" in schema:
        assert isinstance(schema["required"], list)
        reqs = set(schema["required"])
        if schema.get("dependencies"):
            # When the presence of a required property requires other properties via
            # dependencies, those properties can be moved to the base required keys.
            dep_names = {
                k: sorted(v)
                for k, v in schema["dependencies"].items()
                if isinstance(v, list)
            }
            while reqs.intersection(dep_names):
                for r in reqs.intersection(dep_names):
                    reqs.update(dep_names.pop(r))
            for k, v in list(schema["dependencies"].items()):
                if isinstance(v, list) and k not in dep_names:
                    schema["dependencies"].pop(k)
        schema["required"] = sorted(reqs)
        max_ = schema.get("maxProperties", float("inf"))
        assert isinstance(max_, (int, float))
        propnames = schema.get("propertyNames", {})
        if len(schema["required"]) > max_:
            type_.remove("object")
        else:
            validator = make_validator(propnames)
            if not all(validator.is_valid(name) for name in schema["required"]):
                type_.remove("object")

    for t, kw in TYPE_SPECIFIC_KEYS:
        numeric = {"number", "integer"}
        if t in type_ or (t in numeric and numeric.intersection(type_)):
            continue
        for k in kw.split():
            schema.pop(k, None)
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
        type_keys = {k: set(v.split()) for k, v in TYPE_SPECIFIC_KEYS}
        type_constraints = {"type"}
        for v in type_keys.values():
            type_constraints |= v
        if set(not_).issubset(type_constraints):
            not_["type"] = get_type(not_)
            for t in set(type_).intersection(not_["type"]):
                # If some type is allowed and totally unconstrained byt the "not"
                # schema, it cannot be allowed
                if t == "integer" and "number" in type_:
                    continue
                if not type_keys.get(t, set()).intersection(not_):
                    type_.remove(t)
                    if t not in ("integer", "number"):
                        not_["type"].remove(t)
            not_ = canonicalish(not_)
        if not_ != FALSEY:
            # If the "not" key rejects nothing, discard it
            schema["not"] = not_
    assert isinstance(type_, list), type_
    if not type_:
        assert type_ == []
        return FALSEY
    if type_ == ["null"]:
        return {"const": None}
    if type_ == ["boolean"]:
        return {"enum": [False, True]}
    if type_ == ["null", "boolean"]:
        return {"enum": [None, False, True]}
    if len(type_) == 1:
        schema["type"] = type_[0]
    elif type_ == get_type({}):
        schema.pop("type", None)
    else:
        schema["type"] = type_
    # Canonicalise "xxxOf" lists; in each case canonicalising and sorting the
    # sub-schemas then handling any key-specific logic.
    if TRUTHY in schema.get("anyOf", ()):
        schema.pop("anyOf", None)
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
        one_of = schema.pop("oneOf")
        assert isinstance(one_of, list)
        one_of = sorted(one_of, key=encode_canonical_json)
        one_of = [s for s in one_of if s != FALSEY]
        if len(one_of) == 1:
            m = merged([schema, one_of[0]])
            if m is not None:  # pragma: no branch
                return m
        if (not one_of) or one_of.count(TRUTHY) > 1:
            return FALSEY
        schema["oneOf"] = one_of
    # if/then/else schemas are ignored unless if and another are present
    if "if" not in schema:
        schema.pop("then", None)
        schema.pop("else", None)
    if "then" not in schema and "else" not in schema:
        schema.pop("if", None)
    if schema.get("uniqueItems") is False:
        del schema["uniqueItems"]
    return schema


TRUTHY = canonicalish(True)
FALSEY = canonicalish(False)


class LocalResolver(jsonschema.RefResolver):
    def resolve_remote(self, uri: str) -> NoReturn:
        raise HypothesisRefResolutionError(
            f"hypothesis-jsonschema does not fetch remote references (uri={uri!r})"
        )


def resolve_all_refs(schema: Schema, *, resolver: LocalResolver = None) -> Schema:
    """
    Resolve all references in the given schema.

    This handles nested definitions, but not recursive definitions.
    The latter require special handling to convert to strategies and are much
    less common, so we just ignore them (and error out) for now.
    """
    if resolver is None:
        resolver = LocalResolver.from_schema(deepcopy(schema))
    if not isinstance(resolver, jsonschema.RefResolver):
        raise InvalidArgument(
            f"resolver={resolver} (type {type(resolver).__name__}) is not a RefResolver"
        )

    if "$ref" in schema:
        s = dict(schema)
        ref = s.pop("$ref")
        with resolver.resolving(ref) as got:
            if s == {}:
                return resolve_all_refs(got, resolver=resolver)
            m = merged([s, got])
            if m is None:
                msg = f"$ref:{ref!r} had incompatible base schema {s!r}"
                raise HypothesisRefResolutionError(msg)
            return resolve_all_refs(m, resolver=resolver)
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
    return schema


def merged(schemas: List[Any]) -> Optional[Schema]:
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
            if make_validator(out).is_valid(s["const"]):
                out = s
                continue
            return FALSEY
        if "enum" in s:
            validator = make_validator(out)
            enum_ = [v for v in s["enum"] if validator.is_valid(v)]
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
        for key in (
            {"maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties"}
            & set(s)
            & set(out)
        ):
            out[key] = min([out[key], s.pop(key)])
        for key in (
            {"minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties"}
            & set(s)
            & set(out)
        ):
            out[key] = max([out[key], s.pop(key)])
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


def has_divisibles(
    start: float, end: float, divisor: float, exmin: bool, exmax: bool
) -> bool:
    """If the given range from `start` to `end` has any numbers divisible by `divisor`."""
    divisible_num = end // divisor - start // divisor
    if not exmin and not start % divisor:
        divisible_num += 1
    if exmax and not end % divisor:
        divisible_num -= 1
    return divisible_num >= 1
