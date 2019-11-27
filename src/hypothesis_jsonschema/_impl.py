"""A Hypothesis extension for JSON schemata."""

import itertools
import json
import math
import operator
import re
from functools import partial
from typing import Any, Callable, Dict, List, Set, Union

import hypothesis.internal.conjecture.utils as cu
import hypothesis.provisional as prov
import hypothesis.strategies as st
import jsonschema
from hypothesis import assume
from hypothesis.errors import InvalidArgument
from hypothesis.internal.floats import next_down, next_up

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]
# TODO: grep for uses of JSONType which are actually Schema (or bool-or-Schema)
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


def encode_canonical_json(value: JSONType) -> str:
    """Canonical form serialiser, for uniqueness testing."""
    return json.dumps(value, sort_keys=True)


def get_type(schema: Schema) -> List[str]:
    """Return a canonical value for the "type" key.

    If the "type" key is not present, infer a plausible value from other keys.
    If we can't guess based on them, return None.

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
    schema = canonicalish(schema)
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


def canonicalish(schema: JSONType) -> Dict:
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
        if isinstance(schema.get(key), dict):
            schema[key] = canonicalish(schema[key])
    for key in SCHEMA_OBJECT_KEYS:
        if key in schema:
            schema[key] = {
                k: canonicalish(v) if isinstance(v, dict) else v
                for k, v in schema[key].items()
            }
    # Canonicalise the "type" if specified, but avoid changing semantics by
    # adding a type key (which would affect intersection/union logic).
    if "type" in schema:
        type_ = get_type(schema)
        if "array" in type_ and "contains" in schema:
            if canonicalish(schema["contains"]) == FALSEY:
                type_.remove("array")
            else:
                schema["minItems"] = max(schema.get("minItems", 0), 1)
            if canonicalish(schema["contains"]) == TRUTHY:
                schema.pop("contains")
        if (
            "array" in type_
            and "minItems" in schema
            and isinstance(schema.get("items", []), (bool, dict))
        ):
            count = upper_bound_instances(canonicalish(schema["items"]))
            if (count == 0 and schema["minItems"] > 0) or (
                schema.get("uniqueItems", False) and count < schema["minItems"]
            ):
                type_.remove("array")
        if "array" in type_ and isinstance(schema.get("items"), list):
            schema["items"] = schema["items"][: schema.get("maxItems")]
            for idx, s in enumerate(schema["items"]):
                if canonicalish(s) == FALSEY:
                    if schema.get("minItems", 0) > idx:
                        type_.remove("array")
                        break
                    schema["items"] = schema["items"][:idx]
                    schema["maxItems"] = idx
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
        not_ = canonicalish(schema.pop("not"))
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
        schema["anyOf"] = sorted(
            (canonicalish(s) for s in schema["anyOf"]), key=encode_canonical_json
        )
        schema["anyOf"] = [s for s in schema["anyOf"] if s != FALSEY]
        if not schema["anyOf"]:
            return FALSEY
        if len(schema) == len(schema["anyOf"]) == 1:
            return canonicalish(schema["anyOf"][0])
    if "allOf" in schema:
        schema["allOf"] = sorted(
            (canonicalish(s) for s in schema["allOf"]), key=encode_canonical_json
        )
        if any(s == FALSEY for s in schema["allOf"]):
            return FALSEY
        if all(s == TRUTHY for s in schema["allOf"]):
            schema.pop("allOf")
        elif len(schema) == len(schema["allOf"]) == 1:
            return canonicalish(schema["allOf"][0])
        else:
            tmp = schema.copy()
            ao = tmp.pop("allOf")
            out = merged([tmp] + ao)
            if isinstance(out, dict):  # pragma: no branch
                schema = out
                # TODO: this assertion is soley because mypy 0.720 doesn't know
                # that `schema` is a dict otherwise. Needs minimal report upstream.
                assert isinstance(schema, dict)
    if "oneOf" in schema:
        oneOf = schema.pop("oneOf")
        assert isinstance(oneOf, list)
        oneOf = sorted(map(canonicalish, oneOf), key=encode_canonical_json)
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
                    if v is None:  # pragma: no cover
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


def merged_as_strategies(schemas: List[Schema]) -> st.SearchStrategy[JSONType]:
    assert schemas, "internal error: must pass at least one schema to merge"
    if len(schemas) == 1:
        return from_schema(schemas[0])
    # Try to merge combinations of strategies.
    strats = []
    combined: Set[str] = set()
    inputs = {encode_canonical_json(s): s for s in schemas}
    for group in itertools.chain.from_iterable(
        itertools.combinations(inputs, n) for n in range(len(inputs), 0, -1)
    ):
        if combined.issuperset(group):
            continue
        s = merged([inputs[g] for g in group])
        if s is not None and s != FALSEY:  # pragma: no branch
            strats.append(
                from_schema(s).filter(lambda v: all(is_valid(v, s) for s in schemas))
            )
            combined.update(group)
    return st.one_of(strats)


def from_schema(schema: dict) -> st.SearchStrategy[JSONType]:
    """Take a JSON schema and return a strategy for allowed JSON objects.

    Schema reuse with "definitions" and "$ref" is not yet supported, but
    everything else in drafts 04, 05, and 07 is fully tested and working.
    """
    schema = canonicalish(schema)
    # Boolean objects are special schemata; False rejects all and True accepts all.
    if schema == FALSEY:
        return st.nothing()
    if schema == TRUTHY:
        return JSON_STRATEGY
    # Only check if declared, lest we error on inner non-latest-draft schemata.
    if "$schema" in schema:
        jsonschema.validators.validator_for(schema).check_schema(schema)

    # Now we handle as many validation keywords as we can...
    # Applying subschemata with boolean logic
    if "not" in schema:
        return JSON_STRATEGY.filter(partial(is_valid, schema=schema))
    if "anyOf" in schema:
        tmp = schema.copy()
        ao = tmp.pop("anyOf")
        return st.one_of([merged_as_strategies([tmp, s]) for s in ao])
    if "allOf" in schema:  # pragma: no cover
        # This is no-cover because `canonicalish` merges these into the base
        # schema in every known or generated test case, but we keep this
        # fallback logic to use partial merging for any remaining cases.
        tmp = schema.copy()
        ao = tmp.pop("allOf")
        return merged_as_strategies([tmp] + ao)
    if "oneOf" in schema:
        tmp = schema.copy()
        oo = tmp.pop("oneOf")
        schemas = [merged([tmp, s]) for s in oo]
        return st.one_of([from_schema(s) for s in schemas if s is not None]).filter(
            partial(is_valid, schema=schema)
        )
    # Conditional application of subschemata
    if "if" in schema:
        tmp = schema.copy()
        if_ = tmp.pop("if")
        then = tmp.pop("then", {})
        else_ = tmp.pop("else", {})
        return st.one_of([from_schema(s) for s in (then, else_, if_, tmp)]).filter(
            partial(is_valid, schema=schema)
        )
    # Simple special cases
    if "enum" in schema:
        assert schema["enum"], "Canonicalises to non-empty list or FALSEY"
        return st.sampled_from(schema["enum"])
    if "const" in schema:
        return st.just(schema["const"])
    # Finally, resolve schema by type - defaulting to "object"
    map_: Dict[str, Callable[[Schema], st.SearchStrategy[JSONType]]] = {
        "null": lambda _: st.none(),
        "boolean": lambda _: st.booleans(),
        "number": numeric_schema,
        "integer": numeric_schema,
        "string": string_schema,
        "array": array_schema,
        "object": object_schema,
    }
    assert set(map_) == set(TYPE_STRINGS)
    return st.one_of([map_[t](schema) for t in get_type(schema)])


def numeric_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle numeric schemata."""
    lower = schema.get("minimum")
    upper = schema.get("maximum")
    type_ = get_type(schema) or ["integer", "number"]

    exmin = schema.get("exclusiveMinimum")
    if exmin is True and "integer" in type_:
        assert lower is not None, "boolean exclusiveMinimum implies numeric minimum"
        lower += 1
        exmin = False
    elif exmin is not False and exmin is not None:
        lo = exmin + 1 if int(exmin) == exmin else math.ceil(exmin)
        if lower is None:
            lower = lo if "integer" in type_ else exmin
        else:
            lower = max(lower, lo if "integer" in type_ else exmin)
        exmin = False

    exmax = schema.get("exclusiveMaximum")
    if exmax is True and "integer" in type_:
        assert upper is not None, "boolean exclusiveMaximum implies numeric maximum"
        upper -= 1
        exmax = False
    elif exmax is not False and exmax is not None:
        hi = exmax - 1 if int(exmax) == exmax else math.floor(exmax)
        if upper is None:
            upper = hi if "integer" in type_ else exmax
        else:
            upper = min(upper, hi if "integer" in type_ else exmax)
        exmax = False

    if "multipleOf" in schema:
        multiple_of = schema["multipleOf"]
        assert isinstance(multiple_of, (int, float))
        if lower is not None:
            lo = math.ceil(lower / multiple_of)
            assert lo * multiple_of >= lower, (lower, lo)
            lower = lo
        if upper is not None:
            hi = math.floor(upper / multiple_of)
            assert hi * multiple_of <= upper, (upper, hi)
            upper = hi
        strat = st.integers(lower, upper).map(partial(operator.mul, multiple_of))
        # check for and filter out float bounds, inexact multiplication, etc.
        return strat.filter(partial(is_valid, schema=schema))

    strat = st.nothing()
    if "integer" in type_:
        lo = lower if lower is None else math.ceil(lower)
        hi = upper if upper is None else math.floor(upper)
        if lo is None or hi is None or lo <= hi:
            strat = st.integers(lo, hi)
    if "number" in type_:
        # Filter out negative-zero as it does not exist in JSON
        lo = exmin if lower is None else lower
        if lo is not None:
            lower = float(lo)
            if lower < lo:
                lower = next_up(lower)  # scary floats magic
            assert lower >= lo
        hi = exmax if upper is None else upper
        if hi is not None:
            upper = float(hi)
            if upper > hi:
                upper = next_down(upper)  # scary floats magic
            assert upper <= hi
        strat |= st.floats(
            min_value=lower,
            max_value=upper,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=exmin is not None,
            exclude_max=exmax is not None,
        ).filter(lambda n: n != 0 or math.copysign(1, n) == 1)
    return strat


RFC3339_FORMATS = (
    "date-fullyear",
    "date-month",
    "date-mday",
    "time-hour",
    "time-minute",
    "time-second",
    "time-secfrac",
    "time-numoffset",
    "time-offset",
    "partial-time",
    "full-date",
    "full-time",
    "date-time",
)
JSON_SCHEMA_STRING_FORMATS = RFC3339_FORMATS + (
    "email",
    "idn-email",
    "hostname",
    "idn-hostname",
    "ipv4",
    "ipv6",
    "uri",
    "uri-reference",
    "iri",
    "iri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
    "regex",
)


def rfc3339(name: str) -> st.SearchStrategy[str]:
    """Given the name of an RFC3339 date or time format,
    return a strategy for conforming values.

    See https://tools.ietf.org/html/rfc3339#section-5.6
    """
    # Hmm, https://github.com/HypothesisWorks/hypothesis/issues/170
    # would make this a lot easier...
    assert name in RFC3339_FORMATS
    simple = {
        "date-fullyear": st.integers(0, 9999).map(str),
        "date-month": st.integers(1, 12).map(str),
        "date-mday": st.integers(1, 28).map(str),  # incomplete but valid
        "time-hour": st.integers(0, 23).map(str),
        "time-minute": st.integers(0, 59).map(str),
        "time-second": st.integers(0, 59).map(str),  # ignore negative leap seconds
        "time-secfrac": st.from_regex(r"\.[0-9]+"),
    }
    if name in simple:
        return simple[name]
    if name == "time-numoffset":
        return st.tuples(
            st.sampled_from(["+", "-"]), rfc3339("time-hour"), rfc3339("time-minute")
        ).map(":".join)
    if name == "time-offset":
        return st.one_of(st.just("Z"), rfc3339("time-numoffset"))
    if name == "partial-time":
        return st.times().map(str)
    if name == "full-date":
        return st.dates().map(str)
    if name == "full-time":
        return st.tuples(rfc3339("partial-time"), rfc3339("time-offset")).map("".join)
    assert name == "date-time"
    return st.tuples(rfc3339("full-date"), rfc3339("full-time")).map("T".join)


@st.composite  # type: ignore
def regex_patterns(draw: Any) -> str:
    """A strategy for simple regular expression patterns."""
    fragments = st.one_of(
        st.just("."),
        st.from_regex(r"\[\^?[A-Za-z0-9]+\]"),
        REGEX_PATTERNS.map("{}+".format),
        REGEX_PATTERNS.map("{}?".format),
        REGEX_PATTERNS.map("{}*".format),
    )
    result = draw(st.lists(fragments, min_size=1, max_size=3).map("".join))
    assert isinstance(result, str)
    try:
        re.compile(result)
    except re.error:
        assume(False)
    return result


REGEX_PATTERNS = regex_patterns()

STRING_FORMATS = {
    # A value of None indicates a known but unsupported format.
    **{name: rfc3339(name) for name in RFC3339_FORMATS},
    "date": rfc3339("full-date"),
    "time": rfc3339("full-time"),
    # Hypothesis' provisional strategies are not type-annotated.
    # We should get a principled plan for them at some point I guess...
    "email": st.emails(),
    "idn-email": st.emails(),
    "hostname": prov.domains(),
    "idn-hostname": prov.domains(),
    "ipv4": prov.ip4_addr_strings(),
    "ipv6": prov.ip6_addr_strings(),
    **{
        name: prov.domains().map("https://{}".format)
        for name in ["uri", "uri-reference", "iri", "iri-reference", "uri-template"]
    },
    "json-pointer": st.just(""),
    "relative-json-pointer": st.just(""),
    "regex": REGEX_PATTERNS,
}


def string_schema(schema: dict) -> st.SearchStrategy[str]:
    """Handle schemata for strings."""
    # also https://json-schema.org/latest/json-schema-validation.html#rfc.section.7
    min_size = schema.get("minLength", 0)
    max_size = schema.get("maxLength", float("inf"))
    strategy = st.text(min_size=min_size, max_size=schema.get("maxLength"))
    if schema.get("format") in STRING_FORMATS:
        # Unknown "format" specifiers should be ignored for validation.
        # See https://json-schema.org/latest/json-schema-validation.html#format
        strategy = STRING_FORMATS[schema["format"]]
        if "pattern" in schema:  # pragma: no cover
            # This isn't really supported, but we'll do our best.
            strategy = strategy.filter(
                lambda s: re.search(schema["pattern"], string=s) is not None
            )
    elif "pattern" in schema:
        try:
            re.compile(schema["pattern"])
            strategy = st.from_regex(schema["pattern"])
        except re.error:
            # Patterns that are invalid in Python, or just malformed
            return st.nothing()
    # TODO: mypy should be able to tell that the lambda is returning a bool
    # without the explicit cast, but can't as of v 0.720 - report upstream.
    return strategy.filter(lambda s: bool(min_size <= len(s) <= max_size))


def array_schema(schema: dict) -> st.SearchStrategy[List[JSONType]]:
    """Handle schemata for arrays."""
    items = schema.get("items", {})
    additional_items = schema.get("additionalItems", {})
    min_size = schema.get("minItems", 0)
    max_size = schema.get("maxItems")
    unique = schema.get("uniqueItems")
    if isinstance(items, list):
        min_size = max(0, min_size - len(items))
        if max_size is not None:
            max_size -= len(items)
        if unique:

            @st.composite  # type: ignore
            def compose_lists_with_filter(draw: Any) -> List[JSONType]:
                elems = []
                seen: Set[str] = set()

                def not_seen(elem: JSONType) -> bool:
                    return encode_canonical_json(elem) not in seen

                for s in items:
                    elems.append(draw(from_schema(s).filter(not_seen)))
                    seen.add(encode_canonical_json(elems[-1]))
                extra_items = st.lists(
                    from_schema(additional_items).filter(not_seen),
                    min_size=min_size,
                    max_size=max_size,
                    unique_by=encode_canonical_json,
                )
                more_elems: List[JSONType] = draw(extra_items)
                return elems + more_elems

            strat = compose_lists_with_filter()
        else:
            fixed_items = st.tuples(*map(from_schema, items)).map(list)
            extra_items = st.lists(
                from_schema(additional_items), min_size=min_size, max_size=max_size
            )
            strat = st.builds(operator.add, fixed_items, extra_items)
    else:
        strat = st.lists(
            from_schema(items),
            min_size=min_size,
            max_size=max_size,
            unique_by=encode_canonical_json if unique else None,
        )
    if "contains" not in schema:
        return strat
    return strat.filter(lambda val: any(is_valid(x, schema["contains"]) for x in val))


def is_valid(instance: JSONType, schema: Schema) -> bool:
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


def object_schema(schema: dict) -> st.SearchStrategy[Dict[str, JSONType]]:
    """Handle a manageable subset of possible schemata for objects."""
    required = schema.get("required", [])  # required keys
    min_size = max(len(required), schema.get("minProperties", 0))
    names = schema.get("propertyNames", {})  # schema for optional keys
    if isinstance(names, dict) and "type" not in names:
        names["type"] = "string"
    elif names is True:
        names = {"type": "string"}
    elif names is False:
        assert min_size == 0, schema
        return st.builds(dict)

    properties = schema.get("properties", {})  # exact name: value schema
    properties = {k: canonicalish(v) for k, v in properties.items()}
    patterns = schema.get("patternProperties", {})  # regex for names: value schema
    patterns = {k: canonicalish(v) for k, v in patterns.items()}
    # schema for other values; handled specially if nothing matches
    additional = canonicalish(schema.get("additionalProperties", {}))
    additional_allowed = additional != FALSEY

    # When a known set of names is allowed, we cap the max_size at that number
    max_size = min(
        schema.get("maxProperties", float("inf")),
        len(schema.get("propertyNames", [])) + len(schema.get("properties", []))
        if ("propertyNames" in schema or "properties" in schema)
        and not additional_allowed
        else float("inf"),
    )

    dependencies = schema.get("dependencies", {})
    dep_names = {k: v for k, v in dependencies.items() if isinstance(v, list)}
    dep_schemas = {k: v for k, v in dependencies.items() if k not in dep_names}
    del dependencies

    name_strats = (
        st.sampled_from(sorted(dep_names) + sorted(dep_schemas) + sorted(properties))
        if (dep_names or dep_schemas or properties)
        else st.nothing(),
        from_schema(names) if additional_allowed else st.nothing(),
        st.one_of([st.from_regex(p) for p in sorted(patterns)]),
    )
    all_names_strategy = st.one_of([s for s in name_strats if not s.is_empty]).filter(
        partial(is_valid, schema=names)
    )

    @st.composite  # type: ignore
    def from_object_schema(draw: Any) -> Any:
        """Here, we do some black magic with private Hypothesis internals.

        It's unfortunate, but also the only way that I know of to satisfy all
        the interacting constraints without making shrinking totally hopeless.

        If any Hypothesis maintainers are reading this... I'm so, so sorry.
        """
        # Hypothesis internals are not type-annotated... I do mean *black* magic!
        elements = cu.many(
            draw(st.data()).conjecture_data,
            min_size=min_size,
            max_size=max_size,
            average_size=min(min_size + 5, (min_size + max_size) // 2),
        )
        out: dict = {}
        while elements.more():
            for key in required:
                if key not in out:
                    break
            else:
                for k in dep_names:
                    if k in out:
                        key = next((n for n in dep_names[k] if n not in out), None)
                        if key is not None:
                            break
                else:
                    key = draw(all_names_strategy.filter(lambda s: s not in out))

            pattern_schemas = [
                patterns[rgx]
                for rgx in sorted(patterns)
                if re.search(rgx, string=key) is not None
            ]
            if key in properties:
                pattern_schemas.insert(0, properties[key])

            if pattern_schemas:
                out[key] = draw(merged_as_strategies(pattern_schemas))
            else:
                assert additional_allowed
                out[key] = draw(from_schema(additional))

            for k, v in dep_schemas.items():
                if k in out and not is_valid(out, v):
                    out.pop(key)
                    elements.reject()

        for k in dep_names:
            if k in out:
                assume(all(n in out for n in dep_names[k]))
        return out

    return from_object_schema()


# OK, now on to the inverse: a strategy for generating schemata themselves.


def json_schemata() -> st.SearchStrategy[Union[bool, Schema]]:
    """A Hypothesis strategy for arbitrary JSON schemata.

    This strategy may generate anything that can be handled by `from_schema`,
    and is used to provide full branch coverage when testing this package.
    """
    return _json_schemata()


@st.composite  # type: ignore
def _json_schemata(draw: Any, recur: bool = True) -> Any:
    """Wrapped so we can disable the pylint error in one place only."""
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


@st.composite  # type: ignore
def gen_enum(draw: Any) -> Dict[str, List[JSONType]]:
    """Draw an enum schema."""
    elems = draw(st.lists(JSON_STRATEGY, 1, 10, unique_by=encode_canonical_json))
    # We do need this O(n^2) loop; see https://github.com/Julian/jsonschema/issues/529
    for i, val in enumerate(elems):
        assume(not any(val == v for v in elems[i + 1 :]))  # noqa
    return {"enum": elems}


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
    # We skip coverage for version branches because it's a pain to combine.
    boolean_bounds = not hasattr(jsonschema, "Draft7Validator")
    if lower is not None:
        if draw(st.booleans(), label="exclusiveMinimum"):
            if boolean_bounds:  # pragma: no cover
                out["exclusiveMinimum"] = True
                out["minimum"] = lower - 1
            else:
                out["exclusiveMinimum"] = lower - 1
        else:
            out["minimum"] = lower
    if upper is not None:
        if draw(st.booleans(), label="exclusiveMaximum"):
            if boolean_bounds:  # pragma: no cover
                out["exclusiveMaximum"] = True
                out["maximum"] = upper + 1
            else:
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
    format_ = draw(st.none() | st.sampled_from(JSON_SCHEMA_STRING_FORMATS))
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
    if patterns:
        out["patternProperties"] = patterns
    if additional is not None:
        out["additionalProperties"] = additional
    return out
