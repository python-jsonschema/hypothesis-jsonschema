"""A Hypothesis extension for JSON schemata."""

import itertools
import json
import math
import re
from typing import Any, Dict, List, Union

import hypothesis.internal.conjecture.utils as cu
import hypothesis.provisional as prov
import hypothesis.strategies as st
import jsonschema
from hypothesis import assume
from hypothesis.errors import InvalidArgument

# Mypy does not (yet!) support recursive type definitions.
# (and writing a few steps by hand is a DoS attack on the AST walker in Pytest)
JSONType = Union[None, bool, float, str, list, Dict[str, Any]]

JSON_STRATEGY: st.SearchStrategy[JSONType] = st.deferred(
    lambda: st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False).map(lambda x: x or 0.0),
        st.text(),
        st.lists(JSON_STRATEGY, max_size=3),
        st.dictionaries(st.text(), JSON_STRATEGY, max_size=3),
    )
)


def encode_canonical_json(value: JSONType) -> str:
    """Canonical form serialiser, for uniqueness testing."""
    return json.dumps(value, sort_keys=True)


def canonicalish(schema: JSONType) -> Dict:
    """Turn booleans into dict-based schemata."""
    if schema is True:
        return {}
    elif schema is False:
        return {"not": {}}
    assert isinstance(schema, dict), schema
    return schema


def merged(schemas):
    """Merge *n* schemas into a single schema, or None if result is invalid.
    Takes the logical intersection, so any object that validates against the returned
    schema must also validate against all of the input schemas.
    """
    assert schemas, "internal error: must pass at least one schema to merge"
    assert all(s == canonicalish(s) for s in schemas)

    out = dict(schemas[0])
    for s in schemas[1:]:
        if set(s) & set(out):
            # Overlapping keys.
            # TODO: convert types to list, not with allOf, etc.
            return None
        out.update(s)
    # hopefully this catches empty things like allOf different types?
    jsonschema.validators.validator_for(out).check_schema(out)
    return out


def merged_as_strategies(schemas):
    assert schemas, "internal error: must pass at least one schema to merge"
    assert all(s == canonicalish(s) for s in schemas)
    if len(schemas) == 1:
        return from_schema(schemas[0])
    # Try to merge combinations of strategies.
    strats = []
    combined = set()
    inputs = {encode_canonical_json(s): s for s in schemas}
    for group in itertools.chain.from_iterable(
        itertools.combinations(inputs, n) for n in range(len(inputs), 0, -1)
    ):
        if combined.issuperset(group):
            continue
        s = merged([inputs[g] for g in group])
        if s is not None:
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
    # Boolean objects are special schemata; False rejects all and True accepts all.
    if schema is False or schema == {"not": {}}:
        return st.nothing()
    if schema is True or schema == {}:
        return JSON_STRATEGY
    # Otherwise, we're dealing with "objects", i.e. dicts.
    if not isinstance(schema, dict):
        raise InvalidArgument(
            f"Got schema={schema} of type {type(schema).__name__}, "
            "but expected a dict."
        )
    # Only check if declared, lest we error on inner non-latest-draft schemata.
    if "$schema" in schema:
        jsonschema.validators.validator_for(schema).check_schema(schema)

    # Now we handle as many validation keywords as we can...
    # Applying subschemata with boolean logic
    if "not" in schema:
        if schema["not"] is True or schema["not"] == {}:
            return st.nothing()
        if schema["not"] is False:
            return JSON_STRATEGY
        return JSON_STRATEGY.filter(lambda inst: not is_valid(inst, schema["not"]))
    if "anyOf" in schema:
        tmp = schema.copy()
        return st.one_of(
            [from_schema({**tmp, **canonicalish(s)}) for s in tmp.pop("anyOf")]
        )
    if "allOf" in schema:
        tmp = schema.copy()
        ao = [canonicalish(s) for s in tmp.pop("allOf")]
        if any(s == canonicalish(False) for s in ao):
            return st.nothing()
        return merged_as_strategies([tmp] + ao)
    if "oneOf" in schema:
        tmp = schema.copy()
        schemas = [{**tmp, **canonicalish(s)} for s in tmp.pop("oneOf")]
        schemas = [s for s in schemas if s != canonicalish(False)]
        if len(schemas) > len({encode_canonical_json(s) for s in schemas}):
            return st.nothing()
        return st.one_of([from_schema(s) for s in schemas]).filter(
            lambda inst: 1 == sum(is_valid(inst, s) for s in schemas)
        )
    # Conditional application of subschemata
    if "if" in schema:
        if_ = schema["if"]
        then = schema.get("then", {})
        else_ = schema.get("else", {})
        return st.one_of(
            from_schema(then).filter(lambda v: is_valid(v, if_)),
            from_schema(else_).filter(lambda v: not is_valid(v, if_)),
            from_schema(if_).filter(lambda v: is_valid(v, then) or is_valid(v, else_)),
        )
    # Simple special cases
    if "enum" in schema:
        return st.sampled_from(schema["enum"]) if schema["enum"] else st.nothing()
    if "const" in schema:
        return st.just(schema["const"])
    # Finally, resolve schema by type - defaulting to "object"
    if "type" in schema:
        type_ = schema["type"]
    else:
        type_ = []
        for t, kw in [
            ("number", "multipleOf maximum exclusiveMaximim minimum exclusiveMinimum"),
            ("integer", "multipleOf maximum exclusiveMaximim minimum exclusiveMinimum"),
            ("string", "maxLength minLength pattern contentEncoding contentMediaType"),
            ("array", "items additionalItems maxItems uniqueItems contains"),
            (
                "object",
                "maxProperties minProperties required properties patternProperties "
                "additionalProperties dependencies propertyNames",
            ),
        ]:
            if any(k in schema for k in kw.split()):
                type_.append(t)
        if not type_:
            type_ = "object"
    if not isinstance(type_, list):
        assert isinstance(type_, str), schema
        type_ = [type_]
    map_ = dict(
        null=lambda _: st.none(),
        boolean=lambda _: st.booleans(),
        number=numeric_schema,
        integer=numeric_schema,
        string=string_schema,
        array=array_schema,
        object=object_schema,
    )
    return st.one_of([map_[t](schema) for t in type_])  # type: ignore


def numeric_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle numeric schemata."""
    type_ = schema.get("type", "integer")
    multiple_of = schema.get("multipleOf")
    lower = schema.get("minimum")
    upper = schema.get("maximum")

    exmin = schema.get("exclusiveMinimum")
    if exmin is True and "integer" in type_:  # pragma: no cover
        lower += 1
    elif exmin is not False and exmin is not None:
        lo = exmin + 1 if int(exmin) == exmin else math.ceil(exmin)
        if lower is None:
            lower = lo if "integer" in type_ else exmin
        else:  # pragma: no cover
            lower = max(lower, lo if "integer" in type_ else exmin)

    exmax = schema.get("exclusiveMaximum")
    if exmax is True and "integer" in type_:  # pragma: no cover
        upper -= 1
    elif exmax is not False and exmax is not None:
        hi = exmax - 1 if int(exmax) == exmax else math.floor(exmax)
        if upper is None:
            upper = hi if "integer" in type_ else exmax
        else:  # pragma: no cover
            upper = min(upper, hi if "integer" in type_ else exmax)

    if multiple_of is not None:
        assert isinstance(multiple_of, (int, float))
        if lower is not None:
            lo = math.ceil(lower / multiple_of)
            if exmin is not False and exmin is not None and lo * multiple_of == lower:
                lo += 1
            assert lo * multiple_of >= lower, (lower, lo)
            lower = lo
        if upper is not None:
            hi = math.floor(upper / multiple_of)
            if exmax is not False and exmax is not None and hi * multiple_of == upper:
                hi -= 1
            assert hi * multiple_of <= upper, (upper, hi)
            upper = hi
        strat = st.integers(lower, upper).map(lambda x: x * multiple_of)
        if isinstance(multiple_of, float):
            return strat.filter(lambda x: int(x / multiple_of) == x / multiple_of)
        return strat

    strat = st.nothing()
    if "integer" in type_:
        lo = lower if lower is None else math.ceil(lower)
        hi = upper if upper is None else math.floor(upper)
        if lo is None or hi is None or lo <= hi:
            strat = st.integers(lo, hi)
    if "number" in type_:
        # Negative-zero does not round trip through JSON, so force it to positive
        strat |= st.floats(
            min_value=exmin if lower is None else lower,
            max_value=exmax if upper is None else upper,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=exmin is not None,
            exclude_max=exmax is not None,
        ).map(lambda n: 0.0 if n == 0 else n)
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
        return st.just("Z") | rfc3339("time-numoffset")  # type: ignore
    if name == "partial-time":
        return st.times().map(str)
    if name == "full-date":
        return st.dates().map(str)
    if name == "full-time":
        return st.tuples(rfc3339("partial-time"), rfc3339("time-offset")).map("".join)
    assert name == "date-time"
    return st.tuples(rfc3339("full-date"), rfc3339("full-time")).map("T".join)


def string_schema(schema: dict) -> st.SearchStrategy[str]:
    """Handle schemata for strings."""
    # also https://json-schema.org/latest/json-schema-validation.html#rfc.section.7
    min_size = schema.get("minLength", 0)
    max_size = schema.get("maxLength", float("inf"))
    strategy: Any = st.text(min_size=min_size, max_size=schema.get("maxLength"))
    assert not (
        "format" in schema and "pattern" in schema
    ), "format and regex constraints are supported, but not both at once."
    if "pattern" in schema:
        strategy = st.from_regex(schema["pattern"])
    elif "format" in schema:
        url_synonyms = ["uri", "uri-reference", "iri", "iri-reference", "uri-template"]
        domains = prov.domains()  # type: ignore
        strategy = {
            # A value of None indicates a known but unsupported format.
            **{name: rfc3339(name) for name in RFC3339_FORMATS},
            "date": rfc3339("full-date"),
            "time": rfc3339("full-time"),
            "email": st.emails(),  # type: ignore
            "idn-email": st.emails(),  # type: ignore
            "hostname": domains,
            "idn-hostname": domains,
            "ipv4": prov.ip4_addr_strings(),  # type: ignore
            "ipv6": prov.ip6_addr_strings(),  # type: ignore
            **{name: domains.map("https://{}".format) for name in url_synonyms},
            "json-pointer": st.just(""),
            "relative-json-pointer": st.just(""),
            "regex": REGEX_PATTERNS,
        }.get(schema["format"])
        if strategy is None:
            raise InvalidArgument(f"Unsupported string format={schema['format']}")
    return strategy.filter(lambda s: min_size <= len(s) <= max_size)  # type: ignore


def array_schema(schema: dict) -> st.SearchStrategy[List[JSONType]]:
    """Handle schemata for arrays."""
    items = schema.get("items", {})
    additional_items = schema.get("additionalItems", {})
    min_size = schema.get("minItems", 0)
    max_size = schema.get("maxItems")
    unique = schema.get("uniqueItems")
    contains = schema.get("contains")
    if isinstance(items, list):
        for i, s in enumerate(items):
            if canonicalish(s) == canonicalish(False):
                # No item can validate against this position,
                # so we cut off the elements to generate here.
                assert min_size <= i
                assert max_size is None or max_size >= i
                items = items[:i]
                min_size = max_size = i
                break
        min_size = max(0, min_size - len(items))
        if max_size is not None:
            max_size -= len(items)
        if contains is not None:
            assert (
                additional_items == {}
            ), "Cannot handle additionalItems and contains togther"
            additional_items = contains
            min_size = max(min_size, 1)
        fixed_items = st.tuples(*map(from_schema, items))
        extra_items = st.lists(
            from_schema(additional_items), min_size=min_size, max_size=max_size
        )
        return st.tuples(fixed_items, extra_items).map(
            lambda t: list(t[0]) + t[1]  # type: ignore
        )
    if contains is not None:
        assert items == {}, "Cannot handle items and contains togther"
        items = contains
        min_size = max(min_size, 1)
    if unique:
        return st.lists(
            from_schema(items),
            min_size=min_size,
            max_size=max_size,
            unique_by=encode_canonical_json,
        )
    return st.lists(from_schema(items), min_size=min_size, max_size=max_size)


def is_valid(instance: JSONType, schema: JSONType) -> bool:
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
    additional_allowed = additional != canonicalish(False)

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
        st.sampled_from(sorted(dep_names) + sorted(dep_schemas))
        if (dep_names or dep_schemas)
        else st.nothing(),
        from_schema(names) if additional_allowed else st.nothing(),
        st.sampled_from(sorted(properties)) if properties else st.nothing(),
        st.one_of([st.from_regex(p) for p in sorted(patterns)]),
    )
    all_names_strategy = st.one_of([s for s in name_strats if not s.is_empty]).filter(
        lambda instance: is_valid(instance, names)
    )

    @st.composite
    def from_object_schema(draw: Any) -> Any:
        """Here, we do some black magic with private Hypothesis internals.

        It's unfortunate, but also the only way that I know of to satisfy all
        the interacting constraints without making shrinking totally hopeless.

        If any Hypothesis maintainers are reading this... I'm so, so sorry.
        """
        elements = cu.many(  # type: ignore
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


def json_schemata() -> st.SearchStrategy[Union[bool, Dict[str, JSONType]]]:
    """A Hypothesis strategy for arbitrary JSON schemata.

    This strategy may generate anything that can be handled by `from_schema`,
    and is used to provide full branch coverage when testing this package.
    """
    return _json_schemata()


@st.composite
def regex_patterns(draw: Any) -> st.SearchStrategy[str]:
    """A strategy for simple regular expression patterns."""
    fragments = st.one_of(
        st.just("."),
        st.from_regex(r"\[\^?[A-Za-z0-9]+\]"),
        REGEX_PATTERNS.map("{}+".format),
        REGEX_PATTERNS.map("{}?".format),
        REGEX_PATTERNS.map("{}*".format),
    )
    result = draw(st.lists(fragments, min_size=1, max_size=3).map("".join))
    try:
        re.compile(result)
    except re.error:
        assume(False)
    return result  # type: ignore


REGEX_PATTERNS = regex_patterns()


@st.composite
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


@st.composite
def gen_enum(draw: Any) -> Dict[str, List[JSONType]]:
    """Draw an enum schema."""
    elems = draw(st.lists(JSON_STRATEGY, 1, 10, unique_by=encode_canonical_json))
    # We do need this O(n^2) loop; see https://github.com/Julian/jsonschema/issues/529
    for i, val in enumerate(elems):
        assume(not any(val == v for v in elems[i + 1 :]))  # noqa
    return dict(enum=elems)


@st.composite
def gen_if_then_else(draw: Any) -> Dict[str, JSONType]:
    """Draw a conditional schema."""
    # Cheat by using identical if and then schemata, else accepting anything.
    if_schema = draw(json_schemata().filter(lambda v: bool(v and v is not True)))
    return {"if": if_schema, "then": if_schema, "else": {}}


@st.composite
def gen_number(draw: Any, kind: str) -> Dict[str, Union[str, float]]:
    """Draw a numeric schema."""
    max_int_float = 2 ** 53
    lower = draw(st.none() | st.integers(-max_int_float, max_int_float))
    upper = draw(st.none() | st.integers(-max_int_float, max_int_float))
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    multiple_of = draw(st.none() | st.integers(2, 100))
    assume(None in (multiple_of, lower, upper) or multiple_of <= (upper - lower))
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


@st.composite
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


@st.composite
def gen_array(draw: Any) -> Dict[str, JSONType]:
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


@st.composite
def gen_object(draw: Any) -> Dict[str, JSONType]:
    """Draw an object schema."""
    out: Dict[str, JSONType] = {"type": "object"}
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
