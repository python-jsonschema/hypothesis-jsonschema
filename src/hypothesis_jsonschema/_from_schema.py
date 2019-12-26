"""A Hypothesis extension for JSON schemata."""

import itertools
import math
import operator
import re
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Dict, List, Set

import hypothesis.internal.conjecture.utils as cu
import hypothesis.provisional as prov
import hypothesis.strategies as st
import jsonschema
from hypothesis import assume

from ._canonicalise import (
    FALSEY,
    JSON_STRATEGY,
    TRUTHY,
    TYPE_STRINGS,
    JSONType,
    Schema,
    canonicalish,
    encode_canonical_json,
    get_integer_bounds,
    get_number_bounds,
    get_type,
    is_valid,
    merged,
)


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
        if s is not None and s != FALSEY:
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
    if "allOf" in schema:
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
        "number": number_schema,
        "integer": integer_schema,
        "string": string_schema,
        "array": array_schema,
        "object": object_schema,
    }
    assert set(map_) == set(TYPE_STRINGS)
    return st.one_of([map_[t](schema) for t in get_type(schema)])


def integer_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle integer schemata."""
    # TODO: possibly generate value as a float if float(x) == x
    min_value, max_value = get_integer_bounds(schema)

    if "multipleOf" in schema:
        multiple_of = schema["multipleOf"]
        assert isinstance(multiple_of, (int, float))
        if min_value is not None:
            min_value = math.ceil(Fraction(min_value) / Fraction(multiple_of))
        if max_value is not None:
            max_value = math.floor(Fraction(max_value) / Fraction(multiple_of))
        strat = st.integers(min_value, max_value).map(lambda x: x * multiple_of)
        # check for and filter out float bounds, inexact multiplication, etc.
        return strat.filter(partial(is_valid, schema=schema))

    return st.integers(min_value, max_value)


def number_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle numeric schemata."""
    min_value, max_value, exclude_min, exclude_max = get_number_bounds(schema)

    if "multipleOf" in schema:
        multiple_of = schema["multipleOf"]
        assert isinstance(multiple_of, (int, float))
        if min_value is not None:
            min_value = math.ceil(Fraction(min_value) / Fraction(multiple_of))
        if max_value is not None:
            max_value = math.floor(Fraction(max_value) / Fraction(multiple_of))
        strat = st.integers(min_value, max_value).map(lambda x: x * multiple_of)
        # check for and filter out float bounds, inexact multiplication, etc.
        return strat.filter(partial(is_valid, schema=schema))

    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=exclude_min,
        exclude_max=exclude_max,
        # Filter out negative-zero as it does not exist in JSON
    ).filter(lambda n: n != 0 or math.copysign(1, n) == 1)


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
    """Get a strategy for date or time strings in the given RFC3339 format.

    See https://tools.ietf.org/html/rfc3339#section-5.6
    """
    # Hmm, https://github.com/HypothesisWorks/hypothesis/issues/170
    # would make this a lot easier...
    assert name in RFC3339_FORMATS

    def zfill(width: int) -> Callable[[int], str]:
        return lambda v: str(v).zfill(width)

    simple = {
        "date-fullyear": st.integers(0, 9999).map(zfill(4)),
        "date-month": st.integers(1, 12).map(zfill(2)),
        "date-mday": st.integers(1, 28).map(zfill(2)),  # incomplete but valid
        "time-hour": st.integers(0, 23).map(zfill(2)),
        "time-minute": st.integers(0, 59).map(zfill(2)),
        "time-second": st.integers(0, 59).map(zfill(2)),  # ignore negative leap seconds
        "time-secfrac": st.from_regex(r"\.[0-9]+"),
    }
    if name in simple:
        return simple[name]
    if name == "time-numoffset":
        return st.tuples(
            st.sampled_from(["+", "-"]), rfc3339("time-hour"), rfc3339("time-minute")
        ).map("%s%s:%s".__mod__)
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
    """Return a recursive strategy for simple regular expression patterns."""
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
    **{name: rfc3339(name) for name in RFC3339_FORMATS},
    "date": rfc3339("full-date"),
    "time": rfc3339("full-time"),
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
        if "pattern" in schema:
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
    return strategy.filter(lambda s: min_size <= len(s) <= max_size)


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


def object_schema(schema: dict) -> st.SearchStrategy[Dict[str, JSONType]]:
    """Handle a manageable subset of possible schemata for objects."""
    required = schema.get("required", [])  # required keys
    min_size = max(len(required), schema.get("minProperties", 0))
    max_size = schema.get("maxProperties", math.inf)
    assert min_size <= max_size, (min_size, max_size)

    names = schema.get("propertyNames", {})  # schema for optional keys
    if names == FALSEY:
        assert min_size == 0, schema
        return st.builds(dict)
    names["type"] = "string"

    properties = schema.get("properties", {})  # exact name: value schema
    patterns = schema.get("patternProperties", {})  # regex for names: value schema
    # schema for other values; handled specially if nothing matches
    additional = schema.get("additionalProperties", {})
    additional_allowed = additional != FALSEY

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
        """Do some black magic with private Hypothesis internals for objects.

        It's unfortunate, but also the only way that I know of to satisfy all
        the interacting constraints without making shrinking totally hopeless.

        If any Hypothesis maintainers are reading this... I'm so, so sorry.
        """
        # Hypothesis internals are not type-annotated... I do mean *black* magic!
        elements = cu.many(
            draw(st.data()).conjecture_data,
            min_size=min_size,
            max_size=max_size,
            average_size=min(min_size + 5, (min_size + max_size) / 2),
        )
        out: dict = {}
        while elements.more():
            for key in required:
                if key not in out:
                    break
            else:
                for k in set(dep_names).intersection(out):
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
                out[key] = draw(from_schema(additional))

            for k, v in dep_schemas.items():
                if k in out and not is_valid(out, v):
                    out.pop(key)
                    elements.reject()

        for k in set(dep_names).intersection(out):
            assume(set(out).issuperset(dep_names[k]))
        return out

    return from_object_schema()
