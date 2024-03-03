"""A Hypothesis extension for JSON schemata."""

import itertools
import math
import operator
import re
import warnings
from copy import deepcopy
from fractions import Fraction
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union

import jsonschema
import jsonschema.exceptions
from hypothesis import assume, provisional as prov, strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.regex import regex_strategy
from hypothesis.strategies._internal.strings import OneCharStringStrategy

from ._canonicalise import (
    FALSEY,
    TRUTHY,
    TYPE_STRINGS,
    HypothesisRefResolutionError,
    Schema,
    canonicalish,
    get_integer_bounds,
    get_number_bounds,
    get_type,
    make_validator,
    merged,
)
from ._encode import JSONType, encode_canonical_json
from ._resolve import resolve_all_refs

JSON_STRATEGY: st.SearchStrategy[JSONType] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False).map(lambda x: x or 0.0)
    | st.text(),
    lambda strategy: st.lists(strategy, max_size=3)
    | st.dictionaries(st.text(), strategy, max_size=3),
)
_FORMATS_TOKEN = object()


class CharStrategy(OneCharStringStrategy):
    allow_x00: bool
    codec: Optional[str]

    @classmethod
    def from_args(cls, *, allow_x00: bool, codec: Optional[str]) -> "CharStrategy":
        self: CharStrategy = cls.from_characters_args(
            min_codepoint=0 if allow_x00 else 1, codec=codec
        )
        self.allow_x00 = allow_x00
        self.codec = codec
        return self

    def check_name_allowed(self, name: str) -> None:
        if "\x00" in name and not self.allow_x00:
            raise InvalidArgument(f"allow_x00=False makes name {name!a} invalid")
        if self.codec is not None:
            try:
                name.encode(self.codec)
            except Exception:
                msg = f"{name!r} cannot be encoded as {self.codec!r}"
                raise InvalidArgument(msg) from None


def from_js_regex(pattern: str, alphabet: CharStrategy) -> st.SearchStrategy[str]:
    return regex_strategy(
        pattern,
        fullmatch=False,
        alphabet=alphabet,
        _temp_jsonschema_hack_no_end_newline=True,
    )


def merged_as_strategies(
    schemas: List[Schema],
    *,
    alphabet: CharStrategy,
    custom_formats: Optional[Dict[str, st.SearchStrategy[str]]],
) -> st.SearchStrategy[JSONType]:
    assert schemas, "internal error: must pass at least one schema to merge"
    if len(schemas) == 1:
        return __from_schema(
            schemas[0], alphabet=alphabet, custom_formats=custom_formats
        )
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
                __from_schema(
                    s, alphabet=alphabet, custom_formats=custom_formats
                ).filter(
                    lambda obj, validators=tuple(
                        make_validator(s).is_valid for s in schemas
                    ): all(v(obj) for v in validators)
                )
            )
            combined.update(group)
    return st.one_of(strats)


def from_schema(
    schema: Union[bool, Schema],
    *,
    custom_formats: Optional[Dict[str, st.SearchStrategy[str]]] = None,
    allow_x00: bool = True,
    codec: Optional[str] = "utf-8",
) -> st.SearchStrategy[JSONType]:
    """Take a JSON schema and return a strategy for allowed JSON objects.

    To generate specific string formats, pass a ``custom_formats`` dict
    mapping the format name to a strategy for allowed strings.

    You can constrain strings _other than those from custom format strategies_
    by passing ``allow_x00=False`` to exclude the null character ``chr(0)``,
    and/or a ``codec=`` name such as ``"utf-8"``, ``"ascii"``, or any other
    text encoding supported by Python.

    Supports JSONSchema drafts 04, 06, and 07, with the exception of
    recursive references.
    """
    try:
        return __from_schema(
            deepcopy(schema),
            custom_formats=custom_formats,
            alphabet=CharStrategy.from_args(allow_x00=allow_x00, codec=codec),
        )
    except Exception as err:
        error = err

        def error_raiser() -> NoReturn:
            raise error

        return st.builds(error_raiser)


def _get_format_filter(
    format_name: str,
    checker: jsonschema.FormatChecker,
    strategy: st.SearchStrategy[str],
) -> st.SearchStrategy[str]:
    def check_valid(string: str) -> str:
        try:
            if not isinstance(string, str):
                raise jsonschema.exceptions.FormatError(f"{string!r} is not a string")
            checker.check(string, format=format_name)
        except jsonschema.exceptions.FormatError as err:
            raise InvalidArgument(
                f"Got string={string!r} from strategy {strategy!r}, but this "
                f"is not a valid value for the {format_name!r} checker."
            ) from err
        return string

    return strategy.map(check_valid)


def __from_schema(
    schema: Union[bool, Schema],
    *,
    alphabet: CharStrategy,
    custom_formats: Optional[Dict[str, st.SearchStrategy[str]]],
) -> st.SearchStrategy[JSONType]:
    try:
        schema = resolve_all_refs(schema)
    except RecursionError:
        raise HypothesisRefResolutionError(
            f"Could not resolve recursive references in schema={schema!r}"
        ) from None
    # We check for _FORMATS_TOKEN to avoid re-validating known good data.
    if custom_formats is not None and _FORMATS_TOKEN not in custom_formats:
        assert isinstance(custom_formats, dict)
        for name, strat in custom_formats.items():
            if not isinstance(name, str):
                raise InvalidArgument(f"format name {name!r} must be a string")
            if not isinstance(strat, st.SearchStrategy):
                raise InvalidArgument(
                    f"custom_formats[{name!r}]={strat!r} must be a Hypothesis "
                    "strategy which generates strings matching this format."
                )
            if name in STRING_FORMATS:
                warnings.warn(
                    f"Overriding standard format {name!r} - was "
                    f"{STRING_FORMATS[name]!r}, now {strat!r}",
                    HypothesisWarning,
                    stacklevel=2,
                )
        format_checker = jsonschema.FormatChecker()
        custom_formats = {
            name: _get_format_filter(name, format_checker, strategy)
            for name, strategy in custom_formats.items()
        }
        custom_formats[_FORMATS_TOKEN] = None  # type: ignore

    schema = canonicalish(schema)
    # Boolean objects are special schemata; False rejects all and True accepts all.
    if schema == FALSEY:
        return st.nothing()
    if schema == TRUTHY:
        return JSON_STRATEGY
    assert isinstance(schema, dict)
    if schema.get("$schema") == "http://json-schema.org/draft-03/schema#":
        raise InvalidArgument("Draft-03 schemas are not supported")
    # Only check if declared, lest we error on inner non-latest-draft schemata.
    if "$schema" in schema:
        jsonschema.validators.validator_for(schema).check_schema(schema)

    # Now we handle as many validation keywords as we can...
    # Applying subschemata with boolean logic
    if "not" in schema:
        not_ = schema.pop("not")
        assert isinstance(not_, dict)
        validator = make_validator(not_).is_valid
        return __from_schema(
            schema, alphabet=alphabet, custom_formats=custom_formats
        ).filter(lambda v: not validator(v))
    if "anyOf" in schema:
        tmp = schema.copy()
        ao = tmp.pop("anyOf")
        assert isinstance(ao, list)
        return st.one_of(
            [
                merged_as_strategies(
                    [tmp, s], alphabet=alphabet, custom_formats=custom_formats
                )
                for s in ao
            ]
        )
    if "allOf" in schema:
        tmp = schema.copy()
        ao = tmp.pop("allOf")
        assert isinstance(ao, list)
        return merged_as_strategies(
            [tmp, *ao], alphabet=alphabet, custom_formats=custom_formats
        )
    if "oneOf" in schema:
        tmp = schema.copy()
        oo = tmp.pop("oneOf")
        assert isinstance(oo, list)
        schemas = [merged([tmp, s]) for s in oo]
        return st.one_of(
            [
                __from_schema(s, alphabet=alphabet, custom_formats=custom_formats)
                for s in schemas
                if s is not None
            ]
        ).filter(make_validator(schema).is_valid)
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
        "string": partial(string_schema, custom_formats, alphabet),
        "array": partial(array_schema, custom_formats, alphabet),
        "object": partial(object_schema, custom_formats, alphabet),
    }
    assert set(map_) == set(TYPE_STRINGS)
    return st.one_of([map_[t](schema) for t in get_type(schema)])


def _numeric_with_multiplier(
    min_value: Optional[float], max_value: Optional[float], schema: Schema
) -> st.SearchStrategy[float]:
    """Handle numeric schemata containing the multipleOf key."""
    multiple_of = schema["multipleOf"]
    assert isinstance(multiple_of, (int, float))
    if min_value is not None:
        min_value = math.ceil(Fraction(min_value) / Fraction(multiple_of))
    if max_value is not None:
        max_value = math.floor(Fraction(max_value) / Fraction(multiple_of))
    if min_value is not None and max_value is not None and min_value > max_value:  # type: ignore[unreachable]
        # You would think that this is impossible, but it can happen if multipleOf
        # is very small and the bounds are very close togther.  It would be nicer
        # to deal with this when canonicalising, but suffice to say we can't without
        # diverging from the floating-point behaviour of the upstream validator.
        return st.nothing()  # type: ignore[unreachable]
    return (
        st.integers(min_value, max_value)
        .map(lambda x: x * multiple_of)
        .filter(make_validator(schema).is_valid)
    )


def integer_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle integer schemata."""
    min_value, max_value = get_integer_bounds(schema)
    if "multipleOf" in schema:
        return _numeric_with_multiplier(min_value, max_value, schema)
    return st.integers(min_value, max_value)


def number_schema(schema: dict) -> st.SearchStrategy[float]:
    """Handle numeric schemata."""
    min_value, max_value, exclude_min, exclude_max = get_number_bounds(schema)
    if "multipleOf" in schema:
        return _numeric_with_multiplier(min_value, max_value, schema)
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=exclude_min,
        exclude_max=exclude_max,
        # Filter out negative-zero as it does not exist in JSON
    ).map(lambda n: n if n != 0 else abs(n))


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
    if name in ("date", "full-date"):
        return st.dates().map(str)
    if name in ("time", "full-time"):
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
    except (re.error, FutureWarning):
        assume(False)
    return result


REGEX_PATTERNS = regex_patterns()


def json_pointers(alphabet: CharStrategy) -> st.SearchStrategy[str]:
    """Return a strategy for strings in json-pointer format."""
    return st.lists(
        st.text(alphabet).map(lambda p: "/" + p.replace("~", "~0").replace("/", "~1"))
    ).map("".join)


def relative_json_pointers(alphabet: CharStrategy) -> st.SearchStrategy[str]:
    """Return a strategy for strings in relative-json-pointer format."""
    return st.builds(
        operator.add,
        st.from_regex(r"0|[1-9][0-9]*", fullmatch=True, alphabet=alphabet),
        st.just("#") | json_pointers(alphabet),
    )


# Via the `webcolors` package, to match the logic `jsonschema`
# uses to check it's (non-standard?) "color" format.
_WEBCOLOR_REGEX = "^#([a-fA-F0-9]{3}|[a-fA-F0-9]{6})$"
_CSS21_COLOR_NAMES = (
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "green",
    "gray",
    "lime",
    "maroon",
    "navy",
    "olive",
    "orange",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
)

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
STRING_FORMATS = {
    **{name: rfc3339(name) for name in RFC3339_FORMATS},
    "color": st.from_regex(_WEBCOLOR_REGEX) | st.sampled_from(_CSS21_COLOR_NAMES),
    "date": rfc3339("full-date"),
    "time": rfc3339("full-time"),
    "email": st.emails(),
    "idn-email": st.emails(),
    "hostname": prov.domains(),
    "idn-hostname": prov.domains(),
    "ipv4": st.ip_addresses(v=4).map(str),
    "ipv6": st.ip_addresses(v=6).map(str),
    **{
        name: prov.domains().map("https://{}".format)
        for name in ["uri", "uri-reference", "iri", "iri-reference", "uri-template"]
    },
    "json-pointer": json_pointers,
    "relative-json-pointer": relative_json_pointers,
    "regex": REGEX_PATTERNS,
}


def _warn_invalid_regex(pattern: str, err: re.error, kw: str = "pattern") -> None:
    warnings.warn(
        f"Got {kw}={pattern!r}, but this is not valid syntax for a Python regular "
        f"expression ({err}) so it will not be handled by the strategy.  See https://"
        "json-schema.org/understanding-json-schema/reference/regular_expressions.html",
        stacklevel=2,
    )


def string_schema(
    custom_formats: Dict[str, st.SearchStrategy[str]],
    alphabet: CharStrategy,
    schema: dict,
) -> st.SearchStrategy[str]:
    """Handle schemata for strings."""
    # also https://json-schema.org/latest/json-schema-validation.html#rfc.section.7
    min_size = schema.get("minLength", 0)
    max_size = schema.get("maxLength")
    strategy = st.text(alphabet, min_size=min_size, max_size=max_size)
    known_formats = {**STRING_FORMATS, **(custom_formats or {})}
    if schema.get("format") in known_formats:
        # Unknown "format" specifiers should be ignored for validation.
        # See https://json-schema.org/latest/json-schema-validation.html#format
        strategy = known_formats[schema["format"]]
        if not isinstance(strategy, st.SearchStrategy):
            strategy = strategy(alphabet)
        if "pattern" in schema:
            try:
                # This isn't really supported, but we'll do our best with a filter.
                strategy = strategy.filter(re.compile(schema["pattern"]).search)
            except re.error as err:
                _warn_invalid_regex(schema["pattern"], err)
                return st.nothing()
    elif "pattern" in schema:
        try:
            re.compile(schema["pattern"])
            strategy = from_js_regex(schema["pattern"], alphabet=alphabet)
        except re.error as err:
            # Patterns that are invalid in Python, or just malformed
            _warn_invalid_regex(schema["pattern"], err)
            return st.nothing()
    # If we have size bounds but we're generating strings from a regex or pattern,
    # apply a filter to ensure our size bounds are respected.
    if ("format" in schema or "pattern" in schema) and (
        min_size != 0 or max_size is not None
    ):
        max_size = math.inf if max_size is None else max_size
        strategy = strategy.filter(lambda s: min_size <= len(s) <= max_size)
    return strategy


def array_schema(
    custom_formats: Dict[str, st.SearchStrategy[str]],
    alphabet: CharStrategy,
    schema: dict,
) -> st.SearchStrategy[List[JSONType]]:
    """Handle schemata for arrays."""
    _from_schema_ = partial(
        __from_schema, custom_formats=custom_formats, alphabet=alphabet
    )
    items = schema.get("items", {})
    additional_items = schema.get("additionalItems", {})
    min_size = schema.get("minItems", 0)
    max_size = schema.get("maxItems")
    unique = schema.get("uniqueItems")
    if isinstance(items, list):
        min_size = max(0, min_size - len(items))
        if max_size is not None:
            max_size -= len(items)

        items_strats = [_from_schema_(s) for s in items]
        additional_items_strat = _from_schema_(additional_items)

        # If we have a contains schema to satisfy, we try generating from it when
        # allowed to do so.  We'll skip the None (unmergable / no contains) cases
        # below, and let Hypothesis ignore the FALSEY cases for us.
        if "contains" in schema:
            for i, mrgd in enumerate(merged([schema["contains"], s]) for s in items):
                if mrgd is not None:
                    items_strats[i] |= _from_schema_(mrgd)
            contains_additional = merged([schema["contains"], additional_items])
            if contains_additional is not None:
                additional_items_strat |= _from_schema_(contains_additional)

        # Hypothesis raises InvalidArgument for empty elements and non-None
        # max_size, because the user has asked for a possibility which will
        # never happen... but we can work around that here.
        if additional_items_strat.is_empty:
            if min_size >= 1:
                return st.nothing()
            max_size = 0

        if unique:

            @st.composite  # type: ignore
            def compose_lists_with_filter(draw: Any) -> List[JSONType]:
                elems = []
                seen: Set[str] = set()

                def not_seen(elem: JSONType) -> bool:
                    return encode_canonical_json(elem) not in seen

                for strat in items_strats:
                    elems.append(draw(strat.filter(not_seen)))
                    seen.add(encode_canonical_json(elems[-1]))
                if max_size == 0:
                    return elems
                extra_items = st.lists(
                    additional_items_strat.filter(not_seen),
                    min_size=min_size,
                    max_size=max_size,
                    unique_by=encode_canonical_json,
                )
                more_elems: List[JSONType] = draw(extra_items)
                return elems + more_elems

            strat = compose_lists_with_filter()
        elif max_size == 0:
            strat = st.tuples(*items_strats).map(list)
        else:
            strat = st.builds(
                operator.add,
                st.tuples(*items_strats).map(list),
                st.lists(additional_items_strat, min_size=min_size, max_size=max_size),
            )
    else:
        items_strat = _from_schema_(items)
        if "contains" in schema:
            contains_strat = _from_schema_(schema["contains"])
            if merged([items, schema["contains"]]) != schema["contains"]:
                # We only need this filter if we couldn't merge items in when
                # canonicalising.  Note that for list-items, above, we just skip
                # the mixed generation in this case (because they tend to be
                # heterogeneous) and hope it works out anyway.
                contains_strat = contains_strat.filter(make_validator(items).is_valid)
            items_strat |= contains_strat
        elif items_strat.is_empty and min_size == 0 and max_size is not None:
            # As above, work around a Hypothesis check for unsatisfiable max_size.
            return st.builds(list)
        strat = st.lists(
            items_strat,
            min_size=min_size,
            max_size=max_size,
            unique_by=encode_canonical_json if unique else None,
        )
    if "contains" not in schema:
        return strat
    contains = make_validator(schema["contains"]).is_valid
    return strat.filter(lambda val: any(contains(x) for x in val))


def object_schema(
    custom_formats: Dict[str, st.SearchStrategy[str]],
    alphabet: CharStrategy,
    schema: dict,
) -> st.SearchStrategy[Dict[str, JSONType]]:
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

    for key in list(patterns):
        try:
            re.compile(key)
        except re.error as err:
            _warn_invalid_regex(key, err, "patternProperties entry")
            if min_size == 0 and not required:
                return st.builds(dict)
            return st.nothing()

    dependencies = schema.get("dependencies", {})
    dep_names = {k: v for k, v in dependencies.items() if isinstance(v, list)}
    dep_schemas = {k: v for k, v in dependencies.items() if k not in dep_names}
    del dependencies

    valid_name = make_validator(names).is_valid
    known: set = set(filter(valid_name, set(dep_names).union(dep_schemas, properties)))
    for name in sorted(known.union(required)):
        alphabet.check_name_allowed(name)
    known_optional_names: List[str] = sorted(known - set(required))
    name_strats = (
        (
            __from_schema(names, custom_formats=custom_formats, alphabet=alphabet)
            if additional_allowed
            else st.nothing()
        ),
        st.sampled_from(known_optional_names) if known_optional_names else st.nothing(),
        st.one_of(
            [
                from_js_regex(p, alphabet=alphabet).filter(valid_name)
                for p in sorted(patterns)
            ]
        ),
    )
    all_names_strategy = st.one_of([s for s in name_strats if not s.is_empty])

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
                for k in set(dep_names).intersection(out):  # pragma: no cover
                    # nocover because some of these conditionals are rare enough
                    # that not all test runs hit them, but are still essential.
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
                out[key] = draw(
                    merged_as_strategies(
                        pattern_schemas,
                        alphabet=alphabet,
                        custom_formats=custom_formats,
                    )
                )
            else:
                out[key] = draw(
                    __from_schema(
                        additional, custom_formats=custom_formats, alphabet=alphabet
                    )
                )

            for k, v in dep_schemas.items():
                if k in out and not make_validator(v).is_valid(out):
                    out.pop(key)
                    elements.reject()

        for k in set(dep_names).intersection(out):
            assume(set(out).issuperset(dep_names[k]))
        return out

    return from_object_schema()
