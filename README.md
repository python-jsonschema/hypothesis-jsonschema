# hypothesis-jsonschema

A [Hypothesis](https://hypothesis.readthedocs.io) strategy for generating data
that matches some [JSON schema](https://json-schema.org/).
It is currently in early alpha, but you can use it if you want.
[Here's the PyPI page.](https://pypi.org/project/hypothesis-jsonschema/)

The public API consists of a just two functions:


### hypothesis_jsonschema.from_schema
Takes a JSON schema and return a strategy for allowed JSON objects.

This strategy supports almost all of the schema elements described in the
draft RFC as of November 2018 (draft 7), with the following exceptions:

- For objects, the "dependencies" keyword is not supported.
- Subschemata are not supported, i.e. the "if", "then", and "else" keywords,
  and the "allOf, "anyOf", "oneOf", and "not" keywords.
- Schema reuse with "definitions" is not supported.
- string-encoding of non-JSON data is not supported.
- schema annotations, i.e. "title", "description", "default",
  "readOnly", "writeOnly", and "examples" are not supported.
- JSON pointers are not supported.


### hypothesis_jsonschema.json_schemata
A Hypothesis strategy for arbitrary JSON schemata.

This strategy may generate anything that can be handled by `from_schema`,
and is used to provide full branch coverage when testing this package.


## Supported versions

`hypothesis-jsonschema` does not support Python 2, because
[it's close to end of life](https://pythonclock.org/) and Python 3.6+ is a
much nicer language.  Contact me if you would like this changed and are
willing to either pay for or do the work to support Python 2.

In general, 0.x versions will require very recent versions of all dependencies
because I don't want to deal with compatibility workarounds.

`hypothesis-jsonschema` may make backwards-incompatible changes at any time
before version 1.x - that's what semver means! - but I've kept the API surface
small enough that this should be avoidable.  The main source of breaks will be
if or when schema that never really worked turn into explicit errors instead
of generating values that don't quite match.


### Changelog:

#### 0.4.2 - 2019-02-14
- Dropped dependency on `canonicaljson`
- Less warnings on Python 3.7

#### 0.4.1 - 2019-02-06
- Relicensed under the more permissive Mozilla Public License, like Hypothesis
- Requires Hypothesis version 4.0 or later
- Fixed an array bounds bug with `maxItems` and `contains` keywords

#### 0.4.0 - 2018-11-25
Supports string formats (email, datetime, etc) and simple use of the
`"contains"` keyword for arrays.

#### 0.3.0 - 2018-11-25
Good support for all basic types.  MVP.

#### 0.2.0 - 2018-11-24
Inference for null, boolean, string, and numeric types.

#### 0.1.0 - 2018-11-21
Stake in the ground (generate arbitrary JSON and filter it!)
