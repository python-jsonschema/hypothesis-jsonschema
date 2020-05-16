# hypothesis-jsonschema

A [Hypothesis](https://hypothesis.readthedocs.io) strategy for generating data
that matches some [JSON schema](https://json-schema.org/).

[Here's the PyPI page.](https://pypi.org/project/hypothesis-jsonschema/)

## API

The public API consists of just one function: `hypothesis_jsonschema.from_schema`,
which takes a JSON schema and returns a strategy for allowed JSON objects.

JSONSchema drafts 04, 05, and 07 are fully tested and working.
As of version 0.11, this includes resolving non-recursive references!

For details on how to use this strategy in your tests,
[see the Hypothesis docs](https://hypothesis.readthedocs.io/).


## Supported versions

`hypothesis-jsonschema` requires Python 3.6 or later.
In general, 0.x versions will require very recent versions of all dependencies
because I don't want to deal with compatibility workarounds.

`hypothesis-jsonschema` may make backwards-incompatible changes at any time
before version 1.x - that's what semver means! - but I've kept the API surface
small enough that this should be avoidable.  The main source of breaks will be
if or when schema that never really worked turn into explicit errors instead
of generating values that don't quite match.

You can [sponsor me](https://github.com/sponsors/Zac-HD) to get priority
support, roadmap input, and prioritized feature development.


### Changelog:

- Improved canonicalisation of `uniqueItems: false` case
- Reuse `jsonschema` validators during canonicalisation (performance improvement)

#### 0.12.1 - 2020-04-14
- Added a strategy for the `"color"` format
- Only apply string length filter when needed (small performance improvement)

#### 0.12.0 - 2020-04-08
- Fixed error in resolution of certain `$ref`\ s
- Improved canonicalisation of `anyOf` and `contains` keys

#### 0.11.1 - 2020-01-27
- Requires Hypothesis >= 5.3.0, for improved IP address strategies
- Better canoncialisation of array schemata

#### 0.11.0 - 2020-01-26
- Resolve local, non-recursive references via the `$ref` keyword.

This is the largest feature in a while, and for some schemata it is a breaking
change.  It's also a fundamental part of the spec, so I'm OK with that!

Note that `hypothesis-jsonschema` will raise an explicit error rather than fetching
a remote resource via URI.  If think your tests *really should* hit the network,
get in touch and we can discuss adding an off-by-default option for this.

#### 0.10.3 - 2020-01-26
- Improved canonicalisation of conflicting `minProperties` and `maxProperties`
- Explictly reject draft-03 schemata, which are not supported
- More specific type annotations for `from_schema`
- Better performance for certain object schemata

#### 0.10.2 - 2020-01-09
- `enum` schema now shrink to a minimal example rather than the first value listed.
  This also makes the internals more efficient in certain rare cases.
- Improved handling of bounded numeric sub-schemas

#### 0.10.1 - 2019-12-28
- Improved handling of non-integer numeric schemas with `multipleOf`
- Improved handling of `not` in many common cases
- Improved handling of object schemas with `dependencies` on required keys
- Fixed cases where `propertyNames` bans a `required` key

#### 0.10.0 - 2019-12-26
- Improved handling of numeric schemas, especially integer schemas with `multipleOf`.
- Bump the minimum version of Hypothesis to ensure that all users have the unique
  array bugfix from `0.9.12`.
- Fixed a bug where array schemas with an array of `items`, `additionalItems: false`,
  and a `maxItems` larger than the number of allowed items would resolve to an
  invalid strategy.

#### 0.9.13 - 2019-12-18
- Improved internal handling of schemas for arrays which must always be length-zero.

#### 0.9.12 - 2019-12-01
- Fixed RFC 3339 strings generation.  Thanks to Dmitry Dygalo for the patch!
- Fixed a bug where equal floats and ints could be generated in a unique array,
  even though JSONSchema considers 0 === 0.0
  (though this may also require an upstream fix to work...)

#### 0.9.11 - 2019-11-30
- Fixed a bug where objects which could have either zero or one
  properties would always be generated with zero.

#### 0.9.10 - 2019-11-27
- Updated project metadata and development tooling
- Supported and tested on Python 3.8

#### 0.9.9 - 2019-10-02
- Correct handling of `{"items": [...], "uniqueItems": true"}` schemas

#### 0.9.8 - 2019-08-24
- Corrected handling of the `"format"` keyword with unknown values - custom values
  are allowed by the spec and should be treated as annotations (i.e. ignored).

#### 0.9.7 - 2019-08-15
- Improved canonicalisation, especially for deeply nested schemas.

#### 0.9.6 - 2019-08-02
- A performance optimisation for null and boolean schema,
  which relies on a bugfix in `jsonschema >= 3.0.2`.

#### 0.9.5 - 2019-08-02
- Improved handling of the `contains` keyword for arrays

#### 0.9.4 - 2019-07-01
- Improved canonicalisation and merging for a wide range of schemas,
  which as usual unlocks significant optimisations and performance
  improvements for cases where they apply.

#### 0.9.3 - 2019-06-13
- Future-proofed canonicalisation of `type` key.

#### 0.9.2 - 2019-05-23
- Better internal canonicalization, which makes current and future
  optimisations more widely applicable.
- Yet another fix, this time for negative zero and numeric bouds as floats
  with sub-integer precision.  IEEE 754 is *tricky*, even with Hypothesis!
- Fixes handling of `enum` with elements disallowed by base schema,
  handling of `if-then-else` with a base schema, and handling of regex
  patterns that are invalid in Python.

#### 0.9.1 - 2019-05-22
- Fix the fix for numeric schemas with `multipleOf` and exclusive bounds.

#### 0.9.0 - 2019-05-21
- Supports merging schemas for overlapping `patternProperties`,
  a significant performance improvement in most cases.
- If the `"type"` key is missing, it is now inferred from other keys
  rather than always defaulting to `"object"`.
- Fixed handling of complicated numeric bounds.

#### 0.8.2 - 2019-05-21
- Improve performance for object schemas where the min and max size can be
  further constrained from `properties` and `propertyNames` attributes.

#### 0.8.1 - 2019-03-24
- Supports draft-04 schemata with the latest version of `jsonschema`

#### 0.8.0 - 2019-03-23
- Further improved support for `allOf`, `oneOf`, and `anyOf` with base schemata
- Added support for `dependencies`
- Handles overlapping `patternProperties`

#### 0.7.0 - 2019-03-21
- Now requires `jsonschema` >= 3.0
- Improved support for `allOf`, `oneOf`, and `propertyNames`
- Supports schemata with `"type": [an array of types]`
- Warning-free on Hypothesis 4.11

#### 0.6.1 - 2019-02-23
- Fix continuous delivery configuration (*before* the latent bug manifested)

#### 0.6.0 - 2019-02-23
- Support for conditional subschemata, i.e. the `if`, `then`, `else` keywords,
  and the `anyOf`, `allOf`, `oneOf`, and `not` keywords.

#### 0.5.0 - 2019-02-22
- Works with `jsonschema` 3.0 pre-release
- Initial support for draft06 and draft07

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


### Security contact information
To report a security vulnerability, please use the
[Tidelift security contact](https://tidelift.com/security).
Tidelift will coordinate the fix and disclosure.
