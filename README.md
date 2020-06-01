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


## Contributing to `hypothesis-jsonschema`

We love external contributions - and try to make them both easy and fun.
You can [read more details in our contributing guide](https://github.com/Zac-HD/hypothesis-jsonschema/blob/master/CONTRIBUTING.md),
and [see everyone who has contributed on GitHub](https://github.com/Zac-HD/hypothesis-jsonschema/graphs/contributors).
Thanks, everyone!


### Changelog

Patch notes [can be found in `CHANGELOG.md`](https://github.com/Zac-HD/hypothesis-jsonschema/blob/master/CHANGELOG.md).


### Security contact information
To report a security vulnerability, please use the
[Tidelift security contact](https://tidelift.com/security).
Tidelift will coordinate the fix and disclosure.
