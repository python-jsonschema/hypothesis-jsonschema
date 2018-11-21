# hypothesis-jsonschema

A [Hypothesis](https://hypothesis.readthedocs.io) strategy for generating data
that matches some [JSON schema](https://json-schema.org/).
It is currently in early alpha, but you can use it if you want.

The public API consists of a just two functions:

1. `hypothesis_jsonschema.from_schema` takes a JSON schema and returns a
   Hypothesis strategy which generates objects that match the schema.
2. `hypothesis_jsonschema.json_schemata` returns a Hypothesis strategy
   which generates arbitrary JSON schemata.

`hypothesis-jsonschema` does not support Python 2, because
[it's close to end of life](https://pythonclock.org/) and Python 3.6+ is a
much nicer language.  Contact me if you would like this changed and are
willing to either pay for or do the work to support Python 2.
