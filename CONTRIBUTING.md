# Contributing to `hypothesis-jsonschema`

We love external contributions - and try to make them both easy and fun.
This isn't "patches welcome", it's "we'll help you write the patch".


## Walking through your first contribution

We use [Github Flow](https://guides.github.com/introduction/flow/index.html),
so here's the workflow for a new contributor:

1. Choose an issue to work on - preferably one of the
   [issues tagged `good first issue`](https://github.com/Zac-HD/hypothesis-jsonschema/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

2. Fork the repo, clone it to your local machine, and create your branch from `master`.
   (If this sounds scary, [check out GitHub's *Hello World* guide](https://guides.github.com/activities/hello-world/))

3. Check that everything is working by running the tests.  We
   [manage our tests and dependencies with `tox`](https://tox.readthedocs.io/en/latest/index.html),
   so after you `pip install tox`, you can just type `tox` to install and run
   everything else.  Later, you can run `tox -e check` if you just want to run
   the linters and auto-formatting tools without waiting for all the tests.

4. Add your new feature!  Add a test for your new feature!
   (and keep running `tox` to make sure nothing else broke).

5. When you're ready, push your changes to GitHub and open a pull request!
   Note that you *don't need to be finished* - if you're stuck, please open a
   work-in-progress PR and ask for suggestions.

6. Zac will review your code, and either ask for some changes (go to step 4),
   or merge if it looks finished and passes all the tests.  Congratulations!


## Useful background reading

As you might guess, `hypothesis-jsonschema` makes a lot more sense if you know a
bit about Hypothesis and a bit about JSONschema!

- [*Understanding JSONschema*](https://json-schema.org/understanding-json-schema/)
  is the official reference, and a lot easier to read than the specification
- [*In Praise of Property-Based Testing*](https://increment.com/testing/in-praise-of-property-based-testing/)
  is a great explanation of what Hypothesis is for and why it works so well.
- The [official Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/)
  covers all the functions you'll need, and you can get some hands-on experience with
  [the problems in this tutorial](https://github.com/Zac-HD/escape-from-automanual-testing/).


## How things work

### About the code

`_from_schema.py` is responsible for translating JSON schemas into Hypothesis
strategies.  It consists mostly of an internal `from_XXX_schema()` function
for each type (numbers, strings, lists, and dictionaries), and a few general
helper functions.

The catch is that the same notional set of values can be matched by many
different schemas, and there can be tricky interactions between different
parts of a schema that make the obvious translation very inefficient.

`_canonicalise.py` therefore tries to rewrite schemas into standard format which
can be efficiently translated to strategies.  `canonicalish` doesn't standardise
*every* case, but it's still worthwhile and the more we cover the better our
`from_schema()` logic works.  The most significant of the various helper functions
is `merged`, which calculates the intersection of schemas - so that e.g. an `allOf`
can be translated into one strategy, instead of a filter-heavy `st.one_of()`
strategy.


### Code style and documentation

We have a fairly strict code style, but *absolutely no need for opinions*.
`tox -e check` fixes as many style issues as possible, using `black` among
other tools, and then runs `flake8` and `mypy` to detect any remaining style
issues.  Once the tools are happy, we're happy.

This highly automated approach makes it easy to focus on design issues like
"how should I break this feature up into functions" or "does this name help
readers understand the code" rather than style issues like formatting.

The public API of this package consists of a single function, `from_schema()`,
which is explained in the README.  We therefore have very little user-facing
documentation, but encourage the use of comments for the benefit of contributors
who are reading the code and need to understand implementation details.


### About the tests

The layout of our tests matches that of our code: we have `test_from_schema.py`
and `test_canonicalish.py`, which... well, you can guess what they test!
`test_version.py` checks that the package version matches the changelog, and
that the changelog is in order without any missing versions.

The main trick is that as well as some standard hand-written or 'example based'
tests, `hypothesis-jsonschema` relies very heavily on property-based tests and
on data-driven tests using a wide variety of schemas.

We have a custom Hypothesis strategy to generate *schemas* of various kinds
in `gen_schemas.py`. We also use all the schemas from the
[upstream test suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
and (almost) all of [schemastore.org](http://schemastore.org/json/).
Plus schemas from bug reports, or randomly-generated schemas which found bugs...

This approach means that we have relatively few, mostly high-level tests -
and an incredibly powerful test suite.  You can try out wild ideas, and be
confident that if the tests pass, the code really does work!
