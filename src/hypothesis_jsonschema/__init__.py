"""A Hypothesis extension for JSON schemata.

The only public API is `from_schema`; check the docstring for details.
"""

__version__ = "0.14.0"
__all__ = ["from_schema"]

from ._from_schema import from_schema
