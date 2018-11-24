"""A Hypothesis extension for JSON schemata."""

__version__ = "0.2.0"
__all__ = ["from_schema", "json_schemata"]

from ._impl import from_schema, json_schemata
