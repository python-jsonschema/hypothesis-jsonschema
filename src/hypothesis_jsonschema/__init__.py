"""A Hypothesis extension for JSON schemata.

The public API contains only two functions: `from_schema` and `json_schemata`.
Check their docstrings for details.
"""

__version__ = "0.4.1"
__all__ = ["from_schema", "json_schemata"]

from ._impl import from_schema, json_schemata
