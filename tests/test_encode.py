"""Tests for the hypothesis-jsonschema library."""

import json

from hypothesis import given

from hypothesis_jsonschema._encode import encode_canonical_json
from hypothesis_jsonschema._from_schema import JSON_STRATEGY


@given(JSON_STRATEGY)
def test_canonical_json_encoding(v):
    """Test our hand-rolled canonicaljson implementation."""
    encoded = encode_canonical_json(v)
    v2 = json.loads(encoded)
    assert v == v2
    assert encode_canonical_json(v2) == encoded
