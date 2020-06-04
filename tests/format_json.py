"""Ensure that our JSON files are always formatted and sorted."""

import json
import pathlib

p = pathlib.Path(__file__).parent / "corpus-reported.json"
p.write_text(json.dumps(json.loads(p.read_text()), indent=4, sort_keys=True))
