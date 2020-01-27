"""
Fetch lots of real-world schemata to test against.

Uses https://github.com/json-schema-org/JSON-Schema-Test-Suite for big and
complex schemata, and http://schemastore.org/json/ for smaller schemata
that - as the official test suite - should cover each feature in the spec.
"""

import concurrent.futures
import io
import json
import urllib.request
import zipfile
from typing import Any


def get_json(url: str) -> Any:
    """Fetch the json payload at the given url."""
    assert url.startswith("http://") or url.startswith("https://")
    with urllib.request.urlopen(url) as handle:
        return json.load(handle)


# Load cached schemas, so we cope with flaky connections and keep deleted entries
try:
    with open("tests/corpus-schemastore-catalog.json") as f:
        schemata = json.load(f)
except FileNotFoundError:
    schemata = {}

# Download all the examples known to schemastore.org, concurrently!
with concurrent.futures.ThreadPoolExecutor() as ex:
    futures = []

    def add_future(name: str, url: str) -> None:
        futures.append(ex.submit(lambda n, u: (n, get_json(u)), name, url))

    for s in get_json("http://schemastore.org/api/json/catalog.json")["schemas"]:
        if "versions" in s:
            for version, link in s["versions"].items():
                add_future(f"{s['description']}-{version}", link)
        else:
            add_future(s["description"], s["url"])
    for future in concurrent.futures.as_completed(futures, timeout=30):
        try:
            name, schema = future.result()
        except Exception as e:
            print(f"Error: {name!r} ({e})")  # noqa: T001
        else:
            schemata[name] = schema


# Dump them all back to the catalog file.
with open("tests/corpus-schemastore-catalog.json", mode="w") as f:
    json.dump(schemata, f, indent=4, sort_keys=True)


# Part two: fetch the official jsonschema compatibility test suite
suite: dict = {}
invalid_suite: dict = {}

with urllib.request.urlopen(
    "https://github.com/json-schema-org/JSON-Schema-Test-Suite/archive/master.zip"
) as handle:
    start = "JSON-Schema-Test-Suite-master/tests/"
    with zipfile.ZipFile(io.BytesIO(handle.read())) as zf:
        seen = set()
        for path in zf.namelist():
            if path.startswith(start + "draft7/") and path.endswith(".json"):
                for v in json.load(zf.open(path)):
                    if any(t["valid"] for t in v["tests"]):
                        suite["draft7/" + v["description"]] = v["schema"]
                        seen.add(json.dumps(v["schema"], sort_keys=True))
                    elif "/optional/" not in path:
                        invalid_suite["draft7/" + v["description"]] = v["schema"]
                        seen.add(json.dumps(v["schema"], sort_keys=True))
        for path in zf.namelist():
            if path.startswith(start + "draft4/") and path.endswith(".json"):
                for v in json.load(zf.open(path)):
                    if json.dumps(v["schema"], sort_keys=True) in seen:
                        # No point testing an exact duplicate schema, so skip this one
                        continue
                    elif any(t["valid"] for t in v["tests"]):
                        suite["draft4/" + v["description"]] = v["schema"]
                    elif "/optional/" not in path:
                        invalid_suite["draft4/" + v["description"]] = v["schema"]

with open("tests/corpus-suite-schemas.json", mode="w") as f:
    json.dump([suite, invalid_suite], f, indent=4, sort_keys=True)


# Part three: canonicalise tricky schemas reported on the issue tracker
with open("tests/corpus-reported.json") as f:
    schemata = json.load(f)
with open("tests/corpus-reported.json", mode="w") as f:
    json.dump(schemata, f, indent=4, sort_keys=True)
