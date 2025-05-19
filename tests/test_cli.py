import json
from io import StringIO

import pytest
from typer.testing import CliRunner, Result

from hypothesis_jsonschema._cli import app


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
    ],
)
def test_cli_runner(schema):
    stdin_mock = StringIO(json.dumps(schema))
    stdin_mock.seek(0)  # Ensure the StringIO object is at the beginning
    runner = CliRunner()
    result: Result = runner.invoke(app, input=stdin_mock.getvalue(), catch_exceptions=False)
    assert result.exit_code == 0, f"CLI failed with exit code {result.exit_code}. Output: {result.stdout}"
