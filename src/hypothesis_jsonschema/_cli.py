import json
import sys
import typer
from importlib.metadata import version as vers
from typing import Optional, Annotated

from hypothesis_jsonschema._from_schema import from_schema

app = typer.Typer(help="Hypothesis JSONSchema CLI")

def version_callback(value: bool) -> None:
    """Callback to show the version of memtab"""
    if value:
        print(vers("hypothesis-jsonschema"))
        raise typer.Exit()

@app.command()
def main(
    num: Annotated[Optional[int], typer.Option(help="number of examples")]=100,
    seed: Annotated[Optional[int], typer.Option(help="Seed for random number generator")]=0,
    script: Annotated[Optional[str], typer.Option(help="Output script name")]=None,
    schema: Annotated[
        typer.FileText, typer.Option(help="the schema file to read. will read from stdin if not specified")
    ] = None,
    version: Annotated[
        Optional[bool], typer.Option(help="Show the version of memtab", callback=version_callback, is_eager=True)
    ] = None,):
    
    
    if schema:
        schema_data = schema.read()
    else:
        schema_data_via_stdin = sys.stdin
        if (
            hasattr(schema_data_via_stdin, "isatty") and schema_data_via_stdin.isatty()
        ):  # Check if stdin is connected to a terminal
            raise ValueError("No input provided via stdin.")
        schema_data = schema_data_via_stdin.read()
    if not schema_data or not len(schema_data):
        raise ValueError("schema is empty")
    
    sample = from_schema(json.loads(schema_data))
    
    for _ in range(num):
        ex = sample.example()
    typer.Exit(0)
    
