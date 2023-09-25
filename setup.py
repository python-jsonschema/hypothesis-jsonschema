import os
import pathlib

import setuptools


def local_file(name: str) -> str:
    """Interpret filename as relative to this file."""
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))


SOURCE = local_file("src")
README = local_file("README.md")

with open(local_file("src/hypothesis_jsonschema/__init__.py")) as o:
    for line in o:
        if line.startswith("__version__"):
            _, __version__, _ = line.split('"')


setuptools.setup(
    name="hypothesis-jsonschema",
    version=__version__,
    author="Zac Hatfield-Dodds",
    author_email="zac@zhd.dev",
    packages=setuptools.find_packages(SOURCE),
    package_dir={"": SOURCE},
    package_data={"": ["py.typed"]},
    url="https://github.com/Zac-HD/hypothesis-jsonschema",
    project_urls={"Funding": "https://github.com/sponsors/Zac-HD"},
    license="MPL 2.0",
    description="Generate test data from JSON schemata with Hypothesis",
    zip_safe=False,
    install_requires=["hypothesis>=6.84.3", "jsonschema>=4.18.0"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Hypothesis",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education :: Testing",
        "Topic :: Software Development :: Testing",
        "Typing :: Typed",
    ],
    long_description=pathlib.Path(README).read_text(),
    long_description_content_type="text/markdown",
    keywords="python testing fuzzing property-based-testing json-schema",
)
