"""It's a setup.py"""

import os

import setuptools


def local_file(name: str) -> str:
    """Interpret filename as relative to this file."""
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))


SOURCE = local_file("src")
README = local_file("README.md")

with open(local_file("src/hypothesis_jsonschema.py")) as o:
    __version__ = None
    exec(o.read())  # pylint: disable=exec-used
    assert __version__ is not None


setuptools.setup(
    name="hypothesis-jsonschema",
    version=__version__,
    author="Zac Hatfield-Dodds",
    author_email="zac.hatfield.dodds@gmail.com",
    packages=setuptools.find_packages(SOURCE),
    package_dir={"": SOURCE},
    package_data={"": ["py.typed"]},
    url="https://github.com/Zac-HD/hypothesis-jsonschema",
    license="AGPLv3+",
    description="Generate test data from JSON schemata with Hypothesis",
    zip_safe=False,
    install_requires=["hypothesis>=3.82.1", "jsonschema>=2.6.0"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Hypothesis",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Testing",
    ],
    long_description=open(README).read(),
    long_description_content_type="text/markdown",
    keywords="python testing fuzzing property-based-testing json-schema",
)
