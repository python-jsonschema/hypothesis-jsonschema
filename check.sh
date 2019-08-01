autoflake --recursive --in-place --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables .
pyupgrade --py36-plus $(find . -name '*.py')
isort --recursive --apply .
black .
flake8
mypy . src/
pytest
