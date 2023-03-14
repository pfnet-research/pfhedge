#!/bin/bash
set -uex

git clone https://github.com/pyenv/pyenv.git /opt/pyenv
PYENV_ROOT=/opt/pyenv
PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

# Install Python.
if [ -z $python_version ]; then
  python_version="3.9.7"
fi
echo "python ${python_version}"
pyenv install ${python_version} && \
pyenv global ${python_version}

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
PATH=${HOME}/.poetry/env:${PATH}

poetry install

pip list

poetry run pytest -m gpu --cov-report=html --cov pfhedge .

mv htmlcov /output/htmlcov
