#!/bin/bash
set -uex

apt update
apt install -y git curl wget gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev zlib1g-dev liblzma-dev

git clone https://github.com/pyenv/pyenv.git /opt/pyenv
export PYENV_ROOT=/opt/pyenv
export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# Install Python.
python_version=${python_version:-3.9.7}
echo "python ${python_version}"
pyenv install ${python_version}
pyenv global ${python_version}
eval "$(pyenv init --path)"

curl -sSL https://install.python-poetry.org | python -
export PATH="${HOME}/.local/bin:${PATH}"

poetry install

# generate python single files from notebook files
mkdir -p examples_tests/examples/
ls examples/*.ipynb | xargs -I {} bash -c "poetry run jupyter nbconvert --to python {} --stdout > examples_tests/{}.py"

poetry run pip uninstall -y torch
poetry run pip install --no-cache-dir "torch>=2.0.0,<3"

# run normal test using gpu w/ pytorch 2.0
poetry run pytest -m gpu --cov-report=html --cov pfhedge .
mv htmlcov /output/htmlcov

# run notebook test using gpu w/ pytorch 2.0
cd examples_tests/examples
find . -name '*.py' | xargs -I {} poetry run python {};
cd -

poetry run pip uninstall -y torch
poetry run pip install --no-cache-dir "torch<2.0.0"

# run normal test using gpu w/ pytorch 1.x
poetry run pytest -m gpu pfhedge .

# run notebook test using gpu w/ pytorch 1.x
cd examples_tests/examples
find . -name '*.py' -exec poetry run python {} \;
cd -
