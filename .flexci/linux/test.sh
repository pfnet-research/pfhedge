#!/bin/bash
set -uex
poetry install

pip list

poetry run pytest -m gpu --cov-report=html --cov pfhedge .

mv htmlcov /output/htmlcov
