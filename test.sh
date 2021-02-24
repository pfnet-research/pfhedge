#!/bin/sh -eu

python3 -m pytest --doctest-modules pfhedge
python3 -m pytest --doctest-modules tests

python3 -m flake8 pfhedge
python3 -m black --check --quiet pfhedge || read -p "Run black? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m black --quiet pfhedge
python3 -m isort --check --force-single-line-imports pfhedge || read -p "Run isort? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports --quiet pfhedge
python3 -m black --quiet tests
python3 -m isort --force-single-line-imports --quiet tests
