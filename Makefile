PROJECT_NAME := pfhedge

.PHONY: check
check: test lint mypy

.PHONY: install
install:
	@poetry install

.PHONY: test
test: doctest pytest

.PHONY: doctest
doctest:
	@poetry run pytest --doctest-modules $(PROJECT_NAME)

.PHONY: pytest
pytest:
	@poetry run pytest --doctest-modules tests

.PHONY: test-cov
test-cov:
	@poetry run pytest --cov=$(PROJECT_NAME) --cov-report=html

.PHONY: lint
lint: lint-black lint-isort

.PHONY: lint-black
lint-black:
	@poetry run black --check --diff --quiet --skip-magic-trailing-comma .

.PHONY: lint-isort
lint-isort:
	@poetry run isort --check --force-single-line-imports --quiet .

.PHONY: mypy
mypy:
	@poetry run mypy $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	@poetry run black --quiet --skip-magic-trailing-comma .

.PHONY: format-isort
format-isort:
	@poetry run isort --force-single-line-imports --quiet .

.PHONY: doc
doc:
	@cd docs && make html

.PHONY: publish
publish:
	git checkout main
	gh repo sync simaki/$(PROJECT_NAME)
	gh workflow run publish.yml --repo simaki/$(PROJECT_NAME)
