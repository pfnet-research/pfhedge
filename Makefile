PROJECT_NAME := pfhedge
RUN := poetry run

.PHONY: check
check: test lint mypy flake8

.PHONY: install
install:
	@poetry install

.PHONY: test
test: doctest pytest

.PHONY: doctest
doctest:
	$(RUN) pytest --doctest-modules $(PROJECT_NAME)

.PHONY: pytest
pytest:
	$(RUN) pytest --doctest-modules tests

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml

.PHONY: lint
lint: lint-black lint-isort

.PHONY: lint-black
lint-black:
	$(RUN) black --check --diff --quiet --skip-magic-trailing-comma .

.PHONY: lint-isort
lint-isort:
	$(RUN) isort --check --force-single-line-imports --quiet .

.PHONY: mypy
mypy:
	$(RUN) mypy $(PROJECT_NAME)

.PHONY: flake8
flake8:
	$(RUN) flake8 $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet --skip-magic-trailing-comma .

.PHONY: format-isort
format-isort:
	$(RUN) isort --force-single-line-imports --quiet .

.PHONY: doc
doc:
	@cd docs && make html

.PHONY: publish
publish:
	@git checkout main
	@gh repo sync simaki/$(PROJECT_NAME)
	@gh workflow run publish.yml --repo simaki/$(PROJECT_NAME)
