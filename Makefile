PROJECT_NAME := pfhedge
RUN := poetry run

.PHONY: check
check: test lint-pysen

.PHONY: install
install:
	@poetry install

.PHONY: test
test: doctest pytest

.PHONY: doctest
doctest:
	$(RUN) pytest --doctest-modules $(PROJECT_NAME) -m "not gpu"

.PHONY: pytest
pytest:
	$(RUN) pytest --doctest-modules tests -m "not gpu"

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml -m "not gpu"

.PHONY: lint
lint: lint-pysen

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
format: format-pysen

.PHONY: format-black
format-black:
	$(RUN) black --quiet --skip-magic-trailing-comma .

.PHONY: format-isort
format-isort:
	$(RUN) isort --force-single-line-imports --quiet .

.PHONY: format-pysen
format-pysen:
	$(RUN) pysen run format

.PHONY: lint-pysen
lint-pysen:
	$(RUN) pysen run lint

.PHONY: doc
doc:
	@cd docs && make html

.PHONY: publish
publish:
	@git checkout main
	@gh repo sync simaki/$(PROJECT_NAME)
	@gh workflow run publish.yml --repo simaki/$(PROJECT_NAME)
