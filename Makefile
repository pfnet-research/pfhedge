PROJECT_NAME := pfhedge

.PHONY: install
install:
	@poetry install

.PHONY: test
test:
	@poetry run pytest --doctest-modules $(PROJECT_NAME)
	@poetry run pytest --doctest-modules tests

.PHONY: lint
lint:
	@poetry run black --check --quiet .
	@poetry run isort --check --force-single-line-imports --quiet .

.PHONY: format
format:
	@poetry run black --quiet .
	@poetry run isort --force-single-line-imports --quiet .
