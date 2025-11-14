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

.PHONY: test-crypto
test-crypto:
	cd crypto/tests && python -m pytest -v --tb=short

.PHONY: test-crypto-quiet
test-crypto-quiet:
	cd crypto/tests && python -m pytest -q --tb=no

.PHONY: test-crypto-instruments
test-crypto-instruments:
	cd crypto/tests && python -m pytest test_bitcoin_instruments.py test_bitcoin_perpetual_models.py -v

.PHONY: test-crypto-data
test-crypto-data:
	cd crypto/tests && python -m pytest test_deribit_client.py test_downloader.py test_data_loader.py -v

.PHONY: test-all
test-all: test test-crypto

.PHONY: tests
tests: test-crypto

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml -m "not gpu"

.PHONY: lint
lint: lint-black lint-isort flake8 mypy

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
	$(RUN) pflake8 $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet .

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

.PHONY: clean-crypto
clean-crypto:
	find crypto -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find crypto -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find crypto -type f -name "*.pyc" -delete
