# Makefile for Crypto Deep Hedging project

.PHONY: help tests test-verbose test-quiet test-coverage clean install download-data

# Default target
help:
	@echo "Available targets:"
	@echo "  make tests          - Run all unit tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make test-quiet     - Run tests with minimal output"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make test-specific  - Run specific test file (use TEST=filename)"
	@echo "  make clean          - Remove cache and temporary files"
	@echo "  make install        - Install dependencies"
	@echo "  make download-data  - Download sample historical data"

# Run all tests with standard output
tests:
	@echo "Running all tests..."
	@cd tests && python -m pytest -v --tb=short

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	@cd tests && python -m pytest -vv --tb=long

# Run tests with minimal output
test-quiet:
	@echo "Running tests quietly..."
	@cd tests && python -m pytest -q --tb=no

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	@cd tests && python -m pytest --cov=../crypto --cov-report=term-missing

# Run specific test file
# Usage: make test-specific TEST=test_bitcoin_perpetual_models.py
test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=filename.py"; \
		exit 1; \
	fi
	@echo "Running $(TEST)..."
	@cd tests && python -m pytest $(TEST) -v

# Clean up cache and temporary files
clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".coverage" -delete
	@echo "Clean complete"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install torch pandas numpy pyarrow pytest pytest-cov
	@echo "Installation complete"

# Download sample historical data
download-data:
	@echo "Downloading sample historical data..."
	@cd data && python download_historical.py --sample
	@echo "Download complete"

# Run only Bitcoin instrument tests
test-instruments:
	@echo "Running Bitcoin instrument tests..."
	@cd tests && python -m pytest test_bitcoin_instruments.py test_bitcoin_perpetual_models.py -v

# Run only data pipeline tests
test-data:
	@echo "Running data pipeline tests..."
	@cd tests && python -m pytest test_deribit_client.py test_downloader.py test_data_loader.py -v

# Check for test failures
test-check:
	@cd tests && python -m pytest --tb=no -q && echo "✅ All tests passed!" || echo "❌ Some tests failed"

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	@echo "Running tests in parallel..."
	@cd tests && python -m pytest -n auto

# Generate HTML test report (requires pytest-html)
test-report:
	@echo "Generating HTML test report..."
	@cd tests && python -m pytest --html=test_report.html --self-contained-html
	@echo "Report generated: tests/test_report.html"