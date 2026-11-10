export PIPENV_VERBOSITY := -1

.PHONY: help install test cov typecheck check

help:
	@echo "make install    install the packages, including the development tools"
	@echo "make test       run the tests"
	@echo "make cov        run the tests with a coverage report"
	@echo "make typecheck  run pyright"
	@echo "make check      run the type checks and the tests"

install:
	pipenv install --dev

test:
	pipenv run pytest -q

cov:
	pipenv run pytest -q --cov=. --cov-report=term-missing

typecheck:
	pipenv run pyright

check: typecheck test
