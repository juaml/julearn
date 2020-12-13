# Makefile before PR
#

.PHONY: checks

checks: flake spellcheck

flake:
	flake8

spellcheck:
	codespell julearn/ docs/ examples/

test:
	pytest -v