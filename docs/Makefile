# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?="-W" "-E"
SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = .
BUILDDIR      ?= _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) --help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean local news _build_local _html

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples/
	rm -rf api/generated/

# For local build
local: news
	make _build_local; STATUS=$$?; make news_cleanup; exit $$STATUS

news:
	@echo "Generating whats_new.rst"
	cd ../ && towncrier build --keep

news_cleanup:
	@echo "Cleaning up whats_new.rst"
	git restore --staged whats_new.rst
	git restore whats_new.rst

_build_local:
	sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: news
	make _html; STATUS=$$?; make news_cleanup; exit $$STATUS

_html:
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
