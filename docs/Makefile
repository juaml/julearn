# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXBUILD   = sphinx-build
POLYBUILD     = sphinx-polyversion
SOURCEDIR     = .
BUILDDIR      ?= docs/_build
SPHINXOPTS    ?= -E -v


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) --help "$(SOURCEDIR)" "$(BUILDDIR)" $(O)

.PHONY: help Makefile clean cleanall local _build_local _html

clean:
	rm -rf $(BUILDDIR)/main
	rm -rf auto_examples/
	rm -rf api/generated/

cleanall: clean
	rm -rf $(BUILDDIR)/*

# For local build
local:
	make _build_local; STATUS=$$?; exit $$STATUS

_build_local:
# 	$(POLYBUILD) ./poly.py -vv -l -o NO_PATCH=True -o OUTPUT_DIR="$(BUILDDIR)" -o SOURCE_DIR="$(SOURCEDIR)" -o TAG_REGEX="^$$"
	$(SPHINXBUILD) "$(SOURCEDIR)" _build/dev $(SPHINXOPTS) $(O)

html:
	make _html; STATUS=$$?; exit $$STATUS

_html:
	$(POLYBUILD) ./poly.py -vv -o OUTPUT_DIR="$(BUILDDIR)" -o SOURCE_DIR="$(SOURCEDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
