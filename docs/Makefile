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
local:
	make _build_local; STATUS=$$?; exit $$STATUS

## First build will create warnings for auto_examples. Second one will work correctly.
_build_local:
	sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" "-E" $(O)
	sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	make _html; STATUS=$$?; exit $$STATUS


## Build only tags that are not already built, and ignore dev tags. This is useful for CI to avoid building all tags on every push, but still build new tags when they are created.
_html:
	@regex=$$( \
		for tag in $$(git tag -l); do \
			if [ ! -d "$(BUILDDIR)/$$tag" ]; then \
				echo "$$tag"; \
			fi; \
		done | \
		grep -v dev | \
		paste -sd'|' - \
	); \
	echo "$$regex"; \
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -D "smv_tag_whitelist='^($$regex)$$'"  $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
