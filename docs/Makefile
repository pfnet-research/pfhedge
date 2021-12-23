# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD   ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build
RUN           = poetry run

# Put it first so that "make" without argument is like "make help".
help:
	$(RUN) $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(RUN) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	echo "Removing everything under 'build' and 'source/generated'.."
	rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated

livehtml:
	$(RUN) $(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) --open-browser $(O)
