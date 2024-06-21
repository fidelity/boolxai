# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Copy the example notebook and related files to _examples/ so that they can be
# processed by Sphinx, and clean up. Also, move the built HTML docs to the docs/
# folder for easier access.
html:
	@mkdir -p _examples/
	@cp ../notebooks/*.ipynb _examples/
	@cp ../notebooks/*.png _examples/
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf _examples
	@mv ../docs/html _html
	@rm -rf ../docs
	@mv _html ../docs
	@rm -rf ../docs/_examples/*.ipynb
	@echo "The HTML pages are actually in ../docs/"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
