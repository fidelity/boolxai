# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# cd docsrc and then make html
# need to add docsrc and ../ to path so that we can import boolxai
sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.join(os.path.dirname(__name__), '..'))

from boolxai._version import __author__, __copyright__, __version__

# Allow Sphinx to pull files from ../docsrc/
sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

# -- Project information -----------------------------------------------------

project = "BoolXAI"
copyright = __copyright__
author = __author__
release = __version__
version = __version__

# Necessary installations for `make html` to work
# pip install pandoc (on windows, requires separate pandoc installation)
# pip install sphinx-rtd-theme
# pip install sphinx-copybutton
# pip install nbsphinx
# pip install myst-parser

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "myst_parser",  # For parsing markdown, such as README files
    "nbsphinx",  # For parsing Jupyter notebooks
    "sphinx_copybutton",  # Adds copy button to code snippets
    "sphinx.ext.viewcode",  # Adds source button - switch to sphinx.ext.linkcode later
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffixes as a list of string.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Common links
extlinks = {
    "repo": ("https://github.com/fidelity/boolxai%s", None),
    "docs": ("https://fidelity.github.io/boolxai%s", None),
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",  # Otherwise, __init__ is excluded by default
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "signature"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "BoolXAI"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "BoolXAI.tex", "BoolXAI Documentation", author, "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "boolxai", "BoolXAI Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "BoolXAI",
        "BoolXAI Documentation",
        author,
        author,
        "BoolXAI: Explainable AI using expressive Boolean formulas",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------
