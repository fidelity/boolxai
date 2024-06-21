..
   We include the README, but then add a link to it as "Home". So, essentially the same
   content is seen at both index.html and readme.html. The reason is that if we use
   "self" in place of "readme", the subheadings will not show up in the table of
   contents, due to a bug or oversight in Sphinx, dating back to 2015 (and still open):
   https://github.com/sphinx-doc/sphinx/issues/2103)

.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2

   Home <readme>
   examples
   api
   contributing
   changelog
   license
   notices


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

* :ref:`search`
