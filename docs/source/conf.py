# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'discontinuum'
author = 'Timothy Hodson and Keith Doore'
from discontinuum._version import __version__  # noqa: E402
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
os.environ["TQDM_DISABLE"] = "True"

extensions = [
    "myst_nb",
    "sphinxcontrib.mermaid",
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    ]

autosummary_generate = True
autodoc_default_options = {
       'members': True,
   }
templates_path = ['_templates']
exclude_patterns = []
nb_execution_timeout = -1


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
