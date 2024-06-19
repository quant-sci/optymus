# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import os
import shutil

	
sys.path.insert(0, os.path.abspath('../src'))
print(sys.path)

import optymus
# notebooks_path = os.path.join(work_dir, 'notebooks')

# def copy_folder(origen, destino):
#     try:
#         if not os.path.exists(destino):
#             os.makedirs(destino)
#         shutil.copytree(origen, destino, dirs_exist_ok=True)
#         print(f"Folder copied from {origen} to {destino} sucessfully.")
#     except Exception as e:
#         print(f"Error when trying to copy folder: {e}")

# dirname = os.path.basename(notebooks_path)
# copy_folder(notebooks_path, os.path.join(os.getcwd(), 'examples', dirname))


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


# -- Project information -----------------------------------------------------

project = 'optymus'
copyright = '2024, quantsci'
author = 'Kleyton da Costa'

# The full version, including alpha/beta/rc tags
release = '0.1.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    'sphinx.ext.viewcode',
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "IPython.sphinxext.ipython_console_highlighting",
]

nbsphinx_allow_errors = True 
nbsphinx_execute = 'never'

html_show_sourcelink = False
# autodoc options
autodoc_default_options = {"members": True, "inherited-members": True}

# Turn on autosummary
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "generated/*",
    ".ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_logo = '../logo.svg'
html_favicon = '../logo.svg'

# add github link to sidebar
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/quant-sci/optymus",
            "icon": "fab fa-github",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    "css/custom_style.css",
]

html_js_files = [
    "require.min.js",
    "custom.js",
]