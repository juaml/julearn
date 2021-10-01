# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from sphinx_gallery.sorting import ExplicitOrder

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path
import sys
curdir = Path(__file__).parent
sys.path.append((curdir / 'sphinxext').as_posix())

# -- Project information -----------------------------------------------------

project = 'julearn'
copyright = '2020, Authors of julearn'
author = 'Fede Raimondo'


# -- General configuration ---------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'sphinx_rtd_theme',
    'sphinx_multiversion',
    'numpydoc',
    'gh_substitutions'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_sidebars = {
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html']
}

html_logo = 'images/julearn_logo_it.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'sklearn': ('https://scikit-learn.org/stable', None)
}

sphinx_gallery_conf = {
    'examples_dirs': ['../examples/basic', '../examples/advanced'],
    'subsection_order': ExplicitOrder(['../examples/basic/',
                                       '../examples/advanced/'
                                       ]),
    'gallery_dirs': ['auto_examples/basic', 'auto_examples/advanced'],
    'filename_pattern': '/(plot|run)_',
    'backreferences_dir': 'generated',
}


autosummary_generate = True
numpydoc_show_class_members = False
autoclass_content = 'both'

# sphinx-multiversion options
smv_rebuild_tags = False
smv_tag_whitelist = r'^v\d+\.\d+.\d+$'
smv_branch_whitelist = r'main'
smv_released_pattern = r'^tags/v.*$'


# Options for rtd-theme
html_theme_options = {
    'analytics_id': 'G-CB4ETK4DBG',  # Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'display_version': True,
}
