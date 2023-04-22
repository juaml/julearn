# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
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
sys.path.append((curdir / "sphinxext").as_posix())

# -- Project information -----------------------------------------------------

project = "julearn"
copyright = "2023, Authors of julearn"
author = "Fede Raimondo"


# -- General configuration ---------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx_multiversion",
    "numpydoc",
    "gh_substitutions",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False

nitpick_ignore_regex = [
    ("py:class", "numpy._typing.*"),
    ("py:obj", "trimboth"),  # python 3.11 error
    ("py:obj", "tmean"),  # python 3.11 error
    ("py:obj", "subclass"),  # python 3.11 error
    ("py:class", "typing.Any"),  # python 3.11 error
    # ('py:class', 'numpy.typing.ArrayLike')
    ("py:obj", "sqlalchemy.engine.Engine"),  # ignore sqlalchemy
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_logo = "images/julearn_logo_it.png"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

html_js_files = [
    "js/custom.js",
]

html_use_modindex = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "versions.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}


# -- sphinx.ext.autodoc configuration ----------------------------------------

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- sphinx.ext.autosummary configuration ------------------------------------

autosummary_generate = True

autodoc_default_options = {
    "imported-members": True,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    #  We cannot have __init__: it causes duplicated entries
    #  'special-members': '__init__',
}

def touch_example_backreferences(app, what, name, obj, options, lines):
    # generate empty examples files, so that we don't get
    # inclusion errors if there are no examples for a class / module
    examples_path = os.path.join(
        app.srcdir, "api", "generated", f"{name}.examples"
    )
    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, "w").close()


def setup(app):
    app.connect("autodoc-process-docstring", touch_example_backreferences)

# -- sphinx.ext.intersphinx configuration ------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    # "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# -- numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_xref_param_type = False
numpydoc_xref_aliases = {
    "Path": "pathlib.Path",
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "Nifti2Image": "nibabel.nifti2.Nifti2Image",
    # "Engine": "sqlalchemy.engine.Engine",
}
numpydoc_xref_ignore = {
    "of",
    "shape",
    "optional",
    "or",
    "the",
    "options",
    "function",
    "object",
    "class",
    "objects",
    "Engine",
    "positive",
    "negative",
    "compatible",
    "TransformerLike",
    "ModelLike",
    "EstimatorLike",
}
# numpydoc_validation_checks = {
#     "all",
#     "GL01",
#     "GL02",
#     "GL03",
#     "ES01",
#     "SA01",
#     "EX01",
# }


# -- Sphinx-Gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../examples/",
    "gallery_dirs": "auto_examples",
    "nested_sections": True,
    "filename_pattern": "/(plot|run)_",
    "backreferences_dir": "api/generated",
    "doc_module": "julearn",
}

# -- sphinx-multiversion configuration ---------------------------------------

smv_rebuild_tags = False
smv_tag_whitelist = r"^v\d+\.\d+.\d+$"
smv_branch_whitelist = r"main"
smv_released_pattern = r"^tags/v.*$"
