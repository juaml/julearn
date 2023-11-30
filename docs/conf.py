# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pathlib import Path


# Check if sphinx-multiversion is installed
use_multiversion = False
try:
    import sphinx_multiversion  # noqa: F401

    use_multiversion = True
except ImportError:
    pass

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
    "numpydoc",
    "gh_substitutions",
    "sphinx_copybutton",
    "bokeh.sphinxext.bokeh_plot",
]

if use_multiversion:
    extensions.append("sphinx_multiversion")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/99_docs/*",
]

nitpicky = True

nitpick_ignore_regex = [
    ("py:class", "numpy._typing.*"),
    ("py:obj", "trimboth"),  # python 3.11 error
    ("py:obj", "tmean"),  # python 3.11 error
    ("py:obj", "subclass"),  # python 3.11 error
    ("py:class", "typing.Any"),  # python 3.11 error
    # ('py:class', 'numpy.typing.ArrayLike')
    ("py:obj", "sqlalchemy.engine.Engine"),  # ignore sqlalchemy
    # Sklearn doc issue to be solved in next release
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "sklearn.utils.metadata_routing.MetadataRequest"),
    ("py:class", "julearn.inspect._pipeline.PipelineInspector"),
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


# -- sphinx.ext.intersphinx configuration ------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    # "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
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


class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.

    Assumes README.txt exists for all subsections and uses the subsection with
    dashes, '---', as the adornment.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        readme = os.path.join(src_path, "README.txt")

        try:
            with open(readme) as f:
                content = f.read()
        except FileNotFoundError:
            return directory

        title_match = self.regex.search(content)
        if title_match is not None:
            return title_match.group(1)
        return directory


ex_dirs = [
    "00_starting",
    "01_model_comparison",
    "02_inspection",
    "03_complex_models",
    "04_confounds",
    "05_customization",
    "99_docs",
]

example_dirs = []
gallery_dirs = []
for t_dir in ex_dirs:
    example_dirs.append(f"../examples/{t_dir}")
    gallery_dirs.append(f"auto_examples/{t_dir}")

sphinx_gallery_conf = {
    "doc_module": "julearn",
    "backreferences_dir": "api/generated",
    "examples_dirs": example_dirs,
    "gallery_dirs": gallery_dirs,
    "nested_sections": True,
    "subsection_order": SubSectionTitleOrder("../examples"),
    "filename_pattern": "/(plot|run)_",
    "download_all_examples": False,
}

# -- sphinx-multiversion configuration ---------------------------------------

smv_rebuild_tags = False
smv_tag_whitelist = r"^v\d+\.\d+.\d+$"
smv_branch_whitelist = r"main"
smv_released_pattern = r"^tags/v.*$"
