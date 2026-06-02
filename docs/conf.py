"""Provide configuration for Sphinx."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys
import types
from pathlib import Path
from typing import TypeAliasType

from setuptools_scm import get_version
from sphinx_autodoc_typehints import format_annotation
from sphinx_polyversion.api import load
from sphinx_polyversion.git import GitRefType


if os.getenv("POLYVERSION_DATA") is None:
    # If POLYVERSION_DATA is not set, we are likely building the docs locally.
    # In this case, we can use setuptools_scm to get the version information.
    release = get_version(root=Path(__file__).parents[1].resolve())
else:
    load(globals())
    # This adds the following to the global scope
    # html_context = {
    #     "revisions": [GitRef('main', ...), GitRef('v6.8.9', ...), ...],
    #     "current": GitRef('v1.4.6', ...),
    # }

    # process the loaded version information as you wish
    html_context = globals().get("html_context", {})

    if (
        html_context["current"].type_ == GitRefType.BRANCH
        and html_context["current"].name == "main"
    ):
        release = os.getenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_JULEARN")
    else:
        release = html_context["current"].name

version = release

# -- Path setup --------------------------------------------------------------

PROJECT_ROOT_DIR = Path(__file__).parents[1].resolve()
# get_scm_version = partial(get_version, root=PROJECT_ROOT_DIR)

# -- Project information -----------------------------------------------------

github_url = "https://github.com"
github_repo_org = "juaml"
github_repo_name = "julearn"
github_repo_slug = f"{github_repo_org}/{github_repo_name}"
github_repo_url = f"{github_url}/{github_repo_slug}"

project = github_repo_name
author = f"{project} Contributors"
copyright = f"{datetime.date.today().year}, {author}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Built-in extensions:
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx.ext.autosummary",  # generate autodoc summaries
    "sphinx.ext.doctest",  # test snippets in the documentation
    "sphinx.ext.extlinks",  # markup to shorten external links
    "sphinx.ext.intersphinx",  # link to other projects` documentation
    "sphinx.ext.mathjax",  # math support for HTML outputs in Sphinx
    # Third-party extensions:
    "sphinx_gallery.gen_gallery",  # HTML gallery of examples
    "numpydoc",  # support for NumPy style docstrings
    "sphinx_copybutton",  # copy button for code blocks
    "sphinxcontrib.towncrier.ext",  # towncrier fragment support
    "bokeh.sphinxext.bokeh_plot",  # bokeh plot support
    "sphinx_autodoc_typehints",  # automatically document type hints
]

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
]

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
    ("py:class", "sklearn.model_selection._split.*"),
    ("py:class", "sklearn.metrics._scorer.*"),
]

to_ignore = [
    "array-like",
    "shape",
    "optional",
    "or",
    "the",
    "options",
    "n_samples",
    "default=.*",
    "n_features",
    "n_outputs",
    "True",
    "False",
]

for i in to_ignore:
    nitpick_ignore_regex.append(("py:class", i))

suppress_warnings = ["sphinx_autodoc_typehints", "config.cache"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "julearn documentation"
html_logo = "images/julearn_logo_it.png"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
    "css/version_selector.css",
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
# autodoc_typehints = "description"
# autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {
    "ColumnTypes": "julearn.base.column_types.ColumnTypes",
    "Pipeline": "sklearn.pipeline.Pipeline",
    "RandomState": "numpy.random.RandomState",
    #     "ColumnTypesLike": "julearn.utils.typing.ColumnTypesLike",
    #     "DataLike": "julearn.utils.typing.DataLike",
    #     "ScorerLike": "julearn.utils.typing.ScorerLike",
    #     "EstimatorLike": "julearn.utils.typing.EstimatorLike",
    #     "ModelLike": "julearn.utils.typing.ModelLike",
    #     "TransformerLike": "julearn.utils.typing.TransformerLike",
    #     "JuModelLike": "julearn.utils.typing.JuModelLike",
    #     "JuTransformerLike": "julearn.utils.typing.JuTransformerLike",
    #     "CVLike": "julearn.utils.typing.CVLike",
}


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
    "skopt": ("https://scikit-optimize.readthedocs.io/en/latest", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
    "optuna_integration": (
        "https://optuna-integration.readthedocs.io/en/stable",
        None,
    ),
    "panel": ("https://panel.holoviz.org/", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable/", None),
}

# -- sphinx.ext.extlinks configuration ---------------------------------------

extlinks = {
    "gh": (f"{github_repo_url}/issues/%s", "#%s"),
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
    "filename_pattern": "/(plot|run)_",
    "download_all_examples": False,
}


# -- sphinxcontrib-towncrier configuration -----------------------------------

towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = True
towncrier_draft_working_directory = PROJECT_ROOT_DIR

# -- sphinx_autodoc_typehints configuration ----------------------------------
always_use_bars_union = True
simplify_optional_unions = True
typehints_defaults = "comma"
# Don't show return types at all
typehints_document_rtype = False

# Don't show "None" return types, but show all others
typehints_document_rtype_none = False

# Show the return type inline with the return description
# instead of as a separate block
typehints_use_rtype = False


def _get_canonical_type_alias_name(annotation: TypeAliasType) -> str:
    """Get canonical public qualified name for a TypeAliasType.

    For types defined in private modules (e.g. ``numpy._typing.ArrayLike``),
    search ``sys.modules`` for a public re-export
    (e.g. ``numpy.typing.ArrayLike``).
    """
    module = getattr(annotation, "__module__", "") or ""
    name = getattr(annotation, "__name__", "") or ""
    if not module or not name:
        return ""
    if not any(part.startswith("_") for part in module.split(".")):
        return f"{module}.{name}"
    top_pkg = module.split(".")[0]
    for mod_name in sorted(sys.modules):
        if not mod_name.startswith(top_pkg):
            continue
        mod = sys.modules[mod_name]
        if not isinstance(mod, types.ModuleType):
            continue
        if any(part.startswith("_") for part in mod_name.split(".")):
            continue
        if getattr(mod, name, None) is annotation:
            return f"{mod_name}.{name}"
    return f"{module}.{name}"


def _typehints_formatter(annotation, config):
    """Handle formatting of PEP695 type aliases in sphinx-autodoc-typehints."""
    if isinstance(annotation, TypeAliasType):
        module = getattr(annotation, "__module__", "") or ""
        name = getattr(annotation, "__name__", "") or ""
        intersphinx_mapping = getattr(config, "intersphinx_mapping", {})
        is_external = module and any(
            module == pkg or module.startswith(f"{pkg}.")
            for pkg in intersphinx_mapping
        )
        # Handle external PEP695 type aliases
        if is_external and name:
            canonical = _get_canonical_type_alias_name(annotation)
            if canonical:
                return f":py:obj:`~{canonical}`"
        # Unwrap internal PEP695 type aliases to their underlying types
        return format_annotation(annotation.__value__, config)
    return None


typehints_formatter = _typehints_formatter
