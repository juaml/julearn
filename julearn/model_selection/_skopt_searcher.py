"""Module for registering the BayesSearchCV class from scikit-optimize."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from .available_searchers import _recreate_reset_copy, register_searcher


try:
    from skopt import BayesSearchCV
except ImportError:
    from sklearn.model_selection._search import BaseSearchCV

    class BayesSearchCV(BaseSearchCV):
        """Dummy class for BayesSearchCV that raises ImportError.

        This class is used to raise an ImportError when BayesSearchCV is
        requested but scikit-optimize is not installed.

        """

        def __init__(*args, **kwargs):
            raise ImportError(
                "BayesSearchCV requires scikit-optimize to be installed."
            )


def register_bayes_searcher():
    register_searcher("bayes", BayesSearchCV, "search_spaces")

    # Update the "reset copy" of available searchers
    _recreate_reset_copy()
