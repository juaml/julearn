"""Module for registering the BayesSearchCV class from scikit-optimize."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Dict

from ..utils import logger
from .available_searchers import _recreate_reset_copy, register_searcher


try:
    import skopt.space as sksp
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


def _prepare_skopt_hyperparameters_distributions(
    params_to_tune: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare hyperparameters distributions for RandomizedSearchCV.

    This method replaces tuples with distributions for RandomizedSearchCV
    following the skopt convention. That is, if a parameter is a tuple
    with 3 elements, the first two elements are the bounds of the
    distribution and the third element is the type of distribution. In case
    the last element is "categorical", the parameter is considered
    categorical and all the previous elements are the choices.

    Parameters
    ----------
    params_to_tune : dict
        The parameters to tune.

    Returns
    -------
    dict
        The modified parameters to tune.

    """
    out = {}
    for k, v in params_to_tune.items():
        if isinstance(v, tuple) and len(v) == 3:
            prior = v[2]
            if prior == "categorical":
                logger.info(
                    f"Hyperparameter {k} is categorical with 2 "
                    f"options: [{v[0]} and {v[1]}]"
                )
                out[k] = sksp.Categorical(v[:-1])
            elif isinstance(v[0], int) and isinstance(v[1], int):
                logger.info(
                    f"Hyperparameter {k} is {prior} integer "
                    f"[{v[0]}, {v[1]}]"
                )
                out[k] = sksp.Integer(v[0], v[1], prior=prior)
            elif isinstance(v[0], float) and isinstance(v[1], float):
                logger.info(
                    f"Hyperparameter {k} is {prior} float " f"[{v[0]}, {v[1]}]"
                )
                out[k] = sksp.Real(v[0], v[1], prior=prior)
            else:
                logger.info(f"Hyperparameter {k} as is {v}")
                out[k] = v
        elif (
            isinstance(v, tuple)
            and isinstance(v[-1], str)
            and v[-1] == "categorical"
        ):
            out[k] = sksp.Categorical(v[:-1])
        else:
            logger.info(f"Hyperparameter {k} as is {v}")
            out[k] = v
    return out
