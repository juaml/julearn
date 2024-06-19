"""Module for registering the BayesSearchCV class from scikit-optimize."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Dict

from ..utils import logger
from .available_searchers import _recreate_reset_copy, register_searcher


try:
    import optuna.distributions as optd

    from ..external.optuna_searchcv import OptunaSearchCV
except ImportError:
    from sklearn.model_selection._search import BaseSearchCV

    class OptunaSearchCV(BaseSearchCV):
        """Dummy class for OptunaSearchCV that raises ImportError.

        This class is used to raise an ImportError when OptunaSearchCV is
        requested but optuna and optuna-integration ar not installed.

        """

        def __init__(*args, **kwargs):
            raise ImportError(
                "OptunaSearchCV requires optuna and optuna-integration to be "
                "installed."
            )


def register_optuna_searcher():
    register_searcher("optuna", OptunaSearchCV, "param_distributions")

    # Update the "reset copy" of available searchers
    _recreate_reset_copy()


def is_optuna_valid_distribution(obj: Any) -> bool:
    """Check if an object is a valid Optuna distribution.

    Parameters
    ----------
    obj : any
        The object to check.

    Returns
    -------
    bool
        Whether the object is a valid Optuna distribution.

    """
    _valid_classes = [
        "IntDistribution",
        "FloatDistribution",
        "CategoricalDistribution",
    ]

    return obj.__class__.__name__ in _valid_classes


def _prepare_optuna_hyperparameters_distributions(
    params_to_tune: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare hyperparameters distributions for OptunaSearchCV.

    This method replaces tuples with distributions for OptunaSearchCV
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
            if v[2] == "uniform":
                if isinstance(v[0], int) and isinstance(v[1], int):
                    logger.info(
                        f"Hyperparameter {k} is uniform integer "
                        f"[{v[0]}, {v[1]}]"
                    )
                    out[k] = optd.IntDistribution(v[0], v[1], log=False)
                else:
                    logger.info(
                        f"Hyperparameter {k} is uniform float [{v[0]}, {v[1]}]"
                    )
                    out[k] = optd.FloatDistribution(v[0], v[1], log=False)
            elif v[2] == "log-uniform":
                if isinstance(v[0], int) and isinstance(v[1], int):
                    logger.info(
                        f"Hyperparameter {k} is log-uniform int "
                        f"[{v[0]}, {v[1]}]"
                    )
                    out[k] = optd.IntDistribution(v[0], v[1], log=True)
                else:
                    logger.info(
                        f"Hyperparameter {k} is log-uniform float "
                        f"[{v[0]}, {v[1]}]"
                    )
                    out[k] = optd.FloatDistribution(v[0], v[1], log=True)
            elif v[2] == "categorical":
                logger.info(
                    f"Hyperparameter {k} is categorical with 2 "
                    f"options: [{v[0]} and {v[1]}]"
                )
                out[k] = optd.CategoricalDistribution((v[0], v[1]))
            else:
                out[k] = v
        elif (
            isinstance(v, tuple)
            and isinstance(v[-1], str)
            and v[-1] == "categorical"
        ):
            logger.info(f"Hyperparameter {k} is categorical [{v[:-1]}]")
            out[k] = optd.CategoricalDistribution(v[:-1])
        else:
            logger.info(f"Hyperparameter {k} as is {v}")
            out[k] = v
    return out
