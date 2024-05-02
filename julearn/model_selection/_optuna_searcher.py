"""Module for registering the BayesSearchCV class from scikit-optimize."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from typing import Any, Dict

from .available_searchers import _recreate_reset_copy, register_searcher


try:
    from optuna_integration.sklearn import OptunaSearchCV
    import optuna.distributions as od
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


def _prepare_optuna_hyperparameters_distributions(
    params_to_tune: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare hyperparameters distributions for RandomizedSearchCV.

    This method replaces tuples with distributions for RandomizedSearchCV
    following the skopt convention. That is, if a parameter is a tuple
    with 3 elements, the first two elements are the bounds of the
    distribution and the third element is the type of distribution.

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
                    out[k] = od.IntDistribution(v[0], v[1], log=False)
                else:
                    out[k] = od.FloatDistribution(v[0], v[1], log=False)
            elif v[2] in ("loguniform", "log-uniform"):
                if isinstance(v[0], int) and isinstance(v[1], int):
                    out[k] = od.IntDistribution(v[0], v[1], log=True)
                else:
                    out[k] = od.FloatDistribution(v[0], v[1], log=True)
            else:
                out[k] = v
        else:
            out[k] = v
    return out
