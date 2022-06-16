from copy import deepcopy
from sklearn.metrics import _scorer, SCORERS, make_scorer
from julearn.utils import warn, raise_error, logger
from . metrics import r2_corr
_extra_available_scorers = {
    'r2_corr': make_scorer(r2_corr)
}

_extra_available_scorers_reset = deepcopy(_extra_available_scorers)


def get_scorer(name):
    """get available scorer by name

    Parameters
    ----------
    name : str
        name of an available scorer

    Returns
    -------
    scorer : callable
        function signature: `(estimator, X, y)` for more information see:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    """
    scorer = _extra_available_scorers.get(name)
    if scorer is None:
        try:
            scorer = _scorer.get_scorer(name)
        except ValueError:
            raise_error(
                f'{name} is not a valid scorer '
                'please use julearn.scorers.list_scorers to get a list'
                'of possible scorers'
            )
    return scorer


def list_scorers():
    """list all available scorers.

    Returns
    -------
    list
        a list containing all available scorers.
    """
    return {**SCORERS, **_extra_available_scorers}.keys()


def register_scorer(scorer_name, scorer, overwrite=None):
    """register a scorer, so that you can access it in scoring with its name.

    Parameters
    ----------
    scorer_name : str
        name of the scorer you want to register
    scorer : callable
        function of signature (estimator, X, y) see:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    overwrite : bool | None, optional
        decides whether overwrite should be allowed, by default None.
        Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warning
        * False : overwrite is not possible, error is raised instead
    """
    if scorer_name in list_scorers():
        if overwrite is None:
            warn(
                f'scorer named {scorer_name} already exists. '
                f'Therefore, {scorer_name} will be overwritten. '
                'To remove this warning set overwrite=True '
            )
            logger.info(f'registering scorer named {scorer_name}')
        elif overwrite is False:
            raise_error(
                f'scorer named {scorer_name} already exists and '
                'overwrite is set to False, therefore you cannot overwrite '
                'existing scorers. Set overwrite=True in case you want to '
                'overwrite existing scorers')
    logger.info(f'registering scorer named {scorer_name}')
    _extra_available_scorers[scorer_name] = scorer


def reset_scorer_register():
    global _extra_available_scorers
    _extra_available_scorers = deepcopy(_extra_available_scorers_reset)
