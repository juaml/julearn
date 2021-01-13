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


def register_scorer(name, scorer, overwriting=None):
    """register a scorer, so that you can access it in scoring with its name.

    Parameters
    ----------
    name : str
        name of the scorer you want to register
    scorer : callable
        function of signature (estimator, X, y) see:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    overwriting : bool | None, optional
        decides whether overwriting should be allowed, by default None.
        Options are:

        * None : overwriting is possible, but warns the user
        * True : overwriting is possible without any warning
        * False : overwriting is not possible, error is raised instead
    """
    if name in list_scorers():
        if overwriting:
            _extra_available_scorers[name] = scorer
            logger.info(f'registering scorer named {name}')
        elif overwriting is None:
            _extra_available_scorers[name] = scorer
            warn(
                f'scorer named {name} already exists. '
                f'Therefore, {name} will be overwritten. '
                'To remove this warning set overwriting=True '
            )
            logger.info(f'registering scorer named {name}')
        else:
            raise_error(
                f'scorer named {name} already exists and '
                'overwriting is set to False, therefore you cannot overwrite '
                'existing scorers. Set overwriting=True in case you want to '
                'overwrite existing scorers')
    else:
        _extra_available_scorers[name] = scorer
        logger.info(f'registering scorer named {name}')


def reset_scorer_register():
    global _extra_available_scorers
    _extra_available_scorers = deepcopy(_extra_available_scorers_reset)
