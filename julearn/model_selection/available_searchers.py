from copy import deepcopy
from julearn.utils.logging import raise_error, logger, warn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


_available_searchers = {
    'grid': GridSearchCV,
    'random': RandomizedSearchCV
}
_available_searchers_reset = deepcopy(_available_searchers)


def list_searchers():
    """ List all available seraching algorithms

    Returns
    -------
    out : list(str)
        A list of all available seracher names.
    """
    return list(_available_searchers)


def get_searcher(name):
    """Get a searcher by name.

    Parameters
    ----------
    name : str
        The searchers name.

    Returns
    -------
    out : obj
        scikit-learn compatible searcher.
    """
    if name not in _available_searchers:
        raise_error(
            f'The specified searcher ({name}) is not available. '
            f'Valid options are: {list(_available_searchers.keys())}'
        )
    out = _available_searchers[name]
    return out


def register_searcher(name, searcher, overwriting=None):
    """Register searcher to julearn.
    This function allows you to add a scikit-learn compatible searching
    algorithm to julearn. Afterwars, you can call it as all other searchers in
    julearn.

    Parameters
    ----------
    name : str
        Name by which the searcher will be referenced by.
    searcher : obj
        The searcher class by which the searcher can be initialized.
    overwriting : bool | None, optional
        decides whether overwriting should be allowed, by default None.
        Options are:

        * None : overwriting is possible, but warns the user
        * True : overwriting is possible without any warning
        * False : overwriting is not possible, error is raised instead

    """
    if name in list_searchers():
        if overwriting:
            _available_searchers[name] = searcher
            logger.info(f'registering searcher names {name}')
        elif overwriting is None:
            _available_searchers[name] = searcher
            warn(
                f'searcher named {name} already exists. '
                f'Therfore, {name} will be overwritten. '
                'To remove this warning set `overwriting=True`. '
            )

            logger.info(f'registering searcher names {name}')

        else:
            raise_error(
                f'searcher named {name} already exists and '
                'overwriting is set to False, therefore you cannot overwrite '
                'existing searchers. '
                'Set `overwriting=True` in case you want to '
                'overwrite existing scorers.'
            )
    else:
        _available_searchers[name] = searcher
        logger.info(f'registering searcher names {name}')


def reset_searcher_register():
    global _available_searchers
    _available_searchers = deepcopy(_available_searchers_reset)
