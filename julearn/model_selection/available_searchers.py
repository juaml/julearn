# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
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


def register_searcher(name, searcher, overwrite=None):
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
    overwrite : bool | None, optional
        decides whether overwrite should be allowed, by default None.
        Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warning
        * False : overwrite is not possible, error is raised instead

    """
    if name in list_searchers():
        if overwrite is None:
            warn(
                f'searcher named {name} already exists. '
                f'Therefore, {name} will be overwritten. '
                'To remove this warning set `overwrite=True`. '
            )
        elif overwrite is False:
            raise_error(
                f'searcher named {name} already exists and '
                'overwrite is set to False, therefore you cannot overwrite '
                'existing searchers. '
                'Set `overwrite=True` in case you want to '
                'overwrite existing searchers.'
            )
        logger.info(f'Registering new searcher: {name}')
        _available_searchers[name] = searcher


def reset_searcher_register():
    global _available_searchers
    _available_searchers = deepcopy(_available_searchers_reset)
