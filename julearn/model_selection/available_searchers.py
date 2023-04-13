"""Provide registry of searchers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
from typing import List, Optional

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from julearn.utils.logging import logger, raise_error, warn_with_log


_available_searchers = {"grid": GridSearchCV, "random": RandomizedSearchCV}

# Keep a copy for reset
_available_searchers_reset = deepcopy(_available_searchers)


def list_searchers() -> List[str]:
    """List all available searching algorithms.

    Returns
    -------
    out : list(str)
        A list of all available searcher names.
    """
    return list(_available_searchers)


def get_searcher(name: str) -> object:
    """Get a searcher by name.

    Parameters
    ----------
    name : str
        The searchers name.

    Returns
    -------
    obj
        scikit-learn compatible searcher.

    Raises
    ------
    ValueError
        If the specified searcher is not available.
    """
    if name not in _available_searchers:
        raise_error(
            f"The specified searcher ({name}) is not available. "
            f"Valid options are: {list(_available_searchers.keys())}"
        )
    out = _available_searchers[name]
    return out


def register_searcher(
    searcher_name: str, searcher: object, overwrite: Optional[bool] = None
) -> None:
    """Register searcher to julearn.

    This function allows you to add a scikit-learn compatible searching
    algorithm to julearn. After, you can call it as all other searchers in
    julearn.

    Parameters
    ----------
    searcher_name : str
        Name by which the searcher will be referenced by.
    searcher : obj
        The searcher class by which the searcher can be initialized.
    overwrite : bool | None, optional
        decides whether overwrite should be allowed, by default None.
        Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warns
        * False : overwrite is not possible, error is raised instead

    Raises
    ------
    ValueError
        If the specified searcher is already available and overwrite is set to
        False.
    """
    if searcher_name in list_searchers():
        if overwrite is None:
            warn_with_log(
                f"searcher named {searcher_name} already exists. "
                f"Therefore, {searcher_name} will be overwritten. "
                "To remove this warn_with_loging set `overwrite=True`. "
            )
        elif overwrite is False:
            raise_error(
                f"searcher named {searcher_name} already exists and "
                "overwrite is set to False, therefore you cannot overwrite "
                "existing searchers. "
                "Set `overwrite=True` in case you want to "
                "overwrite existing searchers."
            )
    logger.info(f"Registering new searcher: {searcher_name}")
    _available_searchers[searcher_name] = searcher


def reset_searcher_register() -> None:
    """Reset the searcher register to its initial state."""
    global _available_searchers
    _available_searchers = deepcopy(_available_searchers_reset)
