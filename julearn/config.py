"""Global config module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any

from .utils import logger, raise_error
from .utils.logging import DelayedFmtMessage as __


_global_config = {}
_global_config["MAX_X_WARNS"] = 5000
_global_config["max_x_logs"] = 100
_global_config["disable_x_check"] = False
_global_config["disable_xtypes_check"] = False
_global_config["disable_x_verbose"] = False
_global_config["disable_xtypes_verbose"] = False
_global_config["enable_parallel_column_transformers"] = False
_global_config["enable_auto_escape_parenthesis"] = True


def set_config(key: str, value: Any) -> None:
    """Set a global config value.

    Parameters
    ----------
    key : str
        The key to set.
    value : Any
        The value to set.

    """
    if key not in _global_config:
        raise_error(f"Global config {key} does not exist")
    logger.info(
        __("Setting global config {key} to {value}", key=key, value=value)
    )
    _global_config[key] = value


def get_config(key: str) -> Any:
    """Get a global config value.

    Parameters
    ----------
    key : str
        The key to get.

    Returns
    -------
    Any
        The value of the key.

    """
    return _global_config.get(key, None)


def _joblib_htcondor_context_func() -> None:
    """Create a function to seet the config variables.

    Returns
    -------
    function
        A function to set the config variables.

    """
    from copy import deepcopy
    from functools import partial

    import sklearn

    _vars = deepcopy(_global_config)
    sklearn_config = sklearn.get_config()
    _vars["sklearn_config"] = sklearn_config

    def _set_context_vars(**kwargs):
        for key, value in kwargs.items():
            if key == "sklearn_config":
                sklearn.set_config(**value)
            else:
                set_config(key, value)

    return partial(_set_context_vars, **_vars)
