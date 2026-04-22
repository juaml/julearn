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


def _joblib_htcondor_context_func(current_func=None) -> None:
    """Create a function to seet the config variables.

    Parameters
    ----------
    current_func : function
        The current function that was set in the joblib context. If not None,
        it will be called before setting the config variables (default None).

    Returns
    -------
    function
        A function to set the config variables.

    """
    from copy import deepcopy
    from functools import partial

    import sklearn

    # Copy the global config
    _config = {}
    _config["julearn_config"] = deepcopy(_global_config)

    _config["sklearn_config"] = sklearn.get_config()

    # add the logging configuration
    level = logger.level
    fmt = None
    if len(logger.handlers) > 0:
        fmt = logger.handlers[0].formatter._fmt  # type: ignore

    _config["julearn_logging_config"] = {
        "level": level,
        "output_format": fmt,
    }

    # Log the current function, if any, to be called before setting the config
    _config["jht_current_func"] = current_func

    def _set_context_vars(**kwargs):
        # First, call the previous function
        jht_current_func = kwargs.pop("jht_current_func")
        if jht_current_func is not None:
            jht_current_func()

        # Julearn then sets the config variables. If user has set any variable
        # in the context function, current julearn state should override this
        # setting.
        for key, value in kwargs.items():
            if key == "sklearn_config":
                sklearn.set_config(**value)
            elif key == "julearn_logging_config":
                from julearn.utils import configure_logging
                configure_logging(**value)
            elif key == "julearn_config":
                for k, v in value.items():
                    set_config(k, v)

    return partial(_set_context_vars, **_config)  # type: ignore
