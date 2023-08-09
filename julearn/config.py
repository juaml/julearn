"""Global config module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any

from .utils import logger, raise_error


_global_config = {}
_global_config["MAX_X_WARNS"] = 5000
_global_config["disable_x_check"] = False
_global_config["disable_xtypes_check"] = False
_global_config["disable_x_verbose"] = False
_global_config["disable_xtypes_verbose"] = False


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
        raise_error(
            f"Global config {key} does not exist"
        )
    logger.info(f"Setting global config {key} to {value}")
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
