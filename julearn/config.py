from typing import Any
from .utils import logger

_global_config = {}
_global_config["MAX_X_WARNS"] = 1000

def set(key : str, value : Any):
    """Set a global config value.

    Parameters
    ----------
    key : str
        The key to set.
    value : Any
        The value to set.
    """
    logger.info(f"Setting global config {key} to {value}")
    _global_config[key] = value


def get(key : str) -> Any:
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