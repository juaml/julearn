"""Provide registry for target transformers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

from ...utils import logger, raise_error, warn_with_log
from .ju_target_transformer import JuTargetTransformer
from .target_confound_remover import TargetConfoundRemover


_available_target_transformers: Dict[str, Type[JuTargetTransformer]] = {
    "confound_removal": TargetConfoundRemover,
}

_available_target_transformers_reset = deepcopy(_available_target_transformers)


def list_target_transformers() -> List[str]:
    """List all the available target transformers.

    Returns
    -------
    out : list of str
        A list will all the available transformer names.

    """
    return list(_available_target_transformers.keys())


def get_target_transformer(name: str, **params: Any) -> JuTargetTransformer:
    """Get a target transformer by name.

    Parameters
    ----------
    name : str
        The target transformer name
    **params
        Parameters for the transformer.

    Returns
    -------
    JuTargetTransformer
        The transformer object.

    Raises
    ------
    ValueError
        If the specified target transformer name is not available.

    """
    out = None
    if name not in _available_target_transformers:
        raise_error(
            f"The specified target transformer ({name}) is not available. "
            f"Valid options are: {list(_available_target_transformers.keys())}"
        )
    trans = _available_target_transformers[name]
    out = trans(**params)  # type: ignore
    return out


def register_target_transformer(
    transformer_name: str,
    transformer_cls: Type[JuTargetTransformer],
    overwrite: Optional[bool] = None,
):
    """Register a target transformer to julearn.

    Parameters
    ----------
    transformer_name : str
        Name by which the transformer will be referenced by
    transformer_cls : class(JuTargetTransformer)
        The class by which the transformer can be initialized from.
    overwrite : bool, optional
        decides whether overwrite should be allowed.
        Options are:

        * None : overwrite is possible, but warns the user (default).
        * True : overwrite is possible without any warning.
        * False : overwrite is not possible, error is raised instead.

    Raises
    ------
    ValueError
        If `transformer_name` is already registered and `overwrite` is False.

    Warns
    -----
    RuntimeWarning
        If `transformer_name` is already registered and `overwrite` is None.

    """
    if _available_target_transformers.get(transformer_name) is not None:
        if overwrite is None:
            warn_with_log(
                f"Target transformer named {transformer_name} already exists. "
                f"Therefore, {transformer_name} will be overwritten. To "
                "remove this warning set overwrite=True."
            )
        elif overwrite is False:
            raise_error(
                f"Target transformer named {transformer_name} already exists "
                "and overwrite is set to False. Set `overwrite=True` "
                "in case you want to overwrite an existing target "
                "transformer."
            )

    logger.info(f"registering transformer named {transformer_name}.")

    _available_target_transformers[transformer_name] = transformer_cls


def reset_target_transformer_register() -> None:
    """Reset the target transformer register to its initial state."""
    global _available_target_transformers
    _available_target_transformers = deepcopy(
        _available_target_transformers_reset
    )
