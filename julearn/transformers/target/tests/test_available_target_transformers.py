"""Provide tests for the target transformer's registry."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest

from julearn.transformers.target import (
    JuTargetTransformer,
    get_target_transformer,
    list_target_transformers,
    register_target_transformer,
    reset_target_transformer_register,
)


def test_register_target_transformer() -> None:
    """Test registering target transformers."""
    with pytest.raises(ValueError, match=r"\(useless\) is not available"):
        get_target_transformer("useless")

    first = list_target_transformers()

    class MyTransformer(JuTargetTransformer):
        pass

    register_target_transformer("useless", MyTransformer)
    _ = get_target_transformer("useless")

    second = list_target_transformers()

    assert "useless" in second
    assert "useless" not in first

    register_target_transformer("useless", MyTransformer, True)

    with pytest.warns(
        RuntimeWarning,
        match="Target transformer named useless already exists. ",
    ):
        register_target_transformer("useless", MyTransformer, None)

    with pytest.raises(
        ValueError, match="Target transformer named useless already exists and"
    ):
        register_target_transformer("useless", MyTransformer, False)
    reset_target_transformer_register()


def test_reset_target_transformer() -> None:
    """Test resetting the target transformers registry."""
    with pytest.raises(ValueError, match=r"\(useless\) is not available"):
        get_target_transformer("useless")

    class MyTransformer(JuTargetTransformer):
        pass

    register_target_transformer("useless", MyTransformer)
    get_target_transformer("useless")
    reset_target_transformer_register()
    with pytest.raises(ValueError, match=r"\(useless\) is not available"):
        get_target_transformer("useless")
