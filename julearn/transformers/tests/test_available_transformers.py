"""Provide tests for the transformer register."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import warnings
from typing import Optional

import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from julearn.transformers.available_transformers import (
    get_transformer,
    register_transformer,
    reset_transformer_register,
)
from julearn.utils.testing import PassThroughTransformer
from julearn.utils.typing import DataLike


class Fish(BaseEstimator, TransformerMixin):
    """A (flying) fish.

    Parameters
    ----------
    can_it_fly : bool
        Whether the fish can fly.

    """

    def __init__(self, can_it_fly: bool):
        self.can_it_fly = can_it_fly

    def fit(self, X: DataLike, y: Optional[DataLike] = None) -> "Fish":
        """Fit the fish.

        Parameters
        ----------
        X : DataLike
            The data.
        y : DataLike, optional
            The target, by default None

        Returns
        -------
        Fish
            The fitted fish.
        """
        return self

    def transform(self, X: DataLike) -> DataLike:
        """Transform the data.

        Parameters
        ----------
        X : DataLike
            The data.

        Returns
        -------
        DataLike
            The transformed data.
        """
        return X


def test_register_reset() -> None:
    """Test the register reset."""
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer("passthrough", PassThroughTransformer)
    assert get_transformer("passthrough").__class__ == PassThroughTransformer

    with pytest.warns(RuntimeWarning, match="Transformer named"):
        register_transformer("passthrough", PassThroughTransformer)
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer("passthrough", PassThroughTransformer, "continuous")
    assert get_transformer("passthrough").__class__ == PassThroughTransformer


def test_register_class_no_default_params():
    """Test the register with a class that has no default params."""
    reset_transformer_register()
    register_transformer("fish", Fish)
    get_transformer("fish", can_it_fly="dont_be_stupid")


def test_register_warnings_errors():
    """Test the register warning / error."""
    with pytest.warns(RuntimeWarning, match="Transformer name"):
        register_transformer("zscore", Fish)
    reset_transformer_register()

    with pytest.raises(ValueError, match="Transformer name"):
        register_transformer("zscore", Fish, overwrite=False)
    reset_transformer_register()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_transformer("zscore", Fish, overwrite=True)
    reset_transformer_register()
