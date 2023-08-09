"""Class that provides a model that supports transforming the target."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import typing
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if

from ...base import JuBaseEstimator
from ...utils import raise_error, warn_with_log
from ...utils.typing import DataLike, ModelLike


if TYPE_CHECKING:
    from ...pipeline.target_pipeline import JuTargetPipeline


def _wrapped_model_has(attr):
    """Create a function to check if self.model_ has a given attribute.

    This function is usable by
    :func:`sklearn.utils.metaestimators.available_if`

    Parameters
    ----------
    attr : str
        The attribute to check for.

    Returns
    -------
    check : function
        The check function.

    """

    def check(self):
        """Check if self.model has a given attribute.

        Returns
        -------
        bool
            True if self.model_ has the attribute, False otherwise.
        """
        return hasattr(self.model, attr)

    return check


class TransformedTargetWarning(RuntimeWarning):
    """Warning used to notify the user that the target has been transformed."""


class JuTransformedTargetModel(JuBaseEstimator):
    """Class that provides a model that supports transforming the target.

    This _model_ is a wrapper that will transform the target before fitting.

    Parameters
    ----------
    model : ModelLike
        The model to be wrapped. Can be a pipeline.
    transformer : JuTargetPipeline
        The transformer to be used to transform the target.
    """

    def __init__(self, model: ModelLike, transformer: "JuTargetPipeline"):
        self.model = model
        self.transformer = transformer

    def fit(
        self, X: pd.DataFrame, y: DataLike, **fit_params: Any  # noqa: N803
    ) -> "JuTransformedTargetModel":
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.
        **fit_params : dict
            Additional parameters to be passed to the model fit method.

        Returns
        -------
        JuTransformedTargetModel
            The fitted model.

        """
        y = self.transformer.fit_transform(X, y)
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **fit_params)  # type: ignore
        return self

    def predict(self, X: pd.DataFrame) -> DataLike:  # noqa: N803
        """Predict using the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.

        """
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast(ModelLike, self.model_)
        # TODO: Check if we can inverse the y transformations
        # Raise warning if not possible
        y_pred = self.model_.predict(X)
        if self.transformer.can_inverse_transform():
            y_pred = self.transformer.inverse_transform(X, y_pred)
        else:
            warn_with_log(
                "The target has been transformed to fit the model, but cannot "
                "inverse the model's prediction. The output of `predict(X)` "
                "is still in the transformed space. To remove this warning, "
                "use a suitable julearn scorer.",
                category=TransformedTargetWarning,
            )
        return y_pred

    def score(self, X: pd.DataFrame, y: DataLike) -> float:  # noqa: N803
        """Score the model.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        float
            Score for the model.

        """
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast(ModelLike, self.model_)
        y_trans = self.transform_target(X, y)
        return self.model_.score(X, y_trans)

    @available_if(_wrapped_model_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast(ModelLike, self.model_)
        return self.model_.predict_proba(X)  # type: ignore

    @available_if(_wrapped_model_has("decision_function"))
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to obtain the decision function.

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast(ModelLike, self.model_)
        return self.model_.decision_function(X)  # type: ignore

    def transform_target(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> DataLike:
        """Transform target.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        DataLike
            The transformed target.

        """
        return self.transformer.transform(X, y)

    @property
    def classes_(self) -> np.ndarray:
        """Get the classes of the model."""
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast(ModelLike, self.model_)
        return self.model_.classes_
