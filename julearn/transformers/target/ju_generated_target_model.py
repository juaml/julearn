"""Class that provides a model that supports generating the target."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import typing
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if

from ...base import JuBaseEstimator
from ...utils import logger, raise_error, warn_with_log
from ...utils.typing import DataLike, EstimatorLike, JuEstimatorLike, ModelLike


if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline  # noqa: F401


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


class GeneratedTargetWarning(RuntimeWarning):
    """Warning used to notify the user that the target has been generated."""


class JuGeneratedTargetModel(JuBaseEstimator):
    """Class that provides a model that supports generating the target.

    This _model_ is a wrapper that will generate the target before fitting.

    Parameters
    ----------
    model : ModelLike
        The model to be wrapped. Can be a pipeline.
    transformer : Pipeline
        The transformer to be used to generate the target.

    """

    def __init__(
        self,
        model: ModelLike,
        transformer: Union[EstimatorLike, JuEstimatorLike],
    ) -> None:
        self.model = model
        self.transformer = transformer

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: DataLike,
        **fit_params: Any,
    ) -> "JuGeneratedTargetModel":
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : {"__generated__"}
            The target. Should be the string "__generated__"
        **fit_params : dict
            Additional parameters to be passed to the model fit method.

        Returns
        -------
        JuGeneratedTargetModel
            The fitted model.

        """
        if np.any(y != 0):
            warn_with_log(
                "The target should be the generated but a non-zero target was "
                "provided. The target will be ignored and the generated "
                "target will be used instead."
            )
        logger.debug("Fitting the target generator")
        self.transformer.fit(X)  # type: ignore
        y_gen = self.generate_target(X, y)
        self.model_ = clone(self.model)  # type: ignore
        logger.debug("Fitting model from generated target")
        self.model_.fit(X, y_gen, **fit_params)  # type: ignore
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
        self.model_ = typing.cast("ModelLike", self.model_)
        y_pred = self.model_.predict(X)
        warn_with_log(
            "The target has been generated from the features. The output of "
            "`predict(X)` is in the transformed space. You can either "
            "use a suitable julearn scorer or use the `generate_target` "
            "method before scoring",
            category=GeneratedTargetWarning,
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
        if np.any(y != 0):
            warn_with_log(
                "The target should be the generated but a non-zero target was "
                "provided. The target will be ignored and the generated "
                "target will be used instead."
            )
        self.model_ = typing.cast("ModelLike", self.model_)
        y_gen = self.generate_target(X, y)
        logger.debug("Scoring model using generated target")
        return self.model_.score(X, y_gen)

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
        self.model_ = typing.cast("ModelLike", self.model_)
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
        self.model_ = typing.cast("ModelLike", self.model_)
        return self.model_.decision_function(X)  # type: ignore

    def generate_target(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: DataLike,
    ) -> DataLike:
        """Generate the target.

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
        logger.debug("Generating target")
        gen_y = self.transformer.transform(X)  # type: ignore

        # If it's a pandas dataframe convert to series
        if gen_y.shape[1] == 1:
            gen_y = gen_y.iloc[:, 0]
            logger.debug(f"Target generated: {gen_y.name}")
        else:
            logger.debug(f"Target generated: {gen_y.columns}")
        return gen_y

    @property
    def classes_(self) -> np.ndarray:
        """Get the classes of the model.

        Returns
        -------
        np.ndarray
            The classes of the model.

        """
        if not hasattr(self, "model_"):
            raise_error("Model not fitted yet.")
        self.model_ = typing.cast("ModelLike", self.model_)
        return self.model_.classes_
