"""Protocols for type checking."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from sklearn.metrics._scorer import (
    _PredictScorer,
    _ProbaScorer,
    _ThresholdScorer,
)

from ..base import ColumnTypes


DataLike = Union[np.ndarray, pd.DataFrame, pd.Series]

ScorerLike = Union[_ProbaScorer, _ThresholdScorer, _PredictScorer]


@runtime_checkable
class EstimatorLikeFit1(Protocol):
    """Class for estimator-like fit 1."""

    def fit(
        self, X: List[str], y: str, **kwargs: Any  # noqa: N803
    ) -> "EstimatorLikeFit1":
        """Fit estimator.

        Parameters
        ----------
        X : list of str
            The features to use.
        y : str
            The target to use.
        **kwargs : dict
            Extra keyword arguments.

        Returns
        -------
        EstimatorLikeFit1
            The fitted estimator.

        """
        return self

    def get_params(self, deep: bool = True) -> Dict:
        """Get params.

        Parameters
        ----------
        deep : bool, optional
            Whether to get in a deep fashion (default True).

        Returns
        -------
        dict
            The parameters.

        """
        return {}

    def set_params(self, **params: Any) -> "EstimatorLikeFit1":
        """Set params.

        Parameters
        ----------
        **params : dict
            The parameters to set.

        Returns
        -------
        EstimatorLikeFit1
            Estimator with set parameters.

        """
        return self


@runtime_checkable
class EstimatorLikeFit2(Protocol):
    """Class for estimator-like fit 2."""

    def fit(self, X: List[str], y: str) -> "EstimatorLikeFit2":  # noqa: N803
        """Fit estimator.

        Parameters
        ----------
        X : list of str
            The features to use.
        y : str
            The target to use.

        Returns
        -------
        EstimatorLikeFit2
            The fitted estimator.

        """
        return self

    def get_params(self, deep: bool = True) -> Dict:
        """Get params.

        Parameters
        ----------
        deep : bool, optional
            Whether to get in a deep fashion (default True).

        Returns
        -------
        dict
            The parameters.

        """
        return {}

    def set_params(self, **params: Any) -> "EstimatorLikeFit2":
        """Set params.

        Parameters
        ----------
        **params : dict
            The parameters to set.

        Returns
        -------
        EstimatorLikeFit2
            Estimator with set parameters.

        """
        return self


@runtime_checkable
class EstimatorLikeFity(Protocol):
    """Class for estimator-like fit y."""

    def fit(self, y: str) -> "EstimatorLikeFity":
        """Fit estimator.

        Parameters
        ----------
        y : str
            The target to use.

        Returns
        -------
        EstimatorLikeFity
            The fitted estimator.

        """
        return self

    def get_params(self, deep: bool = True) -> Dict:
        """Get params.

        Parameters
        ----------
        deep : bool, optional
            Whether to get in a deep fashion (default True).

        Returns
        -------
        dict
            The parameters.

        """
        return {}

    def set_params(self, **params: Any) -> "EstimatorLikeFity":
        """Set params.

        Parameters
        ----------
        **params : dict
            The parameters to set.

        Returns
        -------
        EstimatorLikeFity
            Estimator with set parameters.

        """
        return self


EstimatorLike = Union[EstimatorLikeFit1, EstimatorLikeFit2, EstimatorLikeFity]


@runtime_checkable
class TransformerLike(EstimatorLikeFit1, Protocol):
    """Class for transformer-like."""

    def fit(
        self,
        X: List[str],  # noqa: N803
        y: Optional[str] = None,
        **fit_params: Any,
    ) -> None:
        """Fit transformer.

        Parameters
        ----------
        X : list of str
            The features to use.
        y : str, optional
            The target to use (default None).
        **fit_params : dict
            Fit parameters.

        """
        pass

    def transform(self, X: DataLike) -> DataLike:  # noqa: N803
        """Transform.

        Parameters
        ----------
        X : DataLike
            The features to use.

        Returns
        -------
        DataLike
            The transformed data.

        """
        return X

    def fit_transform(
        self, X: DataLike, y: Optional[DataLike] = None  # noqa: N803
    ) -> DataLike:
        """Fit and transform.

        Parameters
        ----------
        X : DataLike
            The features to use.
        y : DataLike, optional
            The target to use (default None).

        Returns
        -------
        DataLike
            The fit and transformed object.

        """
        return X


@runtime_checkable
class ModelLike(EstimatorLikeFit1, Protocol):
    """Class for model-like."""

    classes_: np.ndarray

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
        return np.zeros(1)

    def score(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: DataLike,
        sample_weight: Optional[DataLike] = None,
    ) -> float:
        """Score the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.
        y : DataLike
            The true target values.
        sample_weight : DataLike, optional
            Sample weights to use when computing the score (default None).

        Returns
        -------
        float
            The score.

        """
        return 0.0


@runtime_checkable
class JuEstimatorLike(EstimatorLikeFit1, Protocol):
    """Class for juestimator-like."""

    def get_needed_types(self) -> ColumnTypes:
        """Get the column types needed by the estimator.

        Returns
        -------
        ColumnTypes
            The column types needed by the estimator.

        """
        return ColumnTypes("placeholder")

    def get_apply_to(self) -> ColumnTypes:
        """Get the column types the estimator applies to.

        Returns
        -------
        ColumnTypes
            The column types the estimator applies to.

        """
        return ColumnTypes("placeholder")


@runtime_checkable
class JuModelLike(ModelLike, Protocol):
    """Class for jumodel-like."""

    def get_needed_types(self) -> ColumnTypes:
        """Get the column types needed by the estimator.

        Returns
        -------
        ColumnTypes
            The column types needed by the estimator.

        """
        return ColumnTypes("placeholder")

    def get_apply_to(self) -> ColumnTypes:
        """Get the column types the estimator applies to.

        Returns
        -------
        ColumnTypes
            The column types the estimator applies to.

        """
        return ColumnTypes("placeholder")
