"""Provide target confound removal."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import typing
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from ...base.column_types import ColumnTypesLike, ensure_column_types
from ...utils.typing import ModelLike
from .ju_target_transformer import JuTargetTransformer


class TargetConfoundRemover(JuTargetTransformer):
    """Remove confounds from the target.

    Parameters
    ----------
    model_confound : ModelLike, optional
        Sklearn compatible model used to predict specified features
        independently using the confounds as features. The predictions of
        these models are then subtracted from each of the specified
        features, defaults to LinearRegression().
    confounds : str or list of str, optional
        The name of the 'confounds' type(s), i.e. which column type(s)
        represents the confounds. By default this is set to 'confounds'.
    threshold : float, optional
        All residual values after confound removal which fall under the
        threshold will be set to 0. None (default) means that no threshold
        will be applied.

    """

    def __init__(
        self,
        model_confound: Optional[ModelLike] = None,
        confounds: ColumnTypesLike = "confound",
        threshold: Optional[float] = None,
    ):
        if model_confound is None:
            model_confound = LinearRegression()  # type: ignore
        self.model_confound = model_confound
        self.confounds = ensure_column_types(confounds)
        self.threshold = threshold

    @property
    def needed_types(self) -> ColumnTypesLike:
        """Get the needed column types."""
        return self.confounds

    def fit(
        self, X: pd.DataFrame, y: pd.Series  # noqa: N803
    ) -> "TargetConfoundRemover":
        """Fit ConfoundRemover.

        Parameters
        ----------
        X : pd.DataFrame
            Training data for the confound remover.
        y : pd.Series
            Training target values.

        Returns
        -------
        TargetConfoundRemover
            The fitted target confound remover.

        """
        self.model_confounds_ = clone(self.model_confound)
        self.detected_confounds_ = self.confounds.to_type_selector()(X)
        X_confounds = X.loc[:, self.detected_confounds_]
        self.model_confounds_.fit(X_confounds.values, y)  # type: ignore
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.Series  # noqa: N803
    ) -> pd.Series:
        """Remove confounds from the target.

        Parameters
        ----------
        X : pd.DataFrame
            Testing data for the confound remover.
        y : pd.Series
            Target values.

        Returns
        -------
        pd.Series
            The target with confounds removed.

        """
        X_confounds = X.loc[:, self.detected_confounds_]
        y_pred = self.model_confounds_.predict(  # type: ignore
            X_confounds.values
        )
        y_pred = typing.cast(pd.Series, y_pred)
        residuals = y - y_pred
        if self.threshold is not None:
            residuals[np.abs(residuals) < self.threshold] = 0
        return residuals
