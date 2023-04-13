"""Provide scikit-learn-compatible transformers for confound removal."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas._typing import Scalar
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from ..base import ColumnTypesLike, JuTransformer, ensure_column_types
from ..utils import raise_error
from ..utils.typing import DataLike, ModelLike


class ConfoundRemover(JuTransformer):
    """Remove confounds from specific features.

    Transformer which removes the confounds from specific features by
    subtracting the predicted features given the confounds from the actual
    features.

    Parameters
    ----------
    apply_to : ColumnTypesLike, optional
        From which feature types ('X_types') to remove confounds.
        If not specified, 'apply_to' defaults to 'continuous'. To apply
        confound removal to all features, you can use the '*' regular
        expression syntax.
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
    keep_confounds : bool, optional
        Whether you want to return the confound together with the confound
        removed features, default is False.
    row_select_col_type : str or list of str or set of str or ColumnTypes
        The column types needed to select rows (default is None)
    row_select_vals : str, int, bool or list of str, int, bool
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None)
    """

    def __init__(
        self,
        apply_to: ColumnTypesLike = "continuous",
        model_confound: Optional[ModelLike] = None,
        confounds: ColumnTypesLike = "confound",
        threshold: Optional[float] = None,
        keep_confounds: bool = False,
        row_select_col_type:  Optional[ColumnTypesLike] = None,
        row_select_vals:  Optional[Union[str, int, List, bool]] = None,
    ):
        if model_confound is None:
            model_confound = LinearRegression()  # type: ignore
        self.model_confound = model_confound
        self.confounds = ensure_column_types(confounds)
        self.threshold = threshold
        self.keep_confounds = keep_confounds
        super().__init__(
            apply_to=apply_to,
            needed_types=confounds,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals
        )

    def _fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None
    ) -> "ConfoundRemover":
        """Fit ConfoundRemover.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values.

        Returns
        -------
        ConfoundRemover:
            The fitted transformer.
        """
        df_X, ser_confound = self._split_into_X_confound(X)
        self.feature_names_in_ = list(X.columns)
        if self.keep_confounds:
            self.support_mask_ = pd.Series(True, index=X.columns, dtype=bool)
        else:
            self.support_mask_ = pd.Series(False, index=X.columns, dtype=bool)
            output_X = self._add_backed_filtered(X, df_X)
            self.support_mask_[output_X.columns] = True
        self.support_mask_ = self.support_mask_.values

        def fit_confound_models(X: Scalar) -> ModelLike:
            _model = clone(self.model_confound)
            _model.fit(ser_confound.values, X)  # type: ignore
            return _model  # type: ignore

        self.models_confound_ = df_X.apply(
            fit_confound_models, axis=0, result_type="reduce"  # type: ignore
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove confounds from data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be deconfounded.

        Returns
        -------
        out : pd.DataFrame
            Deconfounded data.
        """
        df_X, df_confounds = self._split_into_X_confound(X)
        df_X_prediction = pd.DataFrame(
            [
                model.predict(df_confounds.values)
                for model in self.models_confound_.values
            ],
            index=df_X.columns,
            columns=df_X.index,
        ).T
        residuals = df_X - df_X_prediction
        df_out = self._apply_threshold(residuals)
        df_out = self._add_backed_filtered(X, df_out)

        if self.keep_confounds:
            df_out = df_out.reindex(columns=X.columns)
        else:
            df_out = df_out.reindex(
                columns=X.drop(columns=df_confounds.columns).columns
            )

        return df_out

    def get_support(
        self, indices: bool = False
    ) -> Union[ArrayLike, pd.Series]:
        """Get the support mask.

        Parameters
        ----------
        indices : bool
            If true, return indices.

        Returns
        -------
        support_mask : numpy.array
            The support mask
        """
        if indices:
            return np.arange(len(self.support_mask_))[
                self.support_mask_
            ]  # type: ignore
        else:
            return self.support_mask_  # type: ignore

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get names of features to be returned.

        Parameters
        ----------
        input_features : None
            Parameter to ensure scikit-learn compatibility. It is not used by
            the method.

        Returns
        -------
        list
            Names of features to be kept in the output pd.DataFrame.
        """
        out = self.feature_names_in_
        if self.keep_confounds is False:
            out = [
                feat
                for feat in self.feature_names_in_
                if feat not in self.detected_confounds_
            ]
        return out  # type: ignore

    def _split_into_X_confound(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the original X into the features (X) and confounds.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe including features and confounds.

        Returns
        -------
        df_X : pd.DataFrame
            DataFrame containing only features.
        df_confounds : pd.DataFrame
            DataFrame containing only confounds.
        """
        if not isinstance(X, pd.DataFrame):
            raise_error("ConfoundRemover only supports DataFrames as X")

        try:
            self.detected_confounds_ = self.confounds.to_type_selector()(X)
        except ValueError:
            raise_error(
                "No confound was found using the regex:"
                f"{self.confounds} in   the columns {X.columns}"
            )
        df_confounds = X.loc[:, self.detected_confounds_]
        df_X = self.filter_columns(X.drop(columns=self.detected_confounds_))

        return df_X, df_confounds

    def _apply_threshold(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """Round residuals to 0.

        If residuals are smaller than the absolute threshold specified during
        initialisation of the ConfoundRemover, residuals are rounded
        down to 0. This is done to prevent correlated rounding errors.

        Parameters
        ----------
        residuals : pd.DataFrame
            DataFrame containing the residuals after confound removal.

        Returns
        -------
        pd.DataFrame
            DataFrame containing residuals after rounding down to 0 if they are
            below the threshold.
        """
        if self.threshold is not None:
            # Accounting for correlated rounding errors for very small
            # residuals
            residuals = residuals.applymap(
                lambda x: 0 if abs(x) <= self.threshold else x
            )
        return residuals
