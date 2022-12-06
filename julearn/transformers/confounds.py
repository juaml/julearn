"""Provide sklearn compatible transformers for confound removal."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from .. utils import raise_error
from .. base import JuTransformer


class DataFrameConfoundRemover(JuTransformer):
    """Remove confounds from specific features.

    Transformer which transforms pd.DataFrames and removes the confounds
    from specific features by subtracting the predicted features
    given the confounds from the actual features.

    Parameters
    ----------
    apply_to : str or list of str, optional
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
    """

    def __init__(
        self,
        apply_to=None,
        model_confound=None,
        confounds="confound",
        threshold=None,
        keep_confounds=False,
    ):
        if model_confound is None:
            model_confound = LinearRegression()
        self.apply_to = apply_to
        self.model_confound = model_confound
        self.confounds = confounds
        self.threshold = threshold
        self.keep_confounds = keep_confounds

    def fit(self, X, y=None):
        """Fit DataFrameConfoundRemover.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.apply_to = ("continuous"
                         if self.apply_to is None
                         else self.apply_to)
        self.apply_to = self._ensure_apply_to()
        self.confounds = self._ensure_column_types(self.confounds)
        self.needed_types = deepcopy(self.apply_to).add(self.confounds)

        df_X, ser_confound = self._split_into_X_confound(X)
        self.feature_names_in_ = list(X.columns)
        if self.keep_confounds:
            self.support_mask_ = pd.Series(True, index=X.columns, dtype=bool)
        else:
            self.support_mask_ = pd.Series(False, index=X.columns, dtype=bool)
            output_X = self._add_backed_filtered(X, df_X)
            self.support_mask_[output_X.columns] = True
        self.support_mask_ = self.support_mask_.values

        def fit_confound_models(X):
            _model = clone(self.model_confound)
            _model.fit(ser_confound.values, X)
            return _model

        self.models_confound_ = df_X.apply(
            fit_confound_models, axis=0, result_type="reduce"
        )
        return self

    def transform(self, X):
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

    def get_support(self, indices=False):
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
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_

    def get_feature_names_out(self, input_features=None):
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
        return (
            self.feature_names_in_
            if self.keep_confounds is True
            else [
                feat
                for feat in self.feature_names_in_
                if feat not in self.detected_confounds_
            ]
        )

    def _split_into_X_confound(self, X):
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
            raise_error(
                "DataFrameConfoundRemover only supports DataFrames as X"
            )

        try:
            self.detected_confounds_ = self.confounds.to_type_selector()(X)
        except ValueError:
            raise_error(
                "No confound was found using the regex:"
                f"{self.confounds} in   the columns {X.columns}"
            )
        df_confounds = X.loc[:, self.detected_confounds_]
        df_X = self.filter_columns(
            X
            .drop(columns=self.detected_confounds_)
        )

        return df_X, df_confounds

    def _apply_threshold(self, residuals):
        """Round residuals to 0.

        If residuals are smaller than the absolute threshold specified during
        initialisation of the DataFrameConfoundRemover, residuals are rounded
        down to 0. This is done to prevent correlated rounding errors.

        Parameters
        ----------
        residuals : pd.DataFrame
            DataFrame containing the residuals after confound removal.

        Returns
        -------
        residuals : pd.DataFrame
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

    def get_needed_types(self):
        if hasattr(self, "needed_types"):
            return self.needed_types

        apply_to = ("continuous"
                    if self.apply_to is None
                    else self.apply_to)
        apply_to = self._ensure_column_types(apply_to)
        confounds = self._ensure_column_types(self.confounds)
        print(confounds._column_types, apply_to._column_types)
        return apply_to.add(confounds._column_types)
