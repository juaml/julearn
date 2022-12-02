# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)
from sklearn.linear_model import LinearRegression

from .. utils import raise_error
from .. base import JuTransformer


class DataFrameConfoundRemover(JuTransformer):
    def __init__(
        self,
        apply_to=None,
        model_confound=None,
        confounds="confound",
        threshold=None,
        keep_confounds=False,
    ):
        """Transformer which can use pd.DataFrames and remove the confounds
        from the features by subtracting the predicted features
        given the confounds from the actual features.

        Parameters
        ----------
        apply_to : str or list of str
            Which columns should be confound removed.
        model_confound : obj
            Sklearn compatible model used to predict all features independently
            using the confounds as features. The predictions of these models
            are then subtracted from each feature, defaults to
            LinearRegression().
        confounds : list(str) | str
            A string representing a regular expression by which the confounds
            can be detected from the column names.
            You can use the exact column names or another regex.
            The default follows the naming convention inside of julearn:
            '.*__:type:__*.'
        threshold : float | None
            All residual values after confound removal which fall under the
            threshold will be set to 0.None (default) means that no threshold
            will be applied.
        keep_confounds : bool, optional
            Whether you want to return the confound together with the confound
            removed features, default is False
        """
        if model_confound is None:
            model_confound = LinearRegression()
        self.apply_to = apply_to
        self.model_confound = model_confound
        self.confounds = confounds
        self.threshold = threshold
        self.keep_confounds = keep_confounds

    def fit(self, X, y=None):
        """Fit confound remover

        Parameters
        ----------
        X : pandas.DataFrame
            Training data
        y : pandas.Series | None
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
        """Removes confounds from data

        Parameters
        ----------
        X : pandas.DataFrame
            Data to be deconfounded

        Returns
        -------
        out : pandas.DataFrame
            Data without confounds
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
            print(df_out.columns)
            print("conf: ", df_confounds.columns)
            print("all: ", X.columns)
            df_out = df_out.reindex(columns=X.columns)
        else:
            df_out = df_out.reindex(
                columns=X.drop(columns=df_confounds.columns).columns
            )

        return df_out

    def get_support(self, indices=False):
        """Get the support mask

        Parameters
        ----------
        indices : bool
            If true, return indexes

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
        print("here")
        print(self.get_feature_names_in_)
        print(self.detected_confounds_)
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
        """splits the original X input into the features (X) and confounds."""
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
        """Rounds residuals to 0 when smaller than
        the previously described absolute threshold.
        This is done to prevent correlated rounding errors
        """
        if self.threshold is not None:
            # Accounting for correlated rounding errors for very small
            # residuals
            residuals = residuals.applymap(
                lambda x: 0 if abs(x) <= self.threshold else x
            )
        return residuals


class TargetConfoundRemover(
    BaseEstimator, TransformerMixin,
):
    def __init__(
        self,
        model_confound=None,
        confounds=".*__:type:__confound",
        threshold=None,
    ):
        """Transformer which can use pd.DataFrames and remove the confounds
        from the target by subtracting the predicted target
        given the confounds from the actual target.

        Attributes
        ----------
        model_confound : object
            Model used to predict the target using the confounds as
            features. The predictions of these models are then subtracted
            from the actual target, default is None. Meaning the use of
            a LinearRegression.
        confounds : list(str) | str
            A string representing a regular expression by which the confounds
            can be detected from the column names.
        threshold : float | None
            All residual values after confound removal which fall under the
            threshold will be set to 0. None means that no threshold will be
            applied.
        """
        self.model_confound = model_confound
        self.confounds = confounds
        self.threshold = threshold
        self._confound_remover = DataFrameConfoundRemover(
            model_confound=self.model_confound,
            confounds=self.confounds,
            threshold=self.threshold,
        )

    def fit(self, X, y):

        Xy = X.copy()
        Xy["y"] = y.copy()

        self._confound_remover.fit(X=Xy)

    def transform(self, X, y):
        Xy = X.copy()
        Xy["y"] = y.copy()
        return self._confound_remover.transform(Xy)["y"]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
