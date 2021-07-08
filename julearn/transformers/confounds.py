# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from abc import abstractmethod
from julearn.utils.logging import raise_error
from julearn.utils.array import ensure_2d
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression


class BaseConfoundRemover(TransformerMixin, BaseEstimator):
    @abstractmethod
    def fit(self, X, y=None, confounds=None):
        pass

    @abstractmethod
    def transform(self, X, confounds=None):
        pass

    @abstractmethod
    def fit_transform(self, X, y=None, confounds=None, **fit_params):
        self.fit(X, confounds=confounds, **fit_params)
        return self.transform(X, confounds=confounds)


class ConfoundRemover(BaseConfoundRemover):
    def __init__(self, model_confound=None, threshold=None):
        """Transformer which can use pd.DataFrames and remove the confounds
        from the features by subtracting the predicted features
        given the confounds from the actual features.

        Parameters
        ----------
        model_confound : obj
            Sklearn compatible model used to predict all features independently
            using the confounds as features. The predictions of these models
            are then subtracted from each feature, defaults to
            LinearRegression().
        threshold : float | None
            All residual values after confound removal which fall under the
            threshold will be set to 0. None (default) means that no threshold
            will be applied.
        """
        if model_confound is None:
            model_confound = LinearRegression()
        self.model_confound = model_confound
        self.threshold = threshold

    def fit(self, X, y=None, confounds=None):
        """Fit confound remover

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data
        y : pandas.Series | np.ndarray
            Target values.
        confounds : pandas.DataFrame | np.ndarray
            confounds used to deconfound X

        """
        if confounds is None:
            raise_error('Confound must be set for confound removal to happen')
        confounds = ensure_2d(confounds)

        def fit_confound_models(t_X):
            _model = clone(self.model_confound)
            _model.fit(confounds, t_X)
            return _model

        self.models_confound_ = []
        for i_X in range(X.shape[1]):
            if isinstance(X, pd.DataFrame):
                t_X = X.iloc[:, i_X]
            else:
                t_X = X[:, i_X]
            self.models_confound_.append(fit_confound_models(t_X))

        return self

    def transform(self, X, confounds):  # Todo: fix
        """Removes confounds from data

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Data to be deconfounded
        confounds : pandas.DataFrame | np.ndarray
            confounds used to deconfound X

        Returns
        -------
        out : np.ndarray
            Confound removed X
        """
        if confounds is None:
            raise_error('Confound must be set for confound removal to happen')
        confounds = ensure_2d(confounds)

        X_pred = np.zeros_like(X, dtype=np.float)
        for i_model, model in enumerate(self.models_confound_):
            X_pred[:, i_model] = model.predict(confounds)

        residuals = X - X_pred
        X_removed = self._apply_threshold(residuals)

        return X_removed

    def _apply_threshold(self, residuals):
        """Rounds residuals to 0 when smaller than
        the previously described absolute threshold.
        This is done to prevent correlated rounding errors
        """
        if self.threshold is not None:
            # Accounting for correlated rounding errors for very small
            # residuals
            residuals[np.abs(residuals) <= self.threshold] = 0
        return residuals


class TargetConfoundRemover(TransformerMixin):

    def __init__(self, model_confound=None, threshold=None):
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
        threshold : float | None
            All residual values after confound removal which fall under the
            threshold will be set to 0. None means that no threshold will be
            applied.
        """
        self.model_confound = model_confound
        self.threshold = threshold
        self._confound_remover = ConfoundRemover(
            model_confound=self.model_confound,
            threshold=self.threshold)

    def fit(self, X, y, confounds):
        if confounds is None:
            raise_error('Confound must be set for confound removal to happen')
        Xy = ensure_2d(y)
        self._confound_remover.fit(X=Xy, confounds=confounds)

    def transform(self, X, y, confounds):
        if confounds is None:
            raise_error('Confound must be set for confound removal to happen')
        Xy = ensure_2d(y)
        return self._confound_remover.transform(
            X=Xy, confounds=confounds).squeeze()

    def fit_transform(self, X, y, confounds):
        self.fit(X, y, confounds=confounds)
        return self.transform(X, y, confounds=confounds)
