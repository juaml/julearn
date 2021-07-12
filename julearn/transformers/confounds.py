# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression

from . target import BaseTargetTransformer
from .. utils.logging import raise_error, warn
from .. utils.array import safe_select, safe_set


class BaseConfoundRemover(TransformerMixin, BaseEstimator):
    @abstractmethod
    def fit(self, X, y=None, n_confounds=0):
        pass

    @abstractmethod
    def transform(self, X, n_confounds=0):
        pass

    @abstractmethod
    def fit_transform(self, X, y=None, n_confounds=0, **fit_params):
        self.fit(X, n_confounds=n_confounds, **fit_params)
        return self.transform(X)

    @abstractmethod
    def will_drop_confounds(self):
        pass


class ConfoundRemover(BaseConfoundRemover):
    def __init__(self, model_confound=None, threshold=None,
                 drop_confounds=True):
        """Transformer to remove n_confounds from the features.
        Subtracts the predicted features given confounds from features.
        Resulting residuals can be thresholded in case residuals are so small
        that rouding error can be informative. 

        Parameters
        ----------
        model_confound : obj
            Scikit-learn compatible model used to predict all features
            independently using the confounds as features.
            The predictions of these models
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
        self.drop_confounds = drop_confounds

    def fit(self, X, y=None, n_confounds=0, apply_to=None):
        """Fit confound remover

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.
        y : pandas.Series | np.ndarray
            Target values.
        n_confounds : int
            Number of confounds inside of X.
            The last n_confounds columns in X will be used as confounds.
        apply_to : array-like of int, slice, array-like of bool
            apply_to will be used to index the features inside of X
            (excluding the confound). The selected features will be confound
            removed. The keep as they are.

        """
        self.n_confounds_ = n_confounds
        self.apply_to_ = apply_to
        if self.n_confounds_ <= 0:
            warn(
                'Number of confounds is 0, confound removal will not have any '
                'effect')
            return self
        confounds = safe_select(X, slice(-self.n_confounds_, None))

        def fit_confound_models(t_X):
            _model = clone(self.model_confound)
            _model.fit(confounds, t_X)
            return _model

        t_X = safe_select(X, slice(None, -self.n_confounds_))
        if self.apply_to_ is not None:
            t_X = safe_select(t_X, self.apply_to_)

        self.models_confound_ = []
        for i_X in range(t_X.shape[1]):
            t_X = safe_select(X, i_X)
            self.models_confound_.append(fit_confound_models(t_X))

        return self

    def transform(self, X):
        """Removes confounds from X.

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.

        Returns
        -------
        out : np.ndarray
            Deconfounded X.
        """
        if self.n_confounds_ <= 0:
            return X
        confounds = safe_select(X, slice(-self.n_confounds_, None))
        X = safe_select(X, slice(None, -self.n_confounds_))
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.copy()
        idx = np.arange(0, X.shape[1])
        if self.apply_to_ is not None:
            idx = idx[self.apply_to_]
        for i_model, model in enumerate(self.models_confound_):
            t_idx = idx[i_model]
            t_pred = model.predict(confounds)
            X_res = X[:, t_idx] - t_pred
            if self.threshold is not None:
                X_res[np.abs(X_res) < self.threshold] = 0
            X[:, t_idx] = X_res

        if self.drop_confounds is not True:
            X = np.c_[X, confounds]
        return X

    def will_drop_confounds(self):
        return self.drop_confounds


class TargetConfoundRemover(TransformerMixin, BaseTargetTransformer):

    def __init__(self, model_confound=None, threshold=None):
        """Transformer to remove n_confounds from the target.
        Subtracts the predicted target given confounds from target.
        Resulting residuals can be thresholded in case residuals are so small
        that rouding error can be informative. 

        Attributes
        ----------
        model_confound : object
            Scikit-learn compatible model used to predict the target
            using the confounds as features.
            The predictions of these models are then subtracted
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

    def fit(self, X, y, n_confounds):
        """Fit confound remover

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.
        y : pandas.Series | np.ndarray
            Target values.
        n_confounds : int
            Number of confounds inside of X.
            The last n_confounds columns in X will be used as confounds.
        """
        if n_confounds <= 0:
            raise_error('Confound must be set for confound removal to happen')
        self.n_confounds_ = n_confounds
        confounds = safe_select(X, slice(-self.n_confounds_, None))
        yConf = np.c_[y, confounds]
        self._confound_remover.fit(X=yConf, n_confounds=self.n_confounds_)

    def transform(self, X, y):
        """Removes confounds from target.

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.
        y : pandas.Series | np.ndarray
                Target values.

        Returns
        -------
        out : np.ndarray
            Deconfounded target.
        """
        confounds = safe_select(X, slice(-self.n_confounds_, None))
        yConf = np.c_[y, confounds]
        n_y = self._confound_remover.transform(
            X=yConf).squeeze()  # type:ignore
        return n_y

    def fit_transform(self, X, y, n_confounds):
        self.fit(X, y, n_confounds=n_confounds)
        return self.transform(X, y)
