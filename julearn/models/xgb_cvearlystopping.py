# Classifier wrapper for XGBoost with cross-validated early stopping

from typing import Self

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse._bsr import spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from xgboost import XGBClassifier, XGBRegressor


sklearn.set_config(enable_metadata_routing=True)


class _BaseXGBCVEarlyStopping(BaseEstimator):
    def __init__(
        self,
        base_estimator,
        test_size,
        early_stopping_rounds,
        **kwargs,
    ):
        self.test_size = test_size
        if early_stopping_rounds is None:
            raise ValueError(
                "early_stopping_rounds must be set for CV early stopping."
            )
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = kwargs.get("random_state", None)
        self.base_estimator = base_estimator
        self._xgboost_kwargs = kwargs
        if self._xgboost_kwargs is None:
            self._xgboost_kwargs = {}
        self._model = None
        self._is_fitted = False
        self.set_fit_request(groups=True)

    def fit(self, X, y, groups=None):
        if groups is not None:
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]

            if isinstance(X, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
            self._grouped_cv = True
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            self._grouped_cv = False
        # Build a first model
        model = self.base_estimator(
            early_stopping_rounds=self.early_stopping_rounds,
            **self._xgboost_kwargs,
        )
        model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])

        # Create a model with the max iterations set as the best epochs and refit on full data
        t_kwargs = self._xgboost_kwargs.copy()
        if hasattr(model, "best_ntree_limit"):
            t_kwargs["n_estimators"] = model.best_ntree_limit
        else:
            num_parallel_tree = model.get_params().get("num_parallel_tree")
            if num_parallel_tree is None:
                num_parallel_tree = 1
            t_kwargs["n_estimators"] = (
                model.best_iteration + 1
            ) * num_parallel_tree
        model = self.base_estimator(**t_kwargs)
        model.fit(X=X, y=y)
        self._model = model
        self._is_fitted = True

        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict(X)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted

    def get_params(self, deep=True):
        params = {
            "test_size": self.test_size,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        params.update(self._xgboost_kwargs)
        return params

    def set_params(self, **params) -> Self:
        for param, value in params.items():
            if param in ["test_size", "early_stopping_rounds"]:
                setattr(self, param, value)
            elif param == "random_state":
                self.random_state = value
                self._xgboost_kwargs["random_state"] = value
            else:
                self._xgboost_kwargs[param] = value
        return self


class XGBRegressorCVEarlyStopping(_BaseXGBCVEarlyStopping, ClassifierMixin):
    def __init__(self, test_size, early_stopping_rounds, **kwargs):
        super().__init__(
            base_estimator=XGBRegressor,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )


class XGBClassifierCVEarlyStopping(_BaseXGBCVEarlyStopping, ClassifierMixin):
    def __init__(self, test_size, early_stopping_rounds, **kwargs):
        super().__init__(
            base_estimator=XGBClassifier,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

    def fit(self, X, y, groups=None):
        self._label_encoder = None
        # Check if labels are strings and convert to integers if so, to avoid issues with XGBoost
        if isinstance(y, pd.Series) and y.dtype in ["object", "string"]:
            self._label_encoder = sklearn.preprocessing.LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        elif isinstance(y, np.ndarray) and y.dtype == "object":
            self._label_encoder = sklearn.preprocessing.LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        out = super().fit(X, y, groups)
        self.classes_ = self._model.classes_
        return out

    def predict(self, X):
        out = super().predict(X)
        if self._label_encoder is not None:
            out = self._label_encoder.inverse_transform(out)
        return out

    def predict_proba(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)

    def score(self, X, y, sample_weight=None) -> float:
        if self._label_encoder is not None:
            y = self._label_encoder.transform(y)
        return super().score(X, y, sample_weight)
