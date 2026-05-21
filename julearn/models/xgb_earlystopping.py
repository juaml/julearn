# Classifier wrapper for XGBoost with cross-validated early stopping

import copy
import inspect
from typing import Any, Dict, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from xgboost import XGBClassifier, XGBRegressor∏


sklearn.set_config(enable_metadata_routing=True)


class _BaseXGBCVEarlyStopping(BaseEstimator):
    def __init__(self, base_estimator, test_size, early_stopping_rounds, **kwargs):
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
        # groups = kwargs.get("groups", None)
        if groups is not None:
            print("Using groups for early stopping CV.")
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
        else:
            print("Not using groups for early stopping CV.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
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
        self.classes_ = self._model.classes_

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


class XGBRegressorCVEarlyStopping(
    _BaseXGBCVEarlyStopping, ClassifierMixin
):

    def __init__(self, test_size, early_stopping_rounds, **kwargs):
        super().__init__(
            base_estimator=XGBRegressor,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )


class XGBClassifierCVEarlyStopping(
    _BaseXGBCVEarlyStopping, ClassifierMixin
):

    def __init__(self, test_size, early_stopping_rounds, **kwargs):
        super().__init__(
            base_estimator=XGBClassifier,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )

    def predict_proba(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)