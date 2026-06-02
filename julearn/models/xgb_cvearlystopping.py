"""Classifier wrapper for XGBoost with cross-validated early stopping."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from julearn.utils.typing import DataLike


sklearn.set_config(enable_metadata_routing=True)


class _BaseXGBCVEarlyStopping(BaseEstimator):
    """Base class for XGBoost with cross-validated early stopping.

    A wrapper for XGBoost that performs early stopping using a
    cross-validation split of the data. The model is first trained on a
    training set with early stopping based on a validation set, and then refit
    on the full data using the best number of iterations found.

    Parameters
    ----------
    base_estimator : class
        The base XGBoost estimator class to use (e.g. XGBRegressor or
        XGBClassifier).
    test_size : float or int or None
        The proportion of the data to use as the validation set for early
        stopping. If groups is used on `fit`, this parameter refers to the
        number of groups, otherwise it refers to the number of samples. If
        float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int,
        represents the absolute number. If None, the value is
        set to the complement of the train size. If train_size is also None,
        it will be set to 0.25 in the case of non-grouped data or 0.2 for
        grouped data (scikit-learn's defaults for `train_test_split` and
        `GroupShuffleSplit`).
    early_stopping_rounds : int
        The number of rounds to use for early stopping.
    **kwargs : dict
        Extra keyword arguments to pass to the XGBoost estimator.

    """

    def __init__(
        self,
        base_estimator: XGBRegressor | XGBClassifier,
        test_size: float | int | None,
        early_stopping_rounds: int,
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

    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        groups: DataLike | None = None,
    ) -> "_BaseXGBCVEarlyStopping":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.
        groups : DataLike or None
            The group labels for the samples used while splitting the dataset
            into train/test set for early stopping. If None, standard
            train/test split is used, by default None.

        Returns
        -------
        _BaseXGBCVEarlyStopping
            The fitted model.

        """
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

            if isinstance(y, pd.Series):
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

        # Create a model with the max iterations set as the best epochs and
        # refit on full data
        t_kwargs = self._xgboost_kwargs.copy()
        self._best_iteration = model.best_iteration

        num_parallel_tree = model.get_params().get("num_parallel_tree")
        if num_parallel_tree is None:
            num_parallel_tree = 1
        n_classes = getattr(model, "n_classes_", 1)
        t_kwargs["n_estimators"] = (
            (self._best_iteration + 1) * num_parallel_tree * n_classes
        )
        model = self.base_estimator(**t_kwargs)
        model.fit(X=X, y=y)
        self._model = model
        self._is_fitted = True

        return self

    def predict(self, X: DataLike) -> DataLike:  # noqa: N803
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
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict(X)

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the model is fitted.

        Returns
        -------
        bool
            True if the model is fitted, False otherwise.

        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def get_params(self, deep: bool = True) -> dict:
        """Get the parameters of the model.

        Parameters
        ----------
        deep : bool
            If True, will return the parameters for this model and
            contained subobjects that are estimators (default is True).

        Returns
        -------
        params : dict
            Parameter names mapped to their values.

        """
        params = {
            "test_size": self.test_size,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        params.update(self._xgboost_kwargs)
        return params

    def set_params(self, **params: Any) -> "_BaseXGBCVEarlyStopping":
        """Set the parameters of the model.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        _BaseXGBCVEarlyStopping
            The model with updated parameters.

        """
        for param, value in params.items():
            if param in ["test_size", "early_stopping_rounds"]:
                setattr(self, param, value)
            elif param == "random_state":
                self.random_state = value
                self._xgboost_kwargs["random_state"] = value
            else:
                self._xgboost_kwargs[param] = value
        return self


class XGBRegressorCVEarlyStopping(_BaseXGBCVEarlyStopping, RegressorMixin):
    """XGBRegressor with cross-validated early stopping.

    A wrapper for XGBoost that performs early stopping using a
    cross-validation split of the data. The model is first trained on a
    training set with early stopping based on a validation set, and then refit
    on the full data using the best number of iterations found.

    Parameters
    ----------
    test_size : float or int or None
        The proportion of the data to use as the validation set for early
        stopping. If groups is used on `fit`, this parameter refers to the
        number of groups, otherwise it refers to the number of samples. If
        float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int,
        represents the absolute number. If None, the value is
        set to the complement of the train size. If train_size is also None,
        it will be set to 0.25 in the case of non-grouped data or 0.2 for
        grouped data (scikit-learn's defaults for `train_test_split` and
        `GroupShuffleSplit`).
    early_stopping_rounds : int
        The number of rounds to use for early stopping.
    **kwargs : dict
        Extra keyword arguments to pass to the XGBRegressor.

    """

    def __init__(
        self,
        test_size: float | int | None,
        early_stopping_rounds: int,
        **kwargs: Any,
    ):
        super().__init__(
            base_estimator=XGBRegressor,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )


class XGBClassifierCVEarlyStopping(_BaseXGBCVEarlyStopping, ClassifierMixin):
    """XGBClassifier with cross-validated early stopping.

    A wrapper for XGBoost that performs early stopping using a
    cross-validation split of the data. The model is first trained on a
    training set with early stopping based on a validation set, and then refit
    on the full data using the best number of iterations found.

    Parameters
    ----------
    test_size : float or int or None
        The proportion of the data to use as the validation set for early
        stopping. If groups is used on `fit`, this parameter refers to the
        number of groups, otherwise it refers to the number of samples. If
        float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int,
        represents the absolute number. If None, the value is
        set to the complement of the train size. If train_size is also None,
        it will be set to 0.25 in the case of non-grouped data or 0.2 for
        grouped data (scikit-learn's defaults for `train_test_split` and
        `GroupShuffleSplit`).
    early_stopping_rounds : int
        The number of rounds to use for early stopping.
    **kwargs : dict
        Extra keyword arguments to pass to the XGBClassifier.

    """

    def __init__(
        self,
        test_size: float | int | None,
        early_stopping_rounds: int,
        **kwargs: Any,
    ):
        super().__init__(
            base_estimator=XGBClassifier,
            test_size=test_size,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        groups: DataLike | None = None,
    ) -> "XGBClassifierCVEarlyStopping":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.
        groups : DataLike or None
            The group labels for the samples used while splitting the dataset
            into train/test set for early stopping. If None, standard
            train/test split is used, by default None.

        Returns
        -------
        XGBClassifierCVEarlyStopping
            The fitted model.

        """
        self._label_encoder = None
        # Check if labels are strings and convert to integers if so, to avoid
        # issues with XGBoost
        if isinstance(
            y, pd.Series | np.ndarray | pd.arrays.StringArray
        ) and y.dtype in ["object", "string"]:
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)  # type: ignore
        super().fit(X, y, groups)
        self.classes_ = self._model.classes_  # type: ignore
        return self

    def predict(self, X: DataLike) -> DataLike:  # noqa: N803
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
        out = super().predict(X)
        if self._label_encoder is not None:
            out = self._label_encoder.inverse_transform(out)
        return out

    def predict_proba(self, X: DataLike) -> DataLike:  # noqa: N803
        """Predict probabilities using the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.

        """
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)
