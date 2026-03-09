"""Logit with L1 penalty, fit with largest lambda keeping ≥1 nonzero coef."""

# Authors: Kaustub R. Patil <k.patil@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)

from ..utils.typing import ArrayLike, DataLike


class LogisticL1LambdaMax(BaseEstimator, ClassifierMixin):
    """Logistic regression (L1 penalty), largest lambda keeping ≥1 coef != 0.

    Binary logistic regression with L1 penalty, fitted at the maximally
    regularizing lambda (alpha) that still allows at least one non-zero
    coefficient.

    For logistic loss:
        lambda_max = || (1/n) * X^T (0.5 - y) ||_inf

    sklearn uses C = 1 / lambda, so:
        C_fit = 1 / (lambda_max * (1 - eps))

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    eps : float, default=1e-6
        Use lambda_fit = lambda_max * (1 - eps).
    max_iter : int, default=1000
        Maximum number of iterations for the underlying logistic regression.
    solver : {'liblinear', 'saga'}, default='liblinear'
        Solver to use in the underlying logistic regression.

    Warnings
    --------
    The calculation of lambda_max assumes that the data has been z-scored or
    standardized before. In julearn, this means that a "zscore" preprocess step
    shoule be used as the previous step in the pipeline.

    """

    def __init__(
        self,
        fit_intercept: bool = True,
        eps: float = 1e-6,
        max_iter: int = 1000,
        solver: Literal["liblinear", "saga"] = "liblinear",
    ):
        if solver not in ["liblinear", "saga"]:
            raise ValueError(
                "solver must be 'liblinear' or 'saga' for L1 penalty."
            )

        if not (0.0 <= eps < 1.0):
            raise ValueError("eps must be in [0, 1).")

        self.fit_intercept = fit_intercept
        self.eps = eps
        self.solver = solver
        self.max_iter = max_iter

    def _compute_lambda_max(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> float:
        n = X.shape[0]

        if self.fit_intercept:
            X = X - X.mean(axis=0)

        grad0 = (X.T @ (0.5 - y)) / n
        return float(np.max(np.abs(grad0)))

    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
    ) -> "LogisticL1LambdaMax":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.

        Returns
        -------
        LogisticL1LambdaMax
            The fitted model.

        """
        X, y = check_X_y(X, y)

        if type_of_target(y) != "binary":
            raise ValueError(
                "This estimator supports binary classification only."
            )

        if isinstance(X, pd.DataFrame):
            npX = X.values
        else:
            npX = X
        if isinstance(y, (pd.DataFrame, pd.Series)):
            npy = y.values
        else:
            npy = y

        if not np.issubdtype(npy.dtype, np.number):
            raise ValueError(
                "y must be numeric (0/1) for lambda_max calculation. "
                "If you are using julearn's :func:.run_cross_validation "
                "you can solve this issue by specifying the `pos_labels` "
                "argument."
            )

        self.lambda_max_ = self._compute_lambda_max(npX, npy)  # type: ignore

        self.lambda_fit_ = self.lambda_max_ * (1.0 - self.eps)
        self.C_fit_ = 1.0 / self.lambda_fit_

        self.model_ = LogisticRegression(
            penalty="l1",
            solver=self.solver,  # type: ignore
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
        )

        self.model_.set_params(C=self.C_fit_)

        self.model_.fit(npX, npy)

        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: DataLike) -> ArrayLike:  # noqa: N803
        """Predict using the model.

        Parameters
        ----------
        X : DataLike
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.

        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: DataLike) -> ArrayLike:  # noqa: N803
        """Predict probability estimates for the test data.

        Parameters
        ----------
        X : DataLike
            The data to predict on.

        Returns
        -------
        ArrayLike
            The predicted probabilities.

        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict_proba(X)

    def decision_function(self, X: DataLike) -> ArrayLike:  # noqa: N803
        """Compute the decision function for the test data.

        Parameters
        ----------
        X : DataLike
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.

        """

        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.decision_function(X)

    def score(self, X: DataLike, y: DataLike) -> float:  # noqa: N803
        """Score the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.
        y : DataLike
            The true target values.

        Returns
        -------
        float
            The score.

        """

        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.score(X, y)  # type: ignore
