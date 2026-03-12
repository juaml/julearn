"""Lasso but fit with largest lambda keeping ≥1 nonzero coef."""

# Authors: Kaustub R. Patil <k.patil@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..utils.typing import ArrayLike, DataLike


class LassoLambdaMax(BaseEstimator, RegressorMixin):
    """LASSO regression, largest lambda keeping ≥1 coef != 0.

    Fit Lasso at the maximally regularizing lambda (alpha) that still yields
    at least one non-zero coefficient: alpha_fit = alpha_max * (1 - eps).

    Lasso objective in scikit-learn:
        (1/(2n)) * ||y - Xw||^2_2 + alpha * ||w||_1

    alpha_max (with intercept handling) is:
        max_j | (Xc^T yc)_j | / n
    where Xc and yc are centered if fit_intercept=True.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept in the underlying Lasso.
    eps : float, default=1e-6
        Use alpha_fit = alpha_max * (1 - eps). Must be in [0, 1).
        eps=0 uses alpha_max exactly (often yields all-zero coefficients).

    max_iter : int, default=1000
        Maximum number of iterations for the underlying Lasso.
    tol : float, default=1e-4
        Tolerance for the optimization in the underlying Lasso.
    positive : bool, default=False
        If True, forces coefficients to be positive in the underlying Lasso.
    random_state : int, RandomState instance or None, default=None
        Random state for the underlying Lasso when selection='random'.
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by their coordinate.
        This (setting selection to 'random') often leads to significantly
        faster convergence especially when tol is higher than 1e-4.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        eps: float = 1e-6,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        self.fit_intercept = fit_intercept
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        if not (0.0 <= self.eps < 1.0):
            raise ValueError(
                "eps must be in [0, 1). Use a small value like 1e-6."
            )

    def _compute_alpha_max(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> float:
        n = X.shape[0]
        if self.fit_intercept:
            # center explicitly for alpha_max derivation
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
            return float(np.max(np.abs(Xc.T @ yc)) / n)
        else:
            return float(np.max(np.abs(X.T @ y)) / n)

    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
    ) -> "LassoLambdaMax":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.

        Returns
        -------
        LassoLambdaMax
            The fitted model.

        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
        if isinstance(X, pd.DataFrame):
            npX = X.values
        else:
            npX = X
        if isinstance(y, (pd.DataFrame, pd.Series)):
            npy = y.values
        else:
            npy = y
        base_lasso = Lasso(
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )

        self.alpha_max_ = self._compute_alpha_max(npX, npy)
        self.alpha_fit_ = self.alpha_max_ * (1.0 - self.eps)

        self.model_ = clone(base_lasso)
        self.model_.set_params(alpha=self.alpha_fit_)
        self.model_.fit(X, y)

        self.coef_ = self.model_.coef_
        self.intercept_ = getattr(self.model_, "intercept_", 0.0)

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
