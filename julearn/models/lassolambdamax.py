"""Lasso but fit with largest lambda keeping ≥1 nonzero coef."""

# Authors: Kaustub R. Patil <k.patil@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing import Optional, Union


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
    standardize : bool, default=True
        Whether to standardize X via StandardScaler before fitting.
        (Recommended for Lasso unless you have good reasons not to.)
    fit_intercept : bool, default=True
        Whether to fit an intercept in the underlying Lasso.
    eps : float, default=1e-6
        Use alpha_fit = alpha_max * (1 - eps). Must be in [0, 1).
        eps=0 uses alpha_max exactly (often yields all-zero coefficients).

    """

    def __init__(
        self,
        standardize: bool = True,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        eps: float = 1e-6,
        selection: str = "cyclic",
    ):
        self.standardize = standardize
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

    def _compute_alpha_max(self, X, y):
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

        base_lasso = Lasso(
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )

        if self.standardize:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            Xs = self.scaler_.fit_transform(X)
            self.alpha_max_ = self._compute_alpha_max(Xs, y)
            self.alpha_fit_ = self.alpha_max_ * (1.0 - self.eps)

            self.lasso_ = clone(base_lasso)
            self.lasso_.set_params(alpha=self.alpha_fit_)
            self.model_ = Pipeline(
                [("scaler", self.scaler_), ("lasso", self.lasso_)]
            )
            self.model_.fit(X, y)

            # convenience attributes (mirror sklearn linear models)
            self.coef_ = self.model_.named_steps["lasso"].coef_
            self.intercept_ = self.model_.named_steps["lasso"].intercept_
        else:
            self.alpha_max_ = self._compute_alpha_max(X, y)
            self.alpha_fit_ = self.alpha_max_ * (1.0 - self.eps)

            self.model_ = clone(base_lasso)
            self.model_.set_params(alpha=self.alpha_fit_)
            self.model_.fit(X, y)

            self.coef_ = self.model_.coef_
            self.intercept_ = getattr(self.model_, "intercept_", 0.0)

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict(X)

    def score(self, X, y):
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.score(X, y)

    def get_params(self, deep=True):
        return super().get_params(deep=deep)

    def set_params(self, **params):
        return super().set_params(**params)
