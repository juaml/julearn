"""Lasso but fit with largest lambda keeping ≥1 nonzero coef."""

# Authors: Kaustub R. Patil <k.patil@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Lasso
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..utils import logger
from ..utils.typing import ArrayLike, DataLike


def _generate_alpha_sequence(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    step: float,
    fit_intercept: bool,
    sample_weight: Optional[np.ndarray] = None,
    max_alpha_iter: int = 100,
) -> float:
    """Compute the maximum alpha for Lasso regression.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    y : np.ndarray
        The target data.
    step : float
        The step size for the sequence of alphas to test. Will start at
        alpha_max * (1 - step) and decrease until alpha = 0 or max_alpha_iter
        is reached.
    fit_intercept : bool
        Whether to fit an intercept in the underlying Lasso.
    sample_weight : np.ndarray, optional
        Sample weights. If None, all samples are given equal weight.
    max_alpha_iter : int, default=100
        Maximum number of alphas to generate. This is a safeguard to prevent
        infinite loops in case of numerical issues.

    Returns
    -------
    float
        The alpha.

    """
    alpha_max = _alpha_grid(
        X, y, fit_intercept=fit_intercept, sample_weight=sample_weight
    )[0]
    logger.debug(f"Computed alpha_max: {alpha_max}")
    alpha = alpha_max * (1.0 - step)
    while alpha > 0 and max_alpha_iter > 0:
        yield alpha
        alpha *= 1.0 - step
        max_alpha_iter -= 1


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
    step : float, default=1e-3
        The step size for the sequence of alphas to test. Will start at
        alpha_max * (1 - step) and decrease until alpha = 0 or max_alpha_iter
        is reached.
    n_nonzero_coefs : int, default=1
        The number of non-zero coefficients to aim for. If >1, eps is adjusted
        to achieve at least n_nonzero_coefs non-zero coefficients. If < 1,
        it is used as a proportion of the number of features.
    max_alpha_iter : int, default=100
        Maximum number of alphas to test. This is a safeguard to prevent
        infinite loops in case of numerical issues.
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
        step: float = 1e-3,
        n_nonzero_coefs: int = 1,
        max_alpha_iter: int = 100,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        if not (0 < step < 1):
            raise ValueError("step must be a positive float less than 1.")
        if n_nonzero_coefs <= 0:
            raise ValueError(
                "n_nonzero_coefs must be a positive integer or a "
                "positive float < 1."
            )
        self.step = step
        self.n_nonzero_coefs = n_nonzero_coefs
        self.max_alpha_iter = max_alpha_iter
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        sample_weight: Optional[DataLike] = None,
    ) -> "LassoLambdaMax":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.
        sample_weight : DataLike, optional
            Sample weights. If None, all samples are given equal weight.

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
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )

        target_nonzero = (
            self.n_nonzero_coefs
            if isinstance(self.n_nonzero_coefs, int)
            else int(self.n_nonzero_coefs * npX.shape[1])
        )
        n_iterations = 0
        for alpha in _generate_alpha_sequence(
            npX,
            npy,
            step=self.step,
            fit_intercept=self.fit_intercept,
            sample_weight=sample_weight,
            max_alpha_iter=self.max_alpha_iter,
        ):
            base_lasso.set_params(alpha=alpha)
            base_lasso.fit(npX, npy, sample_weight=sample_weight)
            n_nonzero = np.sum(base_lasso.coef_ != 0)
            logger.debug(
                f"Trying alpha={alpha:.5e}, non-zero coefs: {n_nonzero} "
                f"(target: {target_nonzero})"
            )
            n_iterations += 1
            if n_nonzero >= target_nonzero:
                self.alpha_fit_ = alpha
                logger.debug(
                    f"Selected alpha: {alpha:.5e} with {n_nonzero} "
                    f"non-zero coefficients (took {n_iterations} iterations)."
                )
                self.n_nonzero_coefs_ = n_nonzero
                break

        if not n_nonzero >= target_nonzero:
            self.alpha_fit_ = alpha
            logger.warning(
                "No alpha found with the desired number of non-zero "
                f"coefficients ({target_nonzero}). "
                f"Fitting with alpha={self.alpha_fit_} which yielded "
                f"{n_nonzero} non-zero coefficients. Try adjusting step or "
                f"max_alpha_iter."
            )

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
