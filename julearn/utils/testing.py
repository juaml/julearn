"""Testing utilities for julearn."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from julearn import run_cross_validation
from julearn.base import WrapModel
from julearn.utils.typing import DataLike, EstimatorLike


def compare_models(  # noqa: C901, pragma: no cover
    clf1: EstimatorLike,
    clf2: EstimatorLike,
) -> None:
    """Compare two models.

    Parameters
    ----------
    clf1 : EstimatorLike
        The first model.
    clf2 : EstimatorLike
        The second model.

    Raises
    ------
    AssertionError
        If the models are not equal.
    """
    if isinstance(clf1, WrapModel):
        clf1 = clf1.model
    if isinstance(clf2, WrapModel):
        clf2 = clf2.model
    if clf1.__class__ != clf2.__class__:
        raise AssertionError("Different classes")
    if isinstance(clf1, (SVC, SVR)):
        idx1 = np.argsort(clf1.support_)
        v1 = clf1.support_vectors_[idx1]
        idx2 = np.argsort(clf2.support_)  # type: ignore
        v2 = clf2.support_vectors_[idx2]  # type: ignore
        if hasattr(clf1, "probability"):
            assert clf1.probability == clf2.probability  # type: ignore
    elif isinstance(
        clf1,
        (
            RandomForestClassifier,
            RandomForestRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        ),
    ):
        v1 = clf1.feature_importances_
        v2 = clf2.feature_importances_  # type: ignore
    elif isinstance(clf1, (DummyClassifier, DummyRegressor)):
        v1 = None
        v2 = None
        if hasattr(clf1, "_strategy"):
            assert clf1._strategy == clf2._strategy  # type: ignore
        if hasattr(clf1, "strategy"):
            assert clf1.strategy == clf2.strategy  # type: ignore
        if hasattr(clf1, "class_prior_"):
            assert_array_equal(
                clf1.class_prior_, clf2.class_prior_  # type: ignore
            )
        if hasattr(clf1, "constant_"):
            assert clf1.constant_ == clf2.constant_  # type: ignore
        if hasattr(clf1, "classes_"):
            assert_array_equal(clf1.classes_, clf2.classes_)  # type: ignore
    elif isinstance(clf1, GaussianProcessClassifier):
        if hasattr(clf1.base_estimator_, "estimators_"):
            # Multiclass
            est1 = clf1.base_estimator_.estimators_  # type: ignore
            v1 = np.array([x.pi_ for x in est1])  # type: ignore
            est2 = clf2.base_estimator_.estimators_  # type: ignore
            v2 = np.array([x.pi_ for x in est2])
        else:
            v1 = clf1.base_estimator_.pi_  # type: ignore
            v2 = clf2.base_estimator_.pi_  # type: ignore
    elif isinstance(clf1, GaussianProcessRegressor):
        v1 = np.c_[clf1.L_, clf1.alpha_]
        v2 = np.c_[clf2.L_, clf2.alpha_]  # type: ignore
    elif isinstance(
        clf1,
        (
            LogisticRegression,
            RidgeClassifier,
            RidgeClassifierCV,
            SGDClassifier,
            SGDRegressor,
            LinearRegression,
            Ridge,
            RidgeCV,
            BernoulliNB,
            ComplementNB,
            MultinomialNB,
        ),
    ):
        v1 = _get_coef_over_versions(clf1)
        v2 = _get_coef_over_versions(clf1)
    elif isinstance(clf1, CategoricalNB):
        v1 = None
        v2 = None
        for c1, c2 in zip(
            _get_coef_over_versions(clf1), _get_coef_over_versions(clf2)
        ):
            assert_array_equal(c1, c2)
    elif isinstance(clf1, GaussianNB):
        v1 = clf1.var_
        v2 = clf2.var_  # type: ignore
    elif isinstance(
        clf1,
        (
            AdaBoostClassifier,
            AdaBoostRegressor,
            BaggingClassifier,
            BaggingRegressor,
        ),
    ):
        est1 = clf1.estimators_
        v1 = np.array([x.feature_importances_ for x in est1])  # type: ignore
        est2 = clf2.estimators_  # type: ignore
        v2 = np.array([x.feature_importances_ for x in est2])  # type: ignore
    else:
        raise NotImplementedError(
            f"Model comparison for {clf1} not yet implemented."
        )
    assert_array_equal(v1, v2)  # type: ignore


def do_scoring_test(
    X: List[str],  # noqa: N803
    y: str,
    data: pd.DataFrame,
    api_params: Dict[str, Any],
    sklearn_model: EstimatorLike,
    scorers: List[str],
    groups: Optional[str] = None,
    X_types: Optional[Dict[str, List[str]]] = None,  # noqa: N803
    cv: int = 5,
    sk_y: Optional[np.ndarray] = None,
    decimal: int = 5,
):
    """Test scoring for a model, using the julearn and sklearn API.

    Parameters
    ----------
    X : List[str]
        The feature names.
    y : str
        The target name.
    data : pd.DataFrame
        The data.
    groups : str, optional
        The group name, by default None.
    X_types : Dict[str, List[str]]
        The feature types.
    api_params : Dict[str, Any]
        The parameters for the julearn API.
    sklearn_model : EstimatorLike
        The sklearn model.
    scorers : list of str
        The scorers to use.
    cv : int, optional
        The number of folds to use, by default 5.
    sk_y : np.ndarray, optional
        The target values, by default None.
    decimal : int, optional
        The number of decimals to use for the comparison, by default 5.
    """
    sk_X = data[X].values
    if sk_y is None:
        sk_y = data[y].values  # type: ignore

    params_dict = dict(api_params.items())
    if isinstance(cv, int):
        jucv = KFold(n_splits=cv, random_state=42, shuffle=True)
        sk_cv = KFold(n_splits=cv, random_state=42, shuffle=True)
    else:
        jucv = cv
        sk_cv = cv
    sk_groups = None
    if groups is not None:
        sk_groups = data[groups].values
    np.random.seed(42)
    actual, actual_estimator = run_cross_validation(
        X=X,
        y=y,
        X_types=X_types,
        data=data,
        groups=groups,
        scoring=scorers,
        cv=jucv,
        return_estimator="final",
        **params_dict,
    )

    np.random.seed(42)
    expected = cross_validate(
        sklearn_model, sk_X, sk_y, cv=sk_cv, scoring=scorers, groups=sk_groups
    )

    # Compare the models
    if isinstance(actual_estimator, Pipeline):
        clf1 = actual_estimator.steps[-1][1]  # type: ignore
    else:
        clf1 = actual_estimator

    if isinstance(sklearn_model, Pipeline):
        clf2 = clone(sklearn_model).fit(sk_X, sk_y).steps[-1][1]
    else:
        clf2 = clone(sklearn_model).fit(sk_X, sk_y)
    compare_models(clf1, clf2)

    if decimal > 0:
        for scoring in scorers:
            s_key = f"test_{scoring}"
            assert len(actual.columns) == len(expected) + 5  # type: ignore
            assert len(actual[s_key]) == len(expected[s_key])  # type: ignore
            assert_array_almost_equal(
                actual[s_key], expected[s_key], decimal=decimal  # type: ignore
            )


class PassThroughTransformer(TransformerMixin, BaseEstimator):
    """A transformer doing nothing."""

    def __init__(self):
        pass

    def fit(
        self, X: DataLike, y: Optional[DataLike] = None  # noqa: N803
    ) -> "PassThroughTransformer":
        """Fit the transformer.

        Parameters
        ----------
        X : DataLike
            The data.
        y : Optional[DataLike], optional
            The target, by default None.

        Returns
        -------
        PassThroughTransformer
            The fitted transformer.

        """
        return self

    def transform(self, X: DataLike) -> DataLike:  # noqa: N803
        """Transform the data.

        Parameters
        ----------
        X : DataLike
            The data.

        Returns
        -------
        DataLike
            The transformed data.
        """
        return X


class TargetPassThroughTransformer(PassThroughTransformer):
    """A target transformer doing nothing."""

    def __init__(self):
        super().__init__()

    def transform(
        self,
        X: Optional[DataLike] = None,  # noqa: N803
        y: Optional[DataLike] = None,
    ) -> Optional[DataLike]:
        """Transform the data.

        Parameters
        ----------
        X : DataLike, optional
            The data, by default None.
        y : DataLike, optional
            The target, by default None.

        Returns
        -------
        DataLike or None
            The target.
        """
        return y

    def fit_transform(
        self,
        X: Optional[DataLike] = None,  # noqa: N803
        y: Optional[DataLike] = None,
    ) -> Optional[DataLike]:
        """Fit the model and transform the data.

        Parameters
        ----------
        X : DataLike, optional
            The data, by default None.
        y : DataLike, optional
            The target, by default None.

        Returns
        -------
        DataLike or None
            The target.
        """
        self.fit(X, y)  # type: ignore
        return self.transform(X, y)


def _get_coef_over_versions(clf: EstimatorLike) -> np.ndarray:
    """Get the coefficients of a model, skipping warnings.

    Parameters
    ----------
    clf : EstimatorLike
        The model.

    Returns
    -------
    np.ndarray
        The coefficients.
    """
    if isinstance(
        clf, (BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB)
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            warnings.filterwarnings("error", category=DeprecationWarning)
            return clf.feature_log_prob_  # type: ignore
    else:
        return clf.coef_  # type: ignore
