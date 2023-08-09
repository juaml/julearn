"""Provide tests for all models."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, Type

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
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
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from julearn.models import get_model
from julearn.utils.testing import do_scoring_test
from julearn.utils.typing import ModelLike


@pytest.mark.parametrize(
    "model_name, model_class, model_params",
    [
        ("nb_bernoulli", BernoulliNB, {}),
        ("nb_categorical", CategoricalNB, {}),
        ("nb_complement", ComplementNB, {}),
        ("nb_gaussian", GaussianNB, {}),
        ("nb_multinomial", MultinomialNB, {}),
    ],
)
def test_naive_bayes_estimators(
    df_iris: pd.DataFrame,
    model_name: str,
    model_class: Type[ModelLike],
    model_params: Dict[str, Any],
) -> None:
    """Test all naive bayes estimators.

    Parameters
    ----------
    df_iris : pd.DataFrame
        Iris dataset.
    model_name : str
        Name of the model to test.
    model_class : ModelLike
        Class of the model to test.
    model_params : dict
        Parameters to pass to the model.
    """
    df_binary = df_iris[df_iris["species"].isin(["setosa", "virginica"])]

    # keep only two species
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = {"continuous": X}
    y = "species"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)  # type: ignore
    else:
        t_model = model_class()
    t_df_binary = df_binary.copy(deep=True)
    t_df = df_iris.copy(deep=True)
    if model_name in ["nb_categorical"]:
        t_df_binary[X] = t_df_binary[X] > t_df_binary[X].mean()
        t_df[X] = t_df[X] > t_df[X].mean()
    scorers = ["accuracy"]
    api_params = {
        "model": model_name,
        "model_params": ju_model_params,
        "preprocess": None,
        "problem_type": "classification",
    }
    clf = make_pipeline(clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=t_df_binary,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )
    api_params = {
        "model": model_name,
        "model_params": ju_model_params,
        "preprocess": None,
        "problem_type": "classification",
    }
    clf = make_pipeline(clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=t_df,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )
    if model_name not in ["nb_bernoulli"]:
        # now let's try target-dependent scores
        scorers = ["recall", "precision", "f1"]
        sk_y = (t_df_binary[y] == "setosa").values.astype(int)
        api_params = {
            "model": model_name,
            "pos_labels": "setosa",
            "model_params": ju_model_params,
            "preprocess": None,
            "problem_type": "classification",
        }
        clf = make_pipeline(clone(t_model))
        do_scoring_test(
            X,
            y,
            X_types=X_types,
            data=t_df_binary,
            api_params=api_params,
            sklearn_model=clf,
            scorers=scorers,
            sk_y=sk_y,
        )


@pytest.mark.parametrize(
    "model_name, model_class, model_params",
    [
        ("svm", SVC, {}),
        (
            "rf",
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
        ),
        ("et", ExtraTreesClassifier, {"n_estimators": 10, "random_state": 42}),
        ("dummy", DummyClassifier, {"strategy": "prior"}),
        ("gauss", GaussianProcessClassifier, {}),
        ("logit", LogisticRegression, {}),
        ("logitcv", LogisticRegressionCV, {}),
        ("ridge", RidgeClassifier, {}),
        ("ridgecv", RidgeClassifierCV, {}),
        ("sgd", SGDClassifier, {"random_state": 2}),
        ("adaboost", AdaBoostClassifier, {"random_state": 42}),
        (
            "bagging",
            BaggingClassifier,
            {
                "random_state": 42,
                "estimator": DecisionTreeClassifier(random_state=42),
            },
        ),
        ("gradientboost", GradientBoostingClassifier, {}),
    ],
)
def test_classificationestimators(
    df_binary: pd.DataFrame,
    model_name: str,
    model_class: Type[ModelLike],
    model_params: Dict,
) -> None:
    """Test all classification estimators.

    Parameters
    ----------
    df_binary : pd.DataFrame
        Binary classification dataset.
    model_name : str
        Name of the model to test.
    model_class : ModelLike
        Class of the model to test.
    model_params : dict
        Parameters to pass to the model.
    """

    decimal = 5 if model_name != "sgd" else -1

    # keep only two species
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = {"continuous": X}
    y = "species"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)  # type: ignore
    else:
        t_model = model_class()
    scorers = ["accuracy"]
    api_params = {
        "model": model_name,
        "model_params": ju_model_params,
        "problem_type": "classification",
        "preprocess": "zscore",
    }
    clf = make_pipeline(StandardScaler(), clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=df_binary,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
        decimal=decimal,
    )
    if model_name != "dummy":
        # now let's try target-dependent scores
        scorers = ["recall", "precision", "f1"]
        sk_y = (df_binary[y] == "setosa").values.astype(np.int64)
        api_params = {
            "model": model_name,
            "pos_labels": "setosa",
            "model_params": ju_model_params,
            "problem_type": "classification",
            "preprocess": "zscore",
        }
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(
            X,
            y,
            X_types=X_types,
            data=df_binary,
            api_params=api_params,
            sklearn_model=clf,
            scorers=scorers,
            sk_y=sk_y,
            decimal=decimal,
        )


@pytest.mark.parametrize(
    "model_name, model_class, model_params",
    [
        ("svm", SVR, {}),
        (
            "rf",
            RandomForestRegressor,
            {"n_estimators": 10, "random_state": 42},
        ),
        ("et", ExtraTreesRegressor, {"n_estimators": 10, "random_state": 42}),
        ("dummy", DummyRegressor, {"strategy": "mean"}),
        ("gauss", GaussianProcessRegressor, {"random_state": 42}),
        ("linreg", LinearRegression, {}),
        ("ridge", Ridge, {}),
        ("ridgecv", RidgeCV, {}),
        ("sgd", SGDRegressor, {"random_state": 2}),
        ("adaboost", AdaBoostRegressor, {"random_state": 2}),
        ("bagging", BaggingRegressor, {"random_state": 2}),
        ("gradientboost", GradientBoostingRegressor, {"random_state": 42}),
    ],
)
def test_regression_estimators(
    df_binary: pd.DataFrame,
    model_name: str,
    model_class: Type[ModelLike],
    model_params: Dict,
) -> None:
    """Test all regression estimators.

    Parameters
    ----------
    df_binary : pd.DataFrame
        Binary classification dataset.
    model_name : str
        Name of the model to test.
    model_class : ModelLike
        Class of the model to test.
    model_params : dict
        Parameters to pass to the model.
    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = {"continuous": X}
    y = "petal_width"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)  # type: ignore
    else:
        t_model = model_class()
    scorers = ["neg_root_mean_squared_error", "r2"]
    api_params = {
        "model": model_name,
        "model_params": ju_model_params,
        "preprocess": "zscore",
        "problem_type": "regression",
    }
    clf = make_pipeline(StandardScaler(), clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=df_binary,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
        decimal=2,
    )


def test_wrong_problem_types() -> None:
    """Test models with wrong problem types."""

    with pytest.raises(ValueError, match="is not suitable for"):
        get_model("linreg", "classification")

    with pytest.raises(ValueError, match="is not available"):
        get_model("wrong", "classification")
