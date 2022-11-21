# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    LinearRegression,
    Ridge,
    RidgeClassifier,
    RidgeCV,
    RidgeClassifierCV,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from seaborn import load_dataset

import pytest

from julearn.utils.testing import do_scoring_test
from julearn.estimators import get_model


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
def test_naive_bayes_estimators(model_name, model_class, model_params):
    """Test all naive bayes estimators"""
    df_iris = load_dataset("iris")

    # keep only two species
    df_binary = df_iris[df_iris["species"].isin(["setosa", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = dict(continuous=X)
    y = "species"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)
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
        sk_y = (t_df_binary[y].values == "setosa").astype(np.int64)
        api_params = {
            "model": model_name,
            "pos_labels": "setosa",
            "model_params": ju_model_params,
            "preprocess": None,
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
        ("rf", RandomForestClassifier, {"n_estimators": 10}),
        ("et", ExtraTreesClassifier, {"n_estimators": 10, "random_state": 42}),
        ("dummy", DummyClassifier, {"strategy": "prior"}),
        ("gauss", GaussianProcessClassifier, {}),
        ("logit", LogisticRegression, {}),
        ("logitcv", LogisticRegressionCV, {}),
        ("ridge", RidgeClassifier, {}),
        ("ridgecv", RidgeClassifierCV, {}),
        ("sgd", SGDClassifier, {"random_state": 2}),
        ("adaboost", AdaBoostClassifier, {}),
        ("bagging", BaggingClassifier, {}),
        ("gradientboost", GradientBoostingClassifier, {}),
    ],
)
def test_classificationestimators(model_name, model_class, model_params):
    """Test all classification estimators"""
    df_iris = load_dataset("iris")

    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["setosa", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = dict(continuous=X)
    y = "species"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)
    else:
        t_model = model_class()
    scorers = ["accuracy"]
    api_params = {"model": model_name, "model_params": ju_model_params}
    clf = make_pipeline(StandardScaler(), clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=df_iris,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )
    if model_name != "dummy":
        # now let's try target-dependent scores
        scorers = ["recall", "precision", "f1"]
        sk_y = (df_iris[y].values == "setosa").astype(np.int64)
        api_params = {
            "model": model_name,
            "pos_labels": "setosa",
            "model_params": model_params,
        }
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(
            X,
            y,
            X_types=X_types,
            data=df_iris,
            api_params=api_params,
            sklearn_model=clf,
            scorers=scorers,
            sk_y=sk_y,
        )


@pytest.mark.parametrize(
    "model_name, model_class, model_params",
    [
        ("svm", SVR, {}),
        ("rf", RandomForestRegressor, {"n_estimators": 10}),
        ("et", ExtraTreesRegressor, {"n_estimators": 10}),
        ("dummy", DummyRegressor, {"strategy": "mean"}),
        ("gauss", GaussianProcessRegressor, {}),
        ("linreg", LinearRegression, {}),
        ("ridge", Ridge, {}),
        ("ridgecv", RidgeCV, {}),
        ("sgd", SGDRegressor, {"random_state": 2}),
        ("adaboost", AdaBoostRegressor, {"random_state": 2}),
        ("bagging", BaggingRegressor, {"random_state": 2}),
        ("gradientboost", GradientBoostingRegressor, {}),
    ],
)
def test_regression_estimators(model_name, model_class, model_params):
    """Test all regression estimators"""
    df_iris = load_dataset("iris")

    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = dict(continuous=X)
    y = "petal_width"

    ju_model_params = None
    if len(model_params) > 0:
        ju_model_params = {
            f"{model_name}__{t_param}": t_value
            for t_param, t_value in model_params.items()
        }
        t_model = model_class(**model_params)
    else:
        t_model = model_class()
    scorers = ["neg_root_mean_squared_error", "r2"]
    api_params = {
        "model": model_name,
        "model_params": ju_model_params,
        "problem_type": "regression",
    }
    clf = make_pipeline(StandardScaler(), clone(t_model))
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=df_iris,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )


def test_wrong_problem_types():
    """Test models with wrong problem types"""

    with pytest.raises(ValueError, match="is not suitable for"):
        get_model("linreg", "classification")

    with pytest.raises(ValueError, match="is not available"):
        get_model("wrong", "classification")
