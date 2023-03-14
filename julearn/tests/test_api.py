"""Provide tests for the API."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pandas as pd

import pytest
import warnings

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from julearn import run_cross_validation
from julearn.utils.testing import do_scoring_test
from julearn.pipeline import PipelineCreator


def test_run_cv_simple_binary(
    df_binary: pd.DataFrame, df_iris: pd.DataFrame
) -> None:
    """Test a simple binary classification problem.

    Parameters
    ----------
    df_binary : pd.DataFrame
        The iris dataset as a binary classification problem.
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    scorers = ["accuracy", "balanced_accuracy"]
    api_params = {"model": "svm", "problem_type": "classification"}
    sklearn_model = SVC()

    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        do_scoring_test(
            X=X,
            y=y,
            data=df_binary,
            scorers=scorers,
            api_params=api_params,
            sklearn_model=sklearn_model,
        )

    # now let's try target-dependent scores
    scorers = ["recall", "precision", "f1"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
        }
        sklearn_model = SVC()
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )

    # now let's try proba-dependent scores
    X = ["sepal_length", "petal_length"]
    scorers = ["accuracy", "roc_auc"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
            "model_params": {"svm__probability": True},
        }
        sklearn_model = SVC(probability=True)
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )

    # now let's try for decision_function based scores
    # e.g. svm with probability=False
    X = ["sepal_length", "petal_length"]
    scorers = ["accuracy", "roc_auc"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
        }
        sklearn_model = SVC(probability=False)
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )


def test_run_cv_simple_binary_errors(
    df_binary: pd.DataFrame, df_iris: pd.DataFrame
) -> None:
    """Test a simple classification problem errors

    Parameters
    ----------
    df_binary : pd.DataFrame
        The iris dataset as a binary classification problem.
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """

    # Test error when pos_labels are not provide (target-dependent scores)
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    scorers = ["recall", "precision", "f1"]
    api_params = {"model": "svm", "problem_type": "classification"}
    sklearn_model = SVC()

    with pytest.warns(UserWarning, match="Target is multiclass but average"):
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
        )


def test_run_cv_errors(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation errors and warnings.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    # test error when model is a pipeline
    model = make_pipeline(SVC())
    with pytest.raises(ValueError, match="a scikit-learn pipeline"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
            problem_type="classification",
        )

    # test error when model is a pipeline creator and preprocess is set
    model = PipelineCreator(problem_type="classification")
    with pytest.raises(ValueError, match="preprocess should be None"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
            preprocess="zscore",
        )

    # test error when model is a pipeline creator and problem_type is set
    model = PipelineCreator(problem_type="classification")
    with pytest.raises(ValueError, match="Problem type should be set"):
        run_cross_validation(
            X=X, y=y, data=df_iris, model=model, problem_type="classification"
        )

    model = 2
    with pytest.raises(ValueError, match="has to be a PipelineCreator"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
        )

    model = "svm"
    with pytest.raises(ValueError, match="`problem_type` must be specified"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
        )

    model = "svm"
    with pytest.raises(ValueError, match="preprocess has to be a string"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
            preprocess=2,
            problem_type="classification",
        )

    model = SVC()
    model_params = {"svc__C": 1}
    with pytest.raises(
        ValueError, match="Cannot use model_params with a model object"
    ):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model=model,
            model_params=model_params,
            problem_type="classification",
        )

    model = PipelineCreator(problem_type="classification")
    model_params = {"svc__C": 1}
    with pytest.raises(ValueError, match="must be None"):
        run_cross_validation(
            X=X, y=y, data=df_iris, model=model,
            model_params=model_params,
        )

    model = "svm"
    model_params = {"svc__C": 1}
    with pytest.raises(ValueError, match="model_params are incorrect"):
        run_cross_validation(
            X=X, y=y, data=df_iris, model=model,
            model_params=model_params, problem_type="classification",
        )

    model = "svm"
    model_params = {"probability": True, "svm__C": 1}
    with pytest.raises(ValueError, match="model_params are incorrect"):
        run_cross_validation(
            X=X, y=y, data=df_iris, model=model,
            model_params=model_params, problem_type="classification",
        )


