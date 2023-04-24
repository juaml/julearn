"""Provide tests for inspecting preprocess module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, cast

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from julearn import run_cross_validation
from julearn.inspect import preprocess
from julearn.utils.typing import TransformerLike


@pytest.mark.parametrize(
    "pipeline,transformers,until",
    [
        (
            Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
            [StandardScaler()],
            None,
        ),
        (
            Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
            [StandardScaler()],
            "scaler",
        ),
        (
            Pipeline(
                [("scaler", StandardScaler()), ("pca", PCA()), ("svm", SVC())]
            ),
            [StandardScaler()],
            "scaler",
        ),
        (
            Pipeline(
                [("scaler", StandardScaler()), ("pca", PCA()), ("svm", SVC())]
            ),
            [StandardScaler(), PCA()],
            "pca",
        ),
        (
            Pipeline(
                [("scaler", StandardScaler()), ("pca", PCA()), ("svm", SVC())]
            ),
            [StandardScaler(), PCA()],
            None,
        ),
    ],
)
def test_preprocess_sklearn(
    X_iris: pd.DataFrame,
    y_iris: pd.Series,
    pipeline: Pipeline,
    transformers: List[TransformerLike],
    until: Optional[str],
) -> None:
    """Test the preprocess_sklearn function.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.
    pipeline : Pipeline
        The pipeline to test.
    transformers : list of TransformerLike
        The transformers to test.
    until : str, optional
        The transformer to stop at.
    """
    X = list(X_iris.columns)
    X = cast(List[str], X)
    pipeline.fit(X_iris, y=y_iris)

    X_train = X_iris.copy()
    for transformer in transformers:
        X_train = transformer.fit_transform(X_train, y=y_iris)

    X_preprocessed = preprocess(
        pipeline, X=X, data=X_iris, until=until, with_column_types=True
    )

    X_expected = X_iris.copy()
    for transformer in transformers:
        X_expected = transformer.transform(X_expected)
    assert_array_equal(X_preprocessed, X_expected)


def test_preprocess_sklearn_nodataframe(
    X_iris: pd.DataFrame,
    y_iris: pd.Series,
) -> None:
    """Test preprocess with non-dataframe output and column types removal.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.

    """
    X = list(X_iris.columns)
    X = cast(List[str], X)
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    pipeline.fit(X_iris, y=y_iris)

    with pytest.raises(ValueError, match="not a DataFrame"):
        preprocess(
            pipeline, X=X, data=X_iris, until=None, with_column_types=False
        )


def test_preprocess_no_step(X_iris, y_iris, df_iris):

    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    pipeline.fit(X_iris, y=y_iris)
    with pytest.raises(ValueError, match="No step named"):
        preprocess(pipeline, X=list(X_iris.columns),
                   data=df_iris,
                   until="non_existent")


def test_preprocess_with_column_types(df_iris):
    X = list(df_iris.iloc[:, :-1].columns)
    y = "species"
    _, model = run_cross_validation(
        X=X, y=y, data=df_iris, problem_type="classification",
        model="rf", return_estimator="final")
    X_t = preprocess(model, X=X, data=df_iris, with_column_types=False)
    assert (list(X_t.columns) == X)
