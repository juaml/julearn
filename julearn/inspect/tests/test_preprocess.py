"""Provide tests for inspecting preprocess module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, List

import pytest

from numpy.testing import assert_array_equal
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from julearn.inspect.preprocess import preprocess
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
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    pipeline.fit(X_iris, y=y_iris)

    with pytest.raises(ValueError, match="not a DataFrame"):
        preprocess(
            pipeline, X=X, data=X_iris, until=None, with_column_types=False
        )
