"""Provide tests for the JuTargetPieline class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler

from julearn.pipeline.target_pipeline import JuTargetPipeline
from julearn.transformers.target import JuTargetTransformer


def test_target_pipeline_sklearn(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test the target pipeline using a scikit-learn transformer.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.
    """

    steps = [("scaler", StandardScaler())]
    # TODO: @samihamdan: fix the protocol error
    pipeline = JuTargetPipeline(steps)  # type: ignore
    y_transformed = pipeline.fit_transform(X_iris, y_iris)

    assert isinstance(y_transformed, np.ndarray)
    assert y_transformed.shape == y_iris.shape

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_iris.values[:, None])[:, 0]

    assert_array_equal(y_transformed, y_scaled)


def test_target_pipeline_jutargettransformer(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test the target pipeline using a JuTargetTransformer.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.
    """

    class MedianSplitter(JuTargetTransformer):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> "MedianSplitter":
            self.median = np.median(y)
            return self

        def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
            return (y > self.median).astype(int)

    steps = [("splitter", MedianSplitter())]
    # TODO: @samihamdan: fix the protocol error
    pipeline = JuTargetPipeline(steps)  # type: ignore
    y_transformed = pipeline.fit_transform(X_iris, y_iris)

    assert y_transformed.shape == y_iris.shape

    median = np.median(y_iris)
    y_split = (y_iris > median).astype(int)

    assert_array_equal(y_transformed, y_split)


def test_target_pipeline_multiple_ju_sk(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test the target pipeline using JuTargetTransformer and sklearn.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.
    """

    class DeMeaner(JuTargetTransformer):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> "DeMeaner":
            self.mean = np.mean(X.iloc[:, 0].values)  # type: ignore
            return self

        def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
            return y - self.mean  # type: ignore

    steps = [("demeaner", DeMeaner()), ("scaler", StandardScaler())]
    pipeline = JuTargetPipeline(steps)
    y_transformed = pipeline.fit_transform(X_iris, y_iris)

    assert y_transformed.shape == y_iris.shape

    first_col = X_iris.values[:, 0]
    mean = np.mean(first_col)
    y_demeaned = y_iris.values - mean  # type: ignore

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_demeaned[:, None])[:, 0]

    assert_array_equal(y_transformed, y_scaled)


def test_target_pipeline_multiple_sk_ju(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test the target pipeline using sklearn and JuTargetTransformer.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.
    """

    class DeMeaner(JuTargetTransformer):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> "DeMeaner":
            self.mean = np.mean(X.iloc[:, 0].values)  # type: ignore
            return self

        def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
            return y - self.mean  # type: ignore

    steps = [("scaler", StandardScaler()), ("demeaner", DeMeaner())]
    pipeline = JuTargetPipeline(steps)
    y_transformed = pipeline.fit_transform(X_iris, y_iris)

    assert y_transformed.shape == y_iris.shape

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_iris.values[:, None])[:, 0]

    first_col = X_iris.values[:, 0]
    mean = np.mean(first_col)
    y_demeaned = y_scaled - mean  # type: ignore

    assert_array_equal(y_transformed, y_demeaned)


def test_target_pipeline_errors() -> None:
    """Test the target pipeline errors."""

    steps = ("scaler", StandardScaler())

    with pytest.raises(TypeError, match="steps must be a list"):
        JuTargetPipeline(steps)  # type: ignore