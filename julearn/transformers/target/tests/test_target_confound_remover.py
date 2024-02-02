"""Provides tests for the target confound remover."""

import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.linear_model import LinearRegression

from julearn.transformers.target import TargetConfoundRemover


def test_TargetConfoundRemover(
    X_iris: pd.DataFrame, y_iris: pd.Series  # noqa: N803
) -> None:
    """Test target confound remover.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.Series
        The iris dataset target.

    """

    to_rename = {
        "sepal_length": "sepal_length__:type:__continuous",
        "sepal_width": "sepal_width__:type:__confound",
        "petal_length": "petal_length__:type:__continuous",
        "petal_width": "petal_width__:type:__continuous",
    }
    X_iris.rename(columns=to_rename, inplace=True)

    remover = TargetConfoundRemover()

    y_removed: pd.Series = remover.fit_transform(  # type: ignore
        X_iris, y_iris
    )

    assert y_removed.shape == y_iris.shape

    assert y_removed.name == y_iris.name

    X_confounds = X_iris.loc[:, ["sepal_width__:type:__confound"]].values

    c_model = LinearRegression()
    c_model.fit(X_confounds, y_iris)
    y_pred = c_model.predict(X_confounds)
    residuals = y_iris - y_pred
    assert_series_equal(y_removed, residuals)

    remover2 = TargetConfoundRemover(
        model_confound=LinearRegression(),  # type: ignore
        confounds=["duck"],
    )

    to_rename = {
        "sepal_width__:type:__confound": "sepal_width__:type:__duck",
    }
    X_iris.rename(columns=to_rename, inplace=True)

    y_removed2: pd.Series = remover2.fit_transform(  # type: ignore
        X_iris, y_iris
    )
    assert_series_equal(y_removed, y_removed2)
