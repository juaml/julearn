"""Provides tests for the JuTransformedTargetModel class."""

# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR

from julearn.pipeline import JuTargetPipeline
from julearn.transformers.target import (
    JuTransformedTargetModel,
    TransformedTargetWarning,
)


def test_JuTransformedTargetModel(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test JuTransformedTargetModel."""

    steps = [("scaler", StandardScaler())]
    transformer = JuTargetPipeline(steps)  # type: ignore
    model = SVR()

    # TODO: @samihamdan: fix the protocol error
    ju_transformed_target_model = JuTransformedTargetModel(
        transformer=transformer, model=model  # type: ignore
    )

    ju_transformed_target_model.fit(X_iris, y_iris)
    y_pred = ju_transformed_target_model.predict(X_iris)

    model_sk = SVR()
    scaler_sk = StandardScaler()
    y_scaled = scaler_sk.fit_transform(y_iris.values[:, None])[:, 0]
    model_sk.fit(X_iris, y_scaled)
    y_pred_sk = model_sk.predict(X_iris)
    y_inverse_sk = scaler_sk.inverse_transform(y_pred_sk[:, None])[:, 0]
    assert_array_equal(y_pred, y_inverse_sk)


def test_JuTransformedTargetModel_noinverse(
    X_iris: pd.DataFrame, y_iris: pd.Series
) -> None:
    """Test JuTransformedTargetModel."""
    steps = [("quantile", Normalizer())]
    transformer = JuTargetPipeline(steps)  # type: ignore
    model = SVR()

    # TODO: @samihamdan: fix the protocol error
    ju_transformed_target_model = JuTransformedTargetModel(
        transformer=transformer, model=model  # type: ignore
    )

    ju_transformed_target_model.fit(X_iris, y_iris)
    with pytest.warns(
        TransformedTargetWarning,
        match=r"has been transformed to fit the model",
    ):
        y_pred = ju_transformed_target_model.predict(X_iris)

    model_sk = SVR()
    scaler_sk = Normalizer()
    y_scaled = scaler_sk.fit_transform(y_iris.values[:, None])[:, 0]
    model_sk.fit(X_iris, y_scaled)
    y_pred_sk = model_sk.predict(X_iris)

    assert_array_equal(y_pred, y_pred_sk)
