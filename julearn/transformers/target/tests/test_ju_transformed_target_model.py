"""Provides tests for the JuTransformedTargetModel class."""

# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR, SVC

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


def test_not_fitted(X_iris, y_iris):

    steps = [("scaler", StandardScaler())]
    transformer = JuTargetPipeline(steps)  # type: ignore
    model = SVC(probability=True)

    target_model = JuTransformedTargetModel(
        transformer=transformer, model=model  # type: ignore
    )
    with pytest.raises(ValueError, match='Model not fitted '):
        target_model.score(X_iris, y_iris)
    with pytest.raises(ValueError, match='Model not fitted '):
        target_model.predict(X_iris)
    with pytest.raises(ValueError, match='Model not fitted '):
        target_model.predict_proba(X_iris)
    with pytest.raises(ValueError, match='Model not fitted '):
        target_model.decision_function(X_iris)
    with pytest.raises(ValueError, match='Model not fitted '):
        target_model.classes_
