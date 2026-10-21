"""Provides tests for the JuGeneratedTargetModel class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import SVR

from julearn.transformers.target.ju_generated_target_model import (
    GeneratedTargetWarning,
    JuGeneratedTargetModel,
)


def test_JuGeneratedTargetModel_regression(
    X_iris: pd.DataFrame,  # noqa: N803
) -> None:
    """Test JuGeneratedTargetModel."""
    model = SVR()
    transformer = PCA(n_components=1, random_state=42)
    transformer.set_output(transform="pandas")

    ju_generated_target_model = JuGeneratedTargetModel(
        model=model,  # type: ignore
        transformer=transformer,  # type: ignore
    )

    fake_y = pd.Series(np.zeros(X_iris.shape[0]))

    ju_generated_target_model.fit(X_iris, y=fake_y)
    with pytest.warns(
        GeneratedTargetWarning,
        match=r"has been generated from the features",
    ):
        y_pred = ju_generated_target_model.predict(X_iris)

    model_sk = SVR()
    y_iris = transformer.fit(X_iris).transform(X_iris)
    model_sk.fit(X_iris, y_iris)
    y_pred_sk = model_sk.predict(X_iris)
    assert_array_equal(y_pred, y_pred_sk)

    with pytest.warns(
        RuntimeWarning,
        match=r"non-zero target was provided",
    ):
        score_ju = ju_generated_target_model.score(X_iris, y_iris)
    score_sk = model_sk.score(X_iris, y_iris)

    assert score_ju == score_sk


def test_JuGeneratedTargetModel_classification(
    X_iris: pd.DataFrame,  # noqa: N803
) -> None:
    """Test JuGeneratedTargetModel."""
    model = LogisticRegression()
    transformer = Pipeline(
        steps=[
            ("pca", PCA(n_components=1, random_state=42)),
            ("binarizer", Binarizer()),
        ]
    )
    transformer.set_output(transform="pandas")

    ju_generated_target_model = JuGeneratedTargetModel(
        model=model,  # type: ignore
        transformer=transformer,  # type: ignore
    )

    fake_y = pd.Series(np.zeros(X_iris.shape[0]))

    ju_generated_target_model.fit(X_iris, y=fake_y)
    with pytest.warns(
        GeneratedTargetWarning,
        match=r"has been generated from the features",
    ):
        y_pred = ju_generated_target_model.predict(X_iris)

    model_sk = LogisticRegression()
    y_iris = transformer.fit(X_iris).transform(X_iris)
    model_sk.fit(X_iris, y_iris)
    y_pred_sk = model_sk.predict(X_iris)
    assert_array_equal(y_pred, y_pred_sk)

    y_proba_ju = ju_generated_target_model.predict_proba(X_iris)
    y_proba_sk = model_sk.predict_proba(X_iris)
    assert_array_equal(y_proba_ju, y_proba_sk)

    y_df_ju = ju_generated_target_model.decision_function(X_iris)
    y_df_sk = model_sk.decision_function(X_iris)
    assert_array_equal(y_df_ju, y_df_sk)

    with pytest.warns(
        RuntimeWarning,
        match=r"non-zero target was provided",
    ):
        score_ju = ju_generated_target_model.score(X_iris, y_iris)
    score_ju2 = ju_generated_target_model.score(X_iris, fake_y)
    score_sk = model_sk.score(X_iris, y_iris)

    assert score_ju == score_sk
    assert score_ju2 == score_sk
    ju_klasses = ju_generated_target_model.classes_
    sk_klasses = model_sk.classes_  # type: ignore
    assert_array_equal(ju_klasses, sk_klasses)


def test_JuGeneratedTargetModel_no_fit(
    X_iris: pd.DataFrame,  # noqa: N803
) -> None:
    """Test JuGeneratedTargetModel without fit."""
    model = LogisticRegression()
    transformer = PCA(n_components=1, random_state=42)
    transformer.set_output(transform="pandas")

    ju_generated_target_model = JuGeneratedTargetModel(
        model=model,  # type: ignore
        transformer=transformer,  # type: ignore
    )
    fake_y = pd.Series(np.zeros(X_iris.shape[0]))
    with pytest.raises(ValueError, match="Model not fitted yet"):
        ju_generated_target_model.predict(X_iris)

    with pytest.raises(ValueError, match="Model not fitted yet"):
        ju_generated_target_model.score(X_iris, fake_y)

    with pytest.raises(ValueError, match="Model not fitted yet"):
        ju_generated_target_model.predict_proba(X_iris)

    with pytest.raises(ValueError, match="Model not fitted yet"):
        ju_generated_target_model.decision_function(X_iris)


def test_JuGeneratedTargetModel_score_sample_weight(
    X_iris: pd.DataFrame,  # noqa: N803
) -> None:
    """Test that JuGeneratedTargetModel.score supports sample_weight.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.

    """
    weights = np.linspace(0.1, 1, len(X_iris))
    fake_y = pd.Series(np.zeros(X_iris.shape[0]))

    def _get_model() -> JuGeneratedTargetModel:
        transformer = PCA(n_components=1, random_state=42)
        transformer.set_output(transform="pandas")
        return JuGeneratedTargetModel(
            model=SVR(),  # type: ignore
            transformer=transformer,  # type: ignore
        )

    model_ju = _get_model().fit(X_iris, y=fake_y)

    # The model is scored on the generated target
    transformer = PCA(n_components=1, random_state=42)
    transformer.set_output(transform="pandas")
    y_gen = transformer.fit(X_iris).transform(X_iris)
    model_sk = SVR().fit(X_iris, y_gen)
    score_sk = model_sk.score(X_iris, y_gen)
    score_sk_weighted = model_sk.score(X_iris, y_gen, sample_weight=weights)
    assert score_sk != score_sk_weighted
    assert model_ju.score(X_iris, fake_y) == score_sk
    assert (
        model_ju.score(X_iris, fake_y, sample_weight=weights)
        == score_sk_weighted
    )

    # With metadata routing, Pipeline.score always passes sample_weight
    pipeline = Pipeline([("model", _get_model())]).fit(X_iris, fake_y)
    assert pipeline.score(X_iris, fake_y) == score_sk
