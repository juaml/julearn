"""Provides tests for the JuGeneratedTargetModel class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import SVC, SVR

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
    model = SVC(probability=True, random_state=42)
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

    model_sk = SVC(probability=True, random_state=42)
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
    model = SVC(probability=True)
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
