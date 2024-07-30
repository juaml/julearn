"""Provides tests for the JuGeneratedTargetModel class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from julearn.transformers.target.ju_generated_target_model import (
    JuGeneratedTargetModel,
)


def test_JuGeneratedTargetModel(
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
    y_pred = ju_generated_target_model.predict(X_iris)

    model_sk = SVR()
    y_iris = transformer.fit(X_iris).transform(X_iris)
    model_sk.fit(X_iris, y_iris)
    y_pred_sk = model_sk.predict(X_iris)
    assert_array_equal(y_pred, y_pred_sk)
