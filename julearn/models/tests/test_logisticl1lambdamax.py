"""Provide tests for LogisticL1LambdaMax model."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
from sklearn.preprocessing import StandardScaler

from julearn.models.logisticl1lambdamax import LogisticL1LambdaMax


def test_logistic_l1_lambda_max(
    df_binary: pd.DataFrame,
):
    """Test LogisticL1LambdaMax classification.

    Parameters
    ----------
    df_binary : pd.DataFrame
        Binary classification dataset.

    """
    X = df_binary[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y = (df_binary["species"] == "virginica").astype(int)

    model = LogisticL1LambdaMax()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "classes_")
    assert model.coef_.shape[1] == X.shape[1]
