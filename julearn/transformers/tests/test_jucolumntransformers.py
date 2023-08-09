"""Test the JuColumnTransformers class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Dict, Type

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal
from pytest import fixture
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from julearn.transformers import JuColumnTransformer
from julearn.utils.typing import EstimatorLike


@fixture
def df_X_confounds() -> pd.DataFrame:
    """Create a dataframe with confounds.

    Returns
    -------
    pd.DataFrame
        A dataframe with confounds.

    """
    X = pd.DataFrame(
        {
            "a__:type:__continuous": np.arange(10) + np.random.rand(10),
            "b__:type:__continuous": np.arange(10, 20) + np.random.rand(10),
            "c__:type:__confound": np.arange(30, 40).astype(float),
            "d__:type:__confound": np.arange(40, 50).astype(float),
            "e__:type:__categorical": np.arange(50, 70, 2).astype(float),
            "f__:type:__categorical": np.arange(70, 100, 3).astype(float),
        }
    )
    return X


@pytest.mark.parametrize(
    "name, klass, params",
    [
        ("zscore", StandardScaler, {}),
        ("scaler_robust", RobustScaler, {}),
        ("scaler_minmax", MinMaxScaler, {}),
        ("scaler_maxabs", MaxAbsScaler, {}),
        ("scaler_normalizer", Normalizer, {}),
        ("scaler_quantile", QuantileTransformer, {"n_quantiles": 10}),
        ("scaler_power", PowerTransformer, {}),
    ],
)
def test_JuColumnTransformer(
    name: str,
    klass: Type[EstimatorLike],
    params: Dict,
    df_X_confounds: pd.DataFrame,  # noqa: N803
):
    """Test JuColumnTransformer class."""

    # Create the transformer
    transformer = JuColumnTransformer(
        name=name,
        transformer=klass(),
        apply_to=["continuous"],
    )

    # Set the parameters
    transformer.set_params(**params)

    # Fit the transformer
    transformer.fit(df_X_confounds)

    # Transform the data
    X_transformed = transformer.transform(df_X_confounds.copy())

    df_X_transformed = pd.DataFrame(
        X_transformed, columns=transformer.get_feature_names_out()
    )
    # Check that the columns are as expected
    assert set(df_X_transformed.columns) == {
        "a__:type:__continuous",
        "b__:type:__continuous",
        "c__:type:__confound",
        "d__:type:__confound",
        "e__:type:__categorical",
        "f__:type:__categorical",
    }

    kept = [
        "c__:type:__confound",
        "d__:type:__confound",
        "e__:type:__categorical",
        "f__:type:__categorical",
    ]
    trans = ["a__:type:__continuous", "b__:type:__continuous"]

    sk_trans = klass(**params)
    manual = sk_trans.fit_transform(df_X_confounds[trans])  # type: ignore

    assert_frame_equal(df_X_transformed[kept], df_X_confounds[kept])
    assert_array_equal(df_X_transformed[trans].values, manual)


def test_JuColumnTransformer_row_select():
    """Test row selection for JuColumnTransformer."""
    X = pd.DataFrame(
        {
            "a__:type:__continuous": [0, 0, 1, 1],
            "b__:type:__healthy": [1, 1, 0, 0],
        }
    )

    transformer_healthy = JuColumnTransformer(
        name="zscore",
        transformer=StandardScaler(),
        apply_to="continuous",
        row_select_col_type=["healthy"],
        row_select_vals=1,
    )

    transformer_unhealthy = JuColumnTransformer(
        name="zscore",
        transformer=StandardScaler(),
        apply_to="continuous",
        row_select_col_type=["healthy"],
        row_select_vals=0,
    )

    transformer_both = JuColumnTransformer(
        name="zscore",
        transformer=StandardScaler(),
        apply_to="continuous",
        row_select_col_type=["healthy"],
        row_select_vals=[0, 1],
    )
    mean_healthy = (
        transformer_healthy.fit(X)
        .column_transformer_.transformers_[0][1]
        .mean_
    )
    mean_unhealthy = (
        transformer_unhealthy.fit(X)
        .column_transformer_.transformers_[0][1]
        .mean_
    )

    mean_both = (
        transformer_both.fit(X).column_transformer_.transformers_[0][1].mean_
    )

    assert_almost_equal(
        transformer_healthy._select_rows(X, y=None)["X"].index.values, [0, 1]
    )
    assert_almost_equal(
        transformer_unhealthy._select_rows(X, None)["X"].index.values, [2, 3]
    )
    assert_almost_equal(
        transformer_both._select_rows(X, None)["X"].index.values, [0, 1, 2, 3]
    )

    assert_almost_equal(mean_unhealthy, [1])
    assert_almost_equal(mean_healthy, [0])
    assert_almost_equal(mean_both, [0.5])
