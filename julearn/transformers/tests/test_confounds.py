"""Provide tests for the ConfoundRemover transformer."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, Type

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pytest import FixtureRequest, fixture
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from julearn.models import get_model
from julearn.transformers.confound_remover import ConfoundRemover
from julearn.utils.typing import ModelLike


@fixture(params=["rf", "linreg"], scope="module")
def models_confound_remover(request: FixtureRequest) -> str:
    """Return different models that work with classification and regression.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    str
        The name of the model.
    """
    return request.param


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
            "c__:type:__confound": np.arange(30, 40),
            "d__:type:__confound": np.arange(40, 50),
            "e__:type:__categorical": np.arange(50, 70, 2),
            "f__:type:__categorical": np.arange(70, 100, 3),
        }
    )
    return X


@fixture
def y_confounds() -> np.ndarray:
    """Create the y variable for the df_X_confounds fixture.

    Returns
    -------
    np.ndarray
        The y variable for the df_X_confounds fixture.
    """
    y = np.arange(10)
    return y


def test_ConfoundRemover__apply_threshold() -> None:
    """Test the _apply_threshold method."""
    vals = pd.DataFrame([1e-4, 1e-2, 1e-1, 0, 1])
    confound_remover = ConfoundRemover(threshold=1e-2)
    out_pos_vals = confound_remover._apply_threshold(vals)
    out_neg_vals = confound_remover._apply_threshold(-vals)

    assert_frame_equal(
        out_pos_vals[[True, True, False, False, False]],
        out_neg_vals[[True, True, False, False, False]],
    )

    assert (out_pos_vals.values == [0, 0, 1e-1, 0, 1]).all
    assert (out_neg_vals.values == [0, 0, -1e-1, 0, -1]).all


@pytest.mark.parametrize(
    "drop,confounds",
    [
        [None, ["c__:type:__confound", "d__:type:__confound"]],
        ["c__:type:__confound", ["d__:type:__confound"]],
    ],
)
def test_ConfoundRemover_confound_auto_find_conf(
    df_X_confounds: pd.DataFrame,  # noqa: N803
    drop: Optional[List[str]],
    confounds: Optional[List[str]],
    models_confound_remover: str,
) -> None:
    """Test finding confounds in the data types.

    Parameters
    ----------
    df_X_confounds : pd.DataFrame
        The dataframe with confounds.
    drop : list of str, optional
        The columns to drop from the dataframe.
    confounds : list of str, optional
        The confounds to use.
    models_confound_remover : str
        The model to use for the confound remover.
    """
    if drop is not None:
        df_X = df_X_confounds.drop(columns=drop)
    else:
        df_X = df_X_confounds

    features = df_X.drop(columns=confounds).columns

    model = get_model(models_confound_remover, "regression")
    confound_remover = ConfoundRemover(
        apply_to=["continuous", "categorical"],
        model_confound=model,
    )

    np.random.seed(42)
    df_cofound_removed = confound_remover.fit_transform(df_X)
    np.random.seed(42)
    df_confounds = df_X.loc[:, confounds]  # type: ignore
    confound_regressions = [
        clone(model).fit(df_confounds, df_X.loc[:, feature])  # type: ignore
        for feature in features
    ]

    df_confound_removed_manual = df_X.drop(columns=confounds).copy()
    # Test that each model inside of the confound removal
    # is the same as if we would have trained the same model
    # in sklearn
    for internal_model, confound_regression, feature in zip(
        confound_remover.models_confound_,
        confound_regressions,
        features,
    ):
        manual_pred = confound_regression.predict(  # type: ignore
            df_X.loc[:, confounds]  # type: ignore
        )
        df_confound_removed_manual[feature] = manual_pred

        assert_array_equal(
            internal_model.predict(  # type: ignore
                df_X.loc[:, confounds].values  # type: ignore
            ),
            manual_pred,
        )
    df_confound_removed_manual = (
        df_X.drop(columns=confounds) - df_confound_removed_manual
    )

    # After confound removal the confound should be removed
    assert (
        df_cofound_removed.columns == df_X.drop(columns=confounds).columns
    ).all()

    assert_frame_equal(df_cofound_removed, df_confound_removed_manual)


@pytest.mark.parametrize(
    "model_class", [LinearRegression, RandomForestRegressor]
)
@pytest.mark.parametrize(
    "confounds",
    [
        "a__:type:__continuous",
        ["a__:type:__continuous"],
        ["a__:type:__continuous", "b__:type:__continuous"],
    ],
)
def test_confound_set_confounds(
    model_class: Type[ModelLike],
    confounds: List[str],
    df_X_confounds: pd.DataFrame,  # noqa: N803
):
    """Test confound removal confounds parameter setting.

    Parameters
    ----------
    model_class : sklearn model
        The model to use for the confound remover.
    confounds : list of str
        The confounds to use.
    df_X_confounds : pd.DataFrame
        The dataframe with confounds.
    """
    features = df_X_confounds.drop(columns=confounds).columns
    confound_remover = ConfoundRemover(
        model_confound=model_class(),
        confounds=confounds,
        apply_to="*",
    )

    np.random.seed(42)
    df_cofound_removed = confound_remover.fit_transform(df_X_confounds)

    np.random.seed(42)
    conf_as_feat = confounds if isinstance(confounds, list) else [confounds]
    df_confounds = df_X_confounds.loc[:, conf_as_feat]  # type: ignore
    confound_regressions = [
        model_class().fit(
            df_confounds, df_X_confounds.loc[:, feature]  # type: ignore
        )
        for feature in features
    ]
    df_confound_removed_manual = df_X_confounds.drop(columns=confounds).copy()
    # Test that each model inside of the confound removal
    # is the same as if we would have trained the same model
    # in sklearn

    for internal_model, confound_regression, feature in zip(
        confound_remover.models_confound_,
        confound_regressions,
        features,
    ):
        t_confounds = df_X_confounds.loc[:, conf_as_feat]  # type: ignore
        manual_pred = confound_regression.predict(t_confounds)  # type: ignore
        df_confound_removed_manual[feature] = manual_pred

        assert_array_equal(
            internal_model.predict(t_confounds.values),
            manual_pred,
        )

    df_confound_removed_manual = (
        df_X_confounds.drop(columns=confounds) - df_confound_removed_manual
    )
    # After confound removal the confound should be removed
    assert (
        df_cofound_removed.columns
        == df_X_confounds.drop(columns=confounds).columns
    ).all()

    assert_frame_equal(df_cofound_removed, df_confound_removed_manual)


def test_return_confound(df_X_confounds: pd.DataFrame) -> None:  # noqa: N803
    """Test that the confound is returned if keep_confounds is True.

    Parameters
    ----------
    df_X_confounds : pd.DataFrame
        The dataframe with confounds.

    """
    remover = ConfoundRemover(
        apply_to=["categorical", "continuous"], keep_confounds=True
    )
    X_trans = remover.fit_transform(df_X_confounds)
    assert_array_equal(X_trans.columns, df_X_confounds.columns)


def test_no_confound_found() -> None:
    """Test that an error is raised if no confound is found."""
    _X = pd.DataFrame({"a": np.arange(10)})
    remover = ConfoundRemover()
    with pytest.raises(ValueError, match="No confound was found"):
        remover.fit_transform(_X)


def test_no_dataframe_provided() -> None:
    """Test that an error is raised if a dataframe is not provided."""
    X = np.zeros((10, 2), dtype=float)
    remover = ConfoundRemover()
    with pytest.raises(ValueError, match="ConfoundRemover only sup"):
        remover.fit(X)  # type: ignore
