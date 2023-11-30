"""Provide tests for DynamicSelection."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest import FixtureRequest, fixture
from pytest_lazyfixture import lazy_fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, train_test_split

from julearn.models.dynamic import DynamicSelection


# Check if the test is running on juseless
try:
    import deslib  # type: ignore # noqa: F401
except ImportError:
    pytest.skip("Need deslib to test", allow_module_level=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from deslib.dcs import MCB, OLA
    from deslib.des import DESP, KNOP, KNORAE, KNORAU, METADES
    from deslib.static import SingleBest, StackedClassifier, StaticSelection


_algorithm_objects = {
    "METADES": METADES,
    "SingleBest": SingleBest,
    "StaticSelection": StaticSelection,
    "StackedClassifier": StackedClassifier,
    "KNORAU": KNORAU,
    "KNORAE": KNORAE,
    "DESP": DESP,
    "OLA": OLA,
    "MCB": MCB,
    "KNOP": KNOP,
}

# Ignore deprecation warnings from deslib
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

@fixture(
    params=[
        "METADES",
        "SingleBest",
        "StaticSelection",
        "StackedClassifier",
        "KNORAU",
        "KNORAE",
        "DESP",
        "OLA",
        "MCB",
        "KNOP",
    ],
    scope="module",
)
def all_deslib_algorithms(request: FixtureRequest) -> str:
    """Return different algorithms for the iris dataset features.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    dict or None
        A dictionary with the types for the features.
    """
    return request.param


@pytest.mark.parametrize(
    "algo_name",
    [lazy_fixture("all_deslib_algorithms")],
)
@pytest.mark.skip("Deslib is not compatible with new python. Waiting for PR.")
def test_algorithms(
    df_iris: pd.DataFrame,
    algo_name: str,
) -> None:
    """Test all the algorithms from deslib.

    Parameters
    ----------
    df_iris : pd.DataFrame
        Iris dataset.
    algo_name : str
        Name of the algorithm.
    """

    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    seed = 42
    ds_split = 0.2

    # julearn
    np.random.seed(seed)
    ensemble_model = RandomForestClassifier(random_state=6)

    dynamic_model = DynamicSelection(
        ensemble=ensemble_model,  # type: ignore
        algorithm=algo_name,
        random_state=seed,
        random_state_algorithm=seed,
        ds_split=ds_split,
    )
    dynamic_model.fit(df_iris[X], df_iris[y])
    score_julearn = dynamic_model.score(df_iris[X], df_iris[y])
    pred_julearn = dynamic_model.predict(df_iris[X])

    # deslib
    np.random.seed(seed)
    X_train, X_dsel, y_train, y_dsel = train_test_split(
        df_iris[X], df_iris[y], test_size=ds_split, random_state=seed
    )

    pool_classifiers = RandomForestClassifier(random_state=6)
    pool_classifiers.fit(X_train, y_train)

    cls = _algorithm_objects[algo_name]
    model_deslib = cls(pool_classifiers, random_state=seed)  # type: ignore
    model_deslib.fit(X_dsel, y_dsel)
    score_deslib = model_deslib.score(df_iris[X], df_iris[y])
    pred_deslib = model_deslib.predict(df_iris[X])

    assert score_deslib == score_julearn
    assert (pred_deslib == pred_julearn).all()

    if hasattr(model_deslib, "predict_proba"):
        pred_proba_julearn = dynamic_model.predict_proba(df_iris[X])
        pred_proba_deslib = model_deslib.predict_proba(  # type: ignore
            df_iris[X].values  # Deslib works with numpy arrays only
        )
        assert (pred_proba_deslib == pred_proba_julearn).all()


def test_wrong_algo(df_iris: pd.DataFrame) -> None:
    """Test wrong algorithm.

    Parameters
    ----------
    df_iris : pd.DataFrame
        Iris dataset.
    """
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    ensemble_model = RandomForestClassifier()

    with pytest.raises(ValueError, match="wrong is not a valid or supported"):
        dynamic_model = DynamicSelection(
            ensemble=ensemble_model, algorithm="wrong"  # type: ignore
        )
        dynamic_model.fit(df_iris[X], df_iris[y])


@pytest.mark.parametrize(
    "ds_split",
    [
        0.2,
        0.3,
        [train_test_split(np.arange(20), test_size=0.4, shuffle=True)],
        ShuffleSplit(n_splits=1),
    ],
)
@pytest.mark.skip("Deslib is not compatible with new python. Waiting for PR.")
def test_ds_split_parameter(ds_split: Any, df_iris: pd.DataFrame) -> None:
    """Test ds_split parameter.

    Parameters
    ----------
    ds_split : float or tuple or sklearn.model_selection._split.ShuffleSplit
        ds_split parameter.
    df_iris : pd.DataFrame
        Iris dataset.
    """
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    df_iris = df_iris.sample(n=len(df_iris))
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    ensemble_model = RandomForestClassifier()

    dynamic_model = DynamicSelection(
        ensemble=ensemble_model,  # type: ignore
        algorithm="METADES",
        ds_split=ds_split,
    )
    dynamic_model.fit(df_iris[X], df_iris[y])


@pytest.mark.parametrize("ds_split", [4, ShuffleSplit(n_splits=2)])
@pytest.mark.skip("Deslib is not compatible with new python. Waiting for PR.")
def test_ds_split_error(ds_split: Any, df_iris: pd.DataFrame) -> None:
    """Test ds_split errors.

    Parameters
    ----------
    ds_split : float or tuple or sklearn.model_selection._split.ShuffleSplit
        ds_split parameter.
    df_iris : pd.DataFrame
        Iris dataset.
    """
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    df_iris = df_iris.sample(n=len(df_iris))
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    ensemble_model = RandomForestClassifier()

    with pytest.raises(ValueError, match="ds_split only allows"):
        dynamic_model = DynamicSelection(
            ensemble=ensemble_model,  # type: ignore
            algorithm="METADES",
            ds_split=ds_split,
        )
        dynamic_model.fit(df_iris[X], df_iris[y])
