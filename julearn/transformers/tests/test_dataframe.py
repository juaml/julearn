# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from julearn.transformers import DropColumns, FilterColumns, SetColumnTypes

X = pd.DataFrame(
    dict(A=np.arange(10), B=np.arange(10, 20), C=np.arange(30, 40))
)

X_with_types = pd.DataFrame(
    {
        "a__:type:__continuous": np.arange(10),
        "b__:type:__continuous": np.arange(10, 20),
        "c__:type:__confound": np.arange(30, 40),
        "d__:type:__confound": np.arange(40, 50),
        "e__:type:__categorical": np.arange(40, 50),
        "f__:type:__categorical": np.arange(40, 50),
    }
)


def test_DropColumns():
    drop_columns = DropColumns(apply_to=".*__:type:__confound")
    drop_columns.fit(X_with_types)
    X_trans = drop_columns.transform(X_with_types)

    kept_cols = X_with_types.columns[drop_columns.get_support()].to_list()
    kept_cols_2 = X_with_types.iloc[
        :, drop_columns.get_support(True)
    ].columns.to_list()
    assert_frame_equal(X_trans, X_with_types[kept_cols])
    assert_array_equal(X_trans, X_with_types[kept_cols_2])
    assert_frame_equal(
        X_with_types.drop(
            columns=["c__:type:__confound", "d__:type:__confound"]
        ),
        X_trans,
    )


def test_FilterColumns():
    filter = FilterColumns(
        apply_to=["continuous", "categorical"],
        keep=["continuous"]
    )
    filter.set_output(transform="pandas").fit(X_with_types)
    X_trans = filter.transform(X_with_types)
    assert list(X_trans.columns) == [
        "a__:type:__continuous", "b__:type:__continuous"]


def test_SetDtype(X_iris, X_types_iris):
    _X_types_iris = {} if X_types_iris is None else X_types_iris
    X_iris_with_types = (X_iris
                         .copy()
                         .rename(columns={
                             col: f"{col}__:type:__{dtype}"
                             for dtype, columns in _X_types_iris.items()
                             for col in columns
                         })
                         .rename(
                             columns=lambda col: (
                                 col
                                 if "__:type:__" in col
                                 else f"{col}__:type:__continuous"
                             ))
                         )
    st = SetColumnTypes(X_types_iris).set_output(transform="pandas")
    Xt = st.fit_transform(X_iris)
    Xt_iris_with_types = st.fit_transform(X_iris_with_types)
    assert_frame_equal(Xt, X_iris_with_types)
    assert_frame_equal(Xt_iris_with_types, X_iris_with_types)


def test_SetDtype_input_validation(X_iris):
    with pytest.raises(
        ValueError,
            match="Each value of X_types must be either a list or a tuple."):
        SetColumnTypes(dict(confound="chicken")).fit(X_iris)
