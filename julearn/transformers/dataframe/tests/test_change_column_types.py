"""Provide tests for the ChangeColumnTypes transformer."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import TYPE_CHECKING

from julearn.transformers.dataframe.change_column_types import (
    ChangeColumnTypes,
)


if TYPE_CHECKING:
    import pandas as pd


def test_change_column_types(df_typed_iris: "pd.DataFrame") -> None:
    """Test ChangeColumnTypes transformer.

    Parameters
    ----------
    df_typed_iris : pd.DataFrame
        The iris dataset with typed features.

    """
    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.loc[:, "species"]

    ct = ChangeColumnTypes(
        X_types_renamer={"continuous": "chicken"}, apply_to="*"
    )
    ct.fit(X, y)
    Xt = ct.transform(X)
    Xt_colnames = [x.split("__:")[0] for x in list(ct.get_feature_names_out())]
    assert all(col.endswith("__:type:__chicken") for col in list(Xt.columns))
    assert all(X == Xt_colnames)
