from julearn.transformers.dataframe.change_column_types import (
    ChangeColumnTypes,
)


def test_change_column_types(df_typed_iris):
    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.loc[:, "species"]

    ct = ChangeColumnTypes(
        X_types_renamer=dict(continuous="chicken"), apply_to="*")
    ct.fit(X, y)
    Xt = ct.transform(X)
    Xt_colnames = [x.split("__:")[0] for x in list(ct.get_feature_names_out())]
    assert all([col.endswith("__:type:__chicken") for col in list(Xt.columns)])
    assert all(X == Xt_colnames)
