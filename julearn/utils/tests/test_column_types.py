from julearn.utils import make_type_selector
import pytest


@pytest.mark.parametrize(
    "pattern,column_types,selection",
    [
        ("(?:__:type:__continuous)", ["continuous"]*4, slice(0, 4),),
        ("(?:__:type:__continuous)", ["continuous"]*3 + ["cat"], slice(0, 3),),
        ("(?:__:type:__cont|__:type:__cat)",
         ["cont"]*3 + ["cat"], slice(0, 4),),
        ("(?:__:type:__continuous)", [""]*4, slice(0, 4),),
        (".*", ["continuous", "duck", "quak", "B"], slice(0, 4),),
    ]
)
def test_make_column_selector(X_iris, pattern, column_types, selection):
    column_types = [col or "continuous" for col in column_types]
    X_iris.columns = [f"{col.split('__:type:__')[0]}__:type:__{ctype}"
                      for col, ctype in zip(X_iris.columns, column_types)
                      ]
    col_true_selected = X_iris.iloc[:, selection].columns.tolist()
    col_selected = make_type_selector(pattern)(X_iris)
    assert col_selected == col_true_selected
