from julearn.base import ColumnTypes, make_type_selector
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


@pytest.mark.parametrize(
    "column_types,pattern,resulting_column_types",
    [
        (["continuous"], "(?:__:type:__continuous)", ["continuous"]),
        ("continuous", "(?:__:type:__continuous)", ["continuous"]),
        (ColumnTypes("continuous"),
         "(?:__:type:__continuous)", ["continuous"]),
        (["continuous", "categorical"],
         "(?:__:type:__continuous|__:type:__categorical)",
         ["continuous", "categorical"],
         ),
        (ColumnTypes(["continuous", "categorical"]),
         "(?:__:type:__continuous|__:type:__categorical)",
         ["continuous", "categorical"],),
        ([ColumnTypes("continuous"), ColumnTypes(["categorical"])],
         "(?:__:type:__continuous|__:type:__categorical)",
         ["continuous", "categorical"],),
        ("*", ".*", ["*"]),
        (["*"], ".*", ["*"]),
        (".*", ".*", [".*"]),
        ([".*"], ".*", [".*"]),
    ])
def test_ColumnTypes_basics(column_types, pattern, resulting_column_types):

    ct = ColumnTypes(column_types)
    assert ct.column_types == resulting_column_types
    assert ct.pattern == pattern


@pytest.mark.parametrize(
    "column_types,resulting_column_types,selection",
    [
        (["continuous"], ["continuous"]*4, slice(0, 4),),
        (["continuous"], ["continuous"]*3 + ["cat"], slice(0, 3),),
        (["cont", "cat"],
         ["cont"]*3 + ["cat"], slice(0, 4),),
        (["continuous"], [""]*4, slice(0, 4),),
        (".*", ["continuous", "duck", "quak", "B"], slice(0, 4),),
    ]
)
def test_ColumnTypes_to_column_selector(
        X_iris, column_types, resulting_column_types, selection):
    _column_types = [col or "continuous" for col in resulting_column_types]
    X_iris.columns = [f"{col.split('__:type:__')[0]}__:type:__{ctype}"
                      for col, ctype in zip(X_iris.columns, _column_types)
                      ]
    col_true_selected = X_iris.iloc[:, selection].columns.tolist()
    col_selected = ColumnTypes(column_types).to_type_selector()(X_iris)
    assert col_selected == col_true_selected


@pytest.mark.parametrize(
    "left,right,equal",
    [
        (ColumnTypes(["continuous"]), ["continuous"], True),
        (ColumnTypes(["continuous"]), "continuous", True),
        (ColumnTypes(["continuous"]), ColumnTypes("continuous"), True),
        (ColumnTypes(["continuous", "cat"]), ["continuous", "cat"], True),
        (ColumnTypes(["continuous", "cat"]), "continuous", False),
        (ColumnTypes(["cont", "cat"]), ColumnTypes("continuous"), False),
    ]
)
def test_ColumnTypes_equivelance(left, right, equal):
    assert (left == right) == equal


def test_ColumnTypes_equivelance_error():
    with pytest.raises(ValueError,
                       match="Comparison with ColumnTypes only"):
        ColumnTypes(["cont"]) == 7


@pytest.mark.parametrize(
    "left,right,result",
    [
        (ColumnTypes(["continuous"]),
         ["continuous"], ColumnTypes(["continuous"])),
        (ColumnTypes(["cont"]), "cat", ColumnTypes(["cont", "cat"])),
    ]
)
def test_ColumnTypes_add(left, right, result):
    summed = (left.add(right))
    assert summed == result
