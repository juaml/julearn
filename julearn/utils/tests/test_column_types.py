from julearn.utils import pick_columns, make_type_selector
from pandas.testing import assert_frame_equal
import pytest


def test_pick_using_column_name():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['conf_2', 'Feat_3']

    assert regexes == pick_columns(regexes, columns)

    columns = ['Feat_3', 'conf_1', 'conf_2', 'feat_1', 'feat_2']
    regexes = ['conf_2', 'Feat_3']

    assert regexes == pick_columns(regexes, columns)

    columns = ['120', '121', '122', '123', '124', '125']
    regexes = ['12']
    msg = r"following are missing: \['12'\]"
    with pytest.raises(ValueError, match=msg):
        pick_columns(regexes, columns)

    columns = ['120', '121', '122', '123', '124', '125']
    regexes = ['2']
    msg = r"following are missing: \['2'\]"
    with pytest.raises(ValueError, match=msg):
        pick_columns(regexes, columns)

    columns = ['120', '121', '122', '123', '124', '125']
    regexes = ['24']
    msg = r"following are missing: \['24'\]"
    with pytest.raises(ValueError, match=msg):
        pick_columns(regexes, columns)

    columns = ['120', '121', '122', '123', '124', '125']
    regexes = ['122', '125', '130']
    msg = r"following are missing: \['130'\]"
    with pytest.raises(ValueError, match=msg):
        pick_columns(regexes, columns)

    columns = ['120', '121', '122', '123', '124', '125']
    regexes = ['122', '125']
    assert regexes == pick_columns(regexes, columns)


def test_pick_using_regex_match():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['.*conf.*', '.*feat.*']

    assert columns[:-1] == pick_columns(regexes, columns)


def test_pick_using_regex_and_column_name_match():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['.*conf.*', '.*feat.*', 'Feat_3']

    assert columns == pick_columns(regexes, columns)


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
