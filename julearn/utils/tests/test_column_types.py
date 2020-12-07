from julearn.utils import pick_columns
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
