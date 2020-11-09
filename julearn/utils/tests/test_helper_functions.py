from julearn.utils import pick_columns


def test_pick_using_column_name():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['conf_2', 'Feat_3']

    assert regexes == pick_columns(regexes, columns)


def test_pick_using_regex_match():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['.*conf*.', '.*feat*.']

    assert columns[:-1] == pick_columns(regexes, columns)


def test_pick_using_regex_and_column_name_match():
    columns = ['conf_1', 'conf_2', 'feat_1', 'feat_2', 'Feat_3']
    regexes = ['.*conf*.', '.*feat*.', 'Feat_3']

    assert columns == pick_columns(regexes, columns)
