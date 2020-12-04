import re


def pick_columns(regexes, columns):
    """Pick elements from a list based on matches to a list of regexs

    Parameters
    ----------
    regexes : str or list(str)
        List of regular expressions to match
    columns : list(str)
        Elements to pick

    Returns
    -------
    picks : list(str)
        A list will all the elements from columns that match at least one
        regexp in regexes

    Raises
    ------
    ValueError
        If one or more regexes do not match any element in columns

    """
    if not isinstance(regexes, list):
        regexes = [regexes]

    picks = [
        col
        for col in columns
        if any([re.search(exp, col) for exp in regexes])
    ]
    unmatched = []
    for exp in regexes:
        if not any([re.search(exp, col) for col in columns]):
            unmatched.append(exp)
    if len(unmatched) > 0:
        raise ValueError(
            'All elements must be matched'
            f'The following are missing: {unmatched}')

    return picks


def change_column_type(column, new_type):
    return '__:type:__'.join(column.split('__:type:__')[0:1] + [new_type])


def get_column_type(column):
    return column.split('__:type:__')[1]
