import re
from .logging import raise_error
from sklearn.compose import make_column_selector


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

    picks = []
    matched = []
    # # Pick literal matches (in case regexes are not regexes but column names)
    for regex in regexes:
        picks += [
            col for col in columns
            if '__:type:__' in col and col.split('__:type:__')[0] == regex]
        if len(picks) > 0:
            matched.append(regex)

    for exp in regexes:
        cols = [
            col
            for col in columns
            if any([re.fullmatch(exp, col)])
        ]
        if len(cols) > 0:
            picks.extend(cols)
            matched.append(exp)

    unmatched = [exp for exp in regexes if exp not in matched]
    if len(unmatched) > 0:
        raise ValueError(
            'All elements must be matched. '
            f'The following are missing: {unmatched}')

    return picks


def change_column_type(column, new_type):
    return '__:type:__'.join(column.split('__:type:__')[0:1] + [new_type])


def get_column_type(column):
    return column.split('__:type:__')[1]


def make_type_selector(pattern):
    def get_renamer(X_df):
        return {
            x: (x if "__:type:__" in x else f"{x}__:type:__continuous")
            for x in X_df.columns
        }

    def type_selector(X_df):
        # Rename the columns to add the type if not present
        renamer = get_renamer(X_df)
        _X_df = X_df.rename(columns=renamer)
        reverse_renamer = {
            new_name: name for name, new_name in renamer.items()
        }

        # Select the columns based on the pattern
        selected_columns = make_column_selector(pattern)(_X_df)
        if len(selected_columns) == 0:
            raise_error(
                f"No columns selected with pattern {pattern} in "
                f"{_X_df.columns.to_list()}"
            )

        # Rename the column back to their original name
        return [
            reverse_renamer[col] if col in reverse_renamer else col
            for col in selected_columns
        ]

    return type_selector


def ensure_apply_to(apply_to):
    if apply_to in [".*", [".*"], "*", ["*"]]:
        pattern = ".*"
    elif isinstance(apply_to, list) or isinstance(apply_to, tuple):
        types = [f"__:type:__{_type}" for _type in apply_to]

        pattern = f"(?:{types[0]}"
        if len(types) > 1:
            for t in types[1:]:
                pattern += rf"|{t}"
        pattern += r")"
    elif "__:type:__" in apply_to:
        pattern = apply_to
    else:
        pattern = f"(?:__:type:__{apply_to})"

    return pattern
