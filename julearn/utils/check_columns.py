"""Provide checks that X_types are valid."""

from .logging import raise_error


def check_columns(X, X_types):
    """Check validity of X_types.

    Parameters
    ----------
    X : str or list of str
        Names of features (X).
    X_types: dict
        Dictionary of X_types.
    """
    if isinstance(X, str):
        X = [X]
    values_all = []
    for key, value in X_types.items():
        print(key, value)
        if isinstance(value, str):
            value = [value]
        values_all += value
        for v in value:
            if v not in X:
                raise_error(
                    f'{v} of type {key} not a valid column name in X.'
                )

    print(values_all)

    if len(values_all) != len(set(values_all)):
        raise_error('One column defined multiple times in X_types')
