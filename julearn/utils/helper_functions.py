import re


def pick_columns(regexes, columns):
    if not isinstance(regexes, list):
        regexes = [regexes]

    pick = [
        col
        for col in columns
        if any([re.search(exp, col) for exp in regexes])
    ]

    return pick
