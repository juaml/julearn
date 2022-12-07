from typing import Union, List, Set

from ..utils.logging import raise_error
from sklearn.compose import make_column_selector


def change_column_type(column, new_type):
    return "__:type:__".join(column.split("__:type:__")[0:1] + [new_type])


def get_column_type(column):
    return column.split("__:type:__")[1]


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
    elif "__:type:__" in apply_to or apply_to in ["target", ["target"]]:
        pattern = apply_to
    else:
        pattern = f"(?__:type:__{apply_to})"

    return pattern


class ColumnTypes:
    """Class to hold types in regards to a pd.DataFrame Column.

    Parameters
    ----------
    column_types : ColumnTypes or str or list of str or set of str
        One str representing on type if columns or a list of these.
        Instead of a str you can also provide a ColumnTypes itself.
    """

    def __init__(
        self, column_types: Union[List[str], Set[str], str, "ColumnTypes"]
    ):
        if isinstance(column_types, ColumnTypes):
            _types = column_types._column_types
        elif isinstance(column_types, str):
            _types = set([column_types])
        elif not isinstance(column_types, Set):
            _types = set(column_types)
        else:
            raise_error(f"Cannot construct a ColumnType from {column_types}")
        self._column_types = _types

    def add(
        self, column_types: Union[List[str], Set[str], str, "ColumnTypes"]
    ):
        """Add more column_types to the column_types

        Parameters
        ----------
        column_types : ColumnTypes or str or list of str or ColumnTypes
            One str representing on type if columns or a list of these.
            Instead of a str you can also provide a ColumnTypes itself.


        Returns
        -------
        self: ColumnTypes
            The updates ColumnTypes.

        """
        if not isinstance(column_types, ColumnTypes):
            column_types = ColumnTypes(column_types)
        self._column_types.update(column_types)
        return self

    @property
    def pattern(self):
        return self._to_pattern()

    def to_type_selector(self):
        """Create a type selector usbale by sklearn.compose.ColumnTransformer
        from ColumnTypes.
        """
        return make_type_selector(self.pattern)

    def _to_pattern(self):
        """Converts column_types to pattern/regex usable to make a
        column_selector.

        Returns
        -------
        pattern: str
            The pattern/regex that matches all the column types

        """
        if "*" in self._column_types or ".*" in self._column_types:
            pattern = ".*"
        else:
            types_patterns = []
            for t_type in self._column_types:
                if "__:type:__" in t_type:
                    t_pattern = t_type
                if "target" == t_type:
                    t_pattern = t_type
                else:
                    t_pattern = f"__:type:__{t_type}"
                types_patterns.append(t_pattern)

            pattern = f"(?:{types_patterns[0]}"
            if len(types_patterns) > 1:
                for t in types_patterns[1:]:
                    pattern += rf"|{t}"
            pattern += r")"
        return pattern

    def __eq__(self, other: Union["ColumnTypes", str]):
        other = other if isinstance(other, ColumnTypes) else ColumnTypes(other)
        return self._column_types == other._column_types

    def __iter__(self):
        return self._column_types.__iter__()
