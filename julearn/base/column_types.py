"""Implement column types."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Callable, List, Set, Union

from sklearn.compose import make_column_selector

from ..utils.logging import raise_error


ColumnTypesLike = Union[List[str], Set[str], str, "ColumnTypes"]


def change_column_type(column: str, new_type: str):
    """Change the type of a column.

    Parameters
    ----------
    column : str
        The column to change the type of.
    new_type : str
        The new type of the column.

    Returns
    -------
    str
        The new column name with the type changed.
    """
    return "__:type:__".join(column.split("__:type:__")[0:1] + [new_type])


def get_column_type(column):
    """Get the type of a column.

    Parameters
    ----------
    column : str
        The column to get the type of.

    Returns
    -------
    str
        The type of the column.
    """
    return column.split("__:type:__")[1]


def get_renamer(X_df):  # noqa: N803
    """Get the dictionary that will rename the columns to add the type.

    Parameters
    ----------
    X_df : pd.DataFrame
        The dataframe to rename the columns of.

    Returns
    -------
    dict
        The dictionary that will rename the columns.

    """
    return {
        x: (x if "__:type:__" in x else f"{x}__:type:__continuous")
        for x in X_df.columns
    }


class make_type_selector:
    """Make a type selector.

    This type selector is to be used with
    :class:`sklearn.compose.ColumnTransformer`

    Parameters
    ----------
    pattern : str
        The pattern to select the columns.

    Returns
    -------
    function
        The type selector.

    """

    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, X_df):  # noqa: N803
        """Select the columns based on the pattern.

        Parameters
        ----------
        X_df : pd.DataFrame
            The dataframe to select the columns of.

        Returns
        -------
        list
            The list of selected columns.

        """
        # Rename the columns to add the type if not present
        renamer = get_renamer(X_df)
        _X_df = X_df.rename(columns=renamer)
        reverse_renamer = {
            new_name: name for name, new_name in renamer.items()
        }

        # Select the columns based on the pattern
        selected_columns = make_column_selector(self.pattern)(_X_df)
        if len(selected_columns) == 0:
            raise_error(
                f"No columns selected with pattern {self.pattern} in "
                f"{_X_df.columns.to_list()}"
            )

        # Rename the column back to their original name
        return [
            reverse_renamer[col] if col in reverse_renamer else col
            for col in selected_columns
        ]


class ColumnTypes:
    """Class to hold types in regards to a pd.DataFrame Column.

    Parameters
    ----------
    column_types : ColumnTypes or str or list of str or set of str
        One str representing on type if columns or a list of these.
        Instead of a str you can also provide a ColumnTypes itself.
    """

    def __init__(self, column_types: ColumnTypesLike):
        if isinstance(column_types, ColumnTypes):
            _types = column_types._column_types.copy()
        elif isinstance(column_types, str):
            _types = {column_types}
        elif not isinstance(column_types, Set):
            _types = set(column_types)
        elif isinstance(column_types, Set):
            _types = column_types
        else:
            raise_error(f"Cannot construct a ColumnType from {column_types}")
        self._column_types = _types

    def add(self, column_types: ColumnTypesLike) -> "ColumnTypes":
        """Add more column_types to the column_types.

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
    def pattern(self) -> str:
        """Get the pattern/regex that matches all the column types."""
        return self._to_pattern()

    def to_type_selector(self) -> Callable:
        """Create a type selector from the ColumnType.

        The type selector is usable by
        :class:`sklearn.compose.ColumnTransformer`


        Returns
        -------
        Callable
            The type selector.
        """
        return make_type_selector(self.pattern)

    def _to_pattern(self):
        """Convert column_types to pattern/regex.

        This pattern is usable to make a column_selector.

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
                elif "target" == t_type:
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
        """Check if the column_types are equal to another column_types.

        Parameters
        ----------
        other : ColumnTypes or str
            The other column_types to compare to.

        Returns
        -------
        bool
            True if the column_types are equal, False otherwise.
        """
        other = other if isinstance(other, ColumnTypes) else ColumnTypes(other)
        return self._column_types == other._column_types

    def __iter__(self):
        """Iterate over the column_types."""

        return self._column_types.__iter__()

    def __repr__(self):
        """Get the representation of the ColumnTypes."""
        return (
            f"ColumnTypes<types={self._column_types}; pattern={self.pattern}>"
        )

    def copy(self) -> "ColumnTypes":
        """Get a copy of the ColumnTypes.

        Returns
        -------
        ColumnTypes
            The copy of the ColumnTypes.
        """
        return ColumnTypes(self)


def ensure_column_types(attr: ColumnTypesLike) -> ColumnTypes:
    """Ensure that the attribute is a ColumnTypes.

    Parameters
    ----------
    attr : ColumnTypes or str
        The attribute to check.

    Returns
    -------
    ColumnTypes
        The attribute as a ColumnTypes.
    """
    return ColumnTypes(attr) if not isinstance(attr, ColumnTypes) else attr
