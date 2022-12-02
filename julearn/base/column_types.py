from typing import Union, List
from .. utils.logging import raise_error
from sklearn.compose import make_column_selector


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
    elif "__:type:__" in apply_to or apply_to in ["target", ["target"]]:
        pattern = apply_to
    else:
        pattern = f"(?__:type:__{apply_to})"

    return pattern


class ColumnTypes:
    def __init__(
        self,
        column_types: Union[List[Union[str, 'ColumnTypes']],
                            str, 'ColumnTypes']
    ):
        self.column_types = column_types

    def add(self, column_types: Union[List[str], str]):
        column_types = self.ensure_column_types(column_types)
        self.column_types = self.column_types.extend(column_types)

    @property
    def column_types(self):
        return self._column_types

    @column_types.setter
    def column_types(self, column_types: Union[List[str], str]):
        self._column_types = self.ensure_column_types(column_types)
        self._pattern = self._to_pattern(self._column_types)

    @property
    def pattern(self):
        return self._pattern

    def to_type_selector(self):
        return make_type_selector(self.pattern)

    @staticmethod
    def ensure_column_types(column_types):
        if not isinstance(column_types, (list, str, ColumnTypes)):
            raise_error(
                "ColumnType needs to be provided a list, str or ColumnTypes,"
                f" but got {column_types} with type = {type(column_types)}."
            )
        if not isinstance(column_types, list):
            column_types = [column_types]

        out = []
        for column_type in column_types:
            if isinstance(column_type, ColumnTypes):
                out.extend(column_type.column_types)
            elif isinstance(column_type, str):
                out.append(column_type)
            else:
                raise_error(
                    "Each entry of column_types needs to be a str,"
                    f" but{column_type} is of type {type(column_type)}."
                )
        return out

    @staticmethod
    def _to_pattern(column_types):
        if column_types in [".*", [".*"], "*", ["*"]]:
            pattern = ".*"
        elif isinstance(column_types, list) or isinstance(column_types, tuple):
            types = [f"__:type:__{_type}" for _type in column_types]

            pattern = f"(?:{types[0]}"
            if len(types) > 1:
                for t in types[1:]:
                    pattern += rf"|{t}"
            pattern += r")"
        elif ("__:type:__" in column_types or
              column_types in ["target", ["target"]]):
            pattern = column_types
        else:
            pattern = f"(?__:type:__{column_types})"

        return pattern

    def __eq__(self, other: Union[str, List[str], "ColumnTypes"]):
        if not isinstance(other, (str, list, ColumnTypes)):
            raise_error(
                "Comparison with ColumnTypes only allowed for "
                "following types: str, list, ColumnTypes. "
                f"But you provided {type(other)}"

            )
        other = other if isinstance(other, ColumnTypes) else ColumnTypes(other)
        return self.column_types == other.column_types

    def __iter__(self):
        return self.column_types.__iter__()
