import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from julearn.transformers import get_transformer


def make_type_selector(pattern):
    def get_renamer(X):
        return {x: (x
                if "__:type:__" in x
                else f"{x}__:type:__continuous"
                    )
                for x in X.columns
                }

    def type_selector(X):

        renamer = get_renamer(X)
        _X = X.rename(columns=renamer)
        reverse_renamer = {
            new_name: name
            for name, new_name in renamer.items()}
        selected_columns = make_column_selector(pattern)(_X)
        return [reverse_renamer[col] if col in reverse_renamer else col
                for col in selected_columns]

    return type_selector


def get_default_patter(step):
    return "__:type:__continuous"


def get_step(step_name, pattern):
    return step_name, get_transformer(step_name), pattern


class PreprocessCreator:
    def __init__(self):
        self._steps = list()
        self._params = dict()

    def add(self, transformer, types="continuous", **params):

        if isinstance(types, list) or isinstance(types, tuple):
            types = [f"__:type:__{type}" for type in types]

            pattern = f"({types[0]}"
            if len(types) > 1:
                for t in types[1:]:
                    pattern += fr"|{t}"
            pattern += r")"
        else:
            pattern = f"__type__:{type}"

        if hasattr(transformer, "fit") and hasattr(transformer, "transform"):
            _name = (transformer.__name__
                     if transformer is None
                     else transformer)
        else:
            _name, _step, pattern = get_step(transformer, pattern)

        wrapped_name = self._check_name(f"wrapped_{_name}")
        self._steps.append(
            [wrapped_name, self.wrap_step(_name, _step, pattern)]
        )
        params = {f"{wrapped_name}__{_name}__{param}": val
                  for param, val in params.items()
                  }
        self._params = {**params, **self._params}

        return self

    @property
    def steps(self):
        return self._steps

    @property
    def param_grid(self):
        return self._params

    @classmethod
    def from_list(cls, transformers: list):

        preprocessor = cls()
        for transformer_name in transformers:
            preprocessor.add(transformer_name)
        return preprocessor

    def _check_name(self, name):

        count = np.array(
            [name == _name
             for _name, _ in self._steps
             ]).sum()
        return f"{name}_{count}" if count > 0 else name

    @staticmethod
    def wrap_step(name, step, pattern):
        return ColumnTransformer(
            [(name, step, make_type_selector(pattern))],
            verbose_feature_names_out=False, remainder="passthrough"
        )
