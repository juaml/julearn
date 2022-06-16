# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from inspect import signature
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from .. utils import raise_error


class TargetTransfromerWrapper(TransformerMixin, BaseEstimator):

    def __init__(self, transformer, **params):
        """Using a sklearn transformer and applying them to the target/y.

        Parameters
        ----------
        transformer : sklearn.base.TransformerMixin
            Any sklearn compatible transformer.
        """
        self.transformer = transformer
        self.transformer.set_params(**params)

    def fit(self, X=None, y=None):

        self._validate_XY_input(X, y)
        if type(y) == pd.Series:
            self.transformer.fit(pd.DataFrame(y))
        else:
            self.transformer.fit(y.reshape(-1, 1))

        return self

    def transform(self, X=None, y=None):
        self._validate_XY_input(X, y)
        if type(y) == pd.Series:
            _y = pd.DataFrame(y)
        else:
            _y = y.reshape(-1, 1)
        _y = self.transformer.transform(_y)

        if type(_y) == pd.DataFrame:
            return _y.iloc[:, 0]
        else:
            return _y.reshape(-1)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def get_params(self, deep=True):
        params = self.transformer.get_params()
        params['transformer'] = self.transformer
        return params

    def set_params(self, **params):
        if params.get('transformer') is None:
            self.transformer.set_params(**params)
        else:
            self.transformer = params.pop('transformer')
            self.transformer.set_params(**params)
        return self

    def _validate_XY_input(self, X, y):
        if y is None:
            raise_error('y should not be None when transforming it')


def is_targettransformer(trans):

    is_targettrans = True
    if hasattr(trans, 'transform'):
        try:
            signature(trans.transform).parameters['y']
        except KeyError:
            is_targettrans = False
    else:
        raise_error(f'is_targettransformer can only be applied to '
                    f'sklearn compatible transformers. Object {trans} has no '
                    '.transform method. Therefore, it is no transformer.')

    return is_targettrans
