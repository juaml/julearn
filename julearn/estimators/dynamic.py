# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL
from julearn.utils.logging import raise_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, check_cv

_deslib_algorithms = dict(
    METADES='des',
    KNORAU='des',
    KNORAE="des",
    DESP="des",
    KNOP="des",
    OLA="dcs",
    MCB="dcs",
    SingleBest="static",
    StaticSelection="static",
    StackedClassifier="static"
)


class DynamicSelection(BaseEstimator):

    def __init__(self, ensemble, algorithm, ds_split=.2,
                 random_state=None, random_state_algorithm=None, **kwargs):
        """Creating a Dynamic model using the deslib library.

        Parameters
        ----------
        ensemble : obj
            sklearn compatible ensemble model. E.g RandomForest
        algorithm : str
            algorithm from deslib to make the model dynamic.
            Options:

            * METADES
            * SingleBest
            * StaticSelection
            * StackedClassifier
            * KNORAU
            * KNORAE
            * DESP
            * OLA
            * MCB
            * KNOP

        ds_split : float, optional
            How to split the training data.
            One split is used to train the ensemble model and
            the other to train the dynamic algorithm, by default .2
            You can use any sklearn cv consistent cv splitter, but
            only with `n_splits = 1`.
        random_state : int, optional
            random state to get reproducible train test splits
            in case you use a float for ds_split, by default None
        """

        self.ensemble = ensemble
        self.algorithm = algorithm
        self.ds_split = ds_split
        self.random_state = random_state
        self.random_state_algorithm = random_state_algorithm
        self._ds_params = kwargs

    def fit(self, X, y=None):

        # create the splits to train ensemble and dynamic model
        if isinstance(self.ds_split, float):
            X_train, X_dsel, y_train, y_dsel = train_test_split(
                X, y, test_size=self.ds_split,
                random_state=self.random_state)
        else:
            cv_split = check_cv(self.ds_split)
            if cv_split.get_n_splits() != 1:
                raise_error(
                    'ds_split only allows for one train and one test split.\n'
                    f'You tried to use {cv_split.get_n_splits()} splits'
                )

            train, test = list(cv_split.split(X, y))[0]
            X_train = X.iloc[train, :]
            y_train = y.iloc[train]
            X_dsel = X.iloc[test, :]
            y_dsel = y.iloc[test]

        self.ensemble.fit(X_train, y_train)
        self._dsmodel = self.get_algorithm(
            pool_classifiers=self.ensemble,
            random_state=self.random_state_algorithm,
            ** self._ds_params)
        self._dsmodel.fit(X_dsel, y_dsel)

        return self

    def predict(self, X):
        return self._dsmodel.predict(X)

    def predict_proba(self, X):
        return self._dsmodel.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self._dsmodel.score(X, y, sample_weight)

    def get_algorithm(self, **kwargs):

        try:
            import deslib  # noqa
        except ImportError:
            raise_error('DynamicSelection requires deslib library: '
                        'https://deslib.readthedocs.io/en/latest/index.html')

        import_algorithm = _deslib_algorithms.get(self.algorithm)
        if import_algorithm is None:
            raise_error(f'{self.algorithm} is not a valid or supported '
                        f'deslib algorithm. '
                        f'Valid options are {_deslib_algorithms.keys()}'
                        )
        else:
            exec(
                f'from deslib.{import_algorithm} import {self.algorithm} '
                'as ds_algo', locals(), globals())

        return ds_algo(**kwargs)  # noqa
