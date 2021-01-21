from julearn.utils.logging import raise_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, check_cv

# cv = StratifiedShuffleSplit(1, test_size_size=0.2)
# cv = [[[0, 1], [2, 3]]]
# cv = 0.2


class DynamicSelection(BaseEstimator):

    def __init__(self, ensemble, algorithm, ds_split=.2,
                 random_state=None, **kwargs):
        """Creating a Dynamic model using the deslib library.

        Parameters
        ----------
        ensemble : obj
            sklearn compatible ensemble model. E.g RandomForest 
        algorithm : str
            algorithm from deslib to make the model dynamic.
            Options:

            * METADES

        ds_split : float, optional
            how to split the training data.
            One split is used to train the ensemble model and 
            the other to drain the dynamic algorithm, by default .2
        random_state : int, optional
            random state to get reproducible train test splits
            in case you use a float for ds_split, by default None
        """

        self.ensemble = ensemble
        self.algorithm = algorithm
        self.ds_split = ds_split
        self.random_state = random_state
        self._ds_params = kwargs

    def fit(self, X, y=None):

        # create the splits to train ensemble and dynamic model
        if isinstance(self.ds_split, float):
            X_train, X_dsel, y_train, y_dsel = train_test_split(
                X, y, test_size=self.ds_split,
                random_state=self.random_state)
        else:
            t_split = check_cv(self.ds_split)
            n_cv = self.cv.get_n_splits()
            if n_cv != 1:
                raise_error('The number of splits should be 2')

            train, test = list(t_split.split(X, y))[0]
            X_train = X[train]
            y_train = y[train]
            X_dsel = X[test]
            y_dsel = y[test]

        self.ensemble.fit(X_train, y_train)
        self._dsmodel = self.get_algortihm(pool_classifiers=self.ensemble,
                                           **self._ds_params)
        self._dsmodel.fit(X_dsel, y_dsel)

        return self

    def predict(self, X):
        return self._dsmodel.predict(X)

    def predict_proba(self, X):
        return self._dsmodel.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self._dsmodel.score(X, y, sample_weight)

    def get_algortihm(self, **kwargs):
        ds_algo = None

        try:
            import deslib  # noqa
        except ImportError:
            raise_error('DynamicSelection requires deslib library: '
                        'https://deslib.readthedocs.io/en/latest/index.html')

        if self.algorithm == 'METADES':
            from deslib.des import METADES as ds_algo

        return ds_algo(**kwargs)
