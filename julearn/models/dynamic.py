"""Provide support for DESlib models."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv, train_test_split

from ..utils import raise_error
from ..utils.typing import DataLike, ModelLike


_deslib_algorithms = {
    "METADES": "des",
    "KNORAU": "des",
    "KNORAE": "des",
    "DESP": "des",
    "KNOP": "des",
    "OLA": "dcs",
    "MCB": "dcs",
    "SingleBest": "static",
    "StaticSelection": "static",
    "StackedClassifier": "static",
}


class DynamicSelection(BaseEstimator):
    """Model to use dynamic selection algorithms from DESlib.

    Parameters
    ----------
    ensemble : ModelLike
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
        in case you use a float for ds_split (default is None).
    random_state_algorithm : int, optional
        random state to get reproducible Deslib algorithm models
        (default is None).
    **kwargs : Any
        Any additional parameters to pass to the deslib algorithm.

    """

    def __init__(
        self,
        ensemble: ModelLike,
        algorithm: str,
        ds_split: float = 0.2,
        random_state: Optional[int] = None,
        random_state_algorithm: Optional[int] = None,
        **kwargs: Any,
    ):
        self.ensemble = ensemble
        self.algorithm = algorithm
        self.ds_split = ds_split
        self.random_state = random_state
        self.random_state_algorithm = random_state_algorithm
        self._ds_params = kwargs

    def fit(
        self, X: DataLike, y: DataLike  # noqa: N803
    ) -> "DynamicSelection":
        """Fit the model.

        Parameters
        ----------
        X : DataLike
            The data to fit the model on.
        y : DataLike
            The target data.

        Returns
        -------
        DynamicSelection
            The fitted model.

        """
        # create the splits to train ensemble and dynamic model
        if isinstance(self.ds_split, float):
            X_train, X_dsel, y_train, y_dsel = train_test_split(
                X, y, test_size=self.ds_split, random_state=self.random_state
            )
        else:
            # TODO: Sklearn should fix type hints for check_cv
            cv_split = check_cv(self.ds_split)
            if (n_splits := cv_split.get_n_splits()) != 1:  # type: ignore
                raise_error(
                    "ds_split only allows for one train and one test split. "
                    f"You tried to use {n_splits} splits."
                )

            train, test = next(iter(cv_split.split(X, y)))  # type: ignore
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train, :]
                X_dsel = X.iloc[test, :]
            else:
                X_train = X[train]
                X_dsel = X[test]

            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_train = y.iloc[train]
                y_dsel = y.iloc[test]
            else:
                y_train = y[train]
                y_dsel = y[test]

        self.ensemble.fit(X_train, y_train)
        self._dsmodel = self._get_algorithm()
        self._dsmodel.fit(X_dsel, y_dsel)

        return self

    def predict(self, X: DataLike) -> DataLike:  # noqa: N803
        """Predict using the model.

        Parameters
        ----------
        X : DataLike
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.

        """
        return self._dsmodel.predict(X)

    def predict_proba(self, X: DataLike) -> np.ndarray:  # noqa: N803
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : DataLike
            The data to predict on.

        Returns
        -------
        np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._dsmodel.predict_proba(X)

    def score(
        self,
        X: DataLike,  # noqa: N803
        y: DataLike,
        sample_weight: Optional[DataLike] = None,
    ) -> float:
        """Score the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.
        y : DataLike
            The true target values.
        sample_weight : DataLike, optional
            Sample weights to use when computing the score (default None).

        Returns
        -------
        float
            The score.

        """
        return self._dsmodel.score(X, y, sample_weight)

    def _get_algorithm(self) -> Any:
        """Get the deslib algorithm.

        Returns
        -------
        Any
            The deslib algorithm object.

        """
        try:
            import deslib  # type: ignore # noqa: F401
        except ImportError:
            raise_error(
                "DynamicSelection requires deslib library: "
                "https://deslib.readthedocs.io/en/latest/index.html"
            )

        import_algorithm = _deslib_algorithms.get(self.algorithm)
        if import_algorithm is None:
            raise_error(
                f"{self.algorithm} is not a valid or supported "
                f"deslib algorithm. "
                f"Valid options are {_deslib_algorithms.keys()}"
            )
        else:
            exec(
                f"from deslib.{import_algorithm} import {self.algorithm} "
                "as ds_algo",
                locals(),
                globals(),
            )

        out = ds_algo(  # type: ignore # noqa: F821
            pool_classifiers=self.ensemble,
            random_state=self.random_state_algorithm,
            **self._ds_params,
        )
        return out
