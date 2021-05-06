# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

import numpy as np
import pytest
from julearn.estimators.dynamic import DynamicSelection
from sklearn.ensemble import RandomForestClassifier
from seaborn import load_dataset

from deslib.dcs import (OLA, MCB)
from deslib.des import (DESP, KNORAU, METADES, KNOP, KNORAE)
from deslib.static import (StackedClassifier, SingleBest, StaticSelection)
from sklearn.model_selection import train_test_split, ShuffleSplit


all_algorithms = {'METADES': METADES,
                  'SingleBest': SingleBest,
                  'StaticSelection': StaticSelection,
                  'StackedClassifier': StackedClassifier,
                  'KNORAU': KNORAU,
                  'KNORAE': KNORAE,
                  'DESP': DESP,
                  'OLA': OLA,
                  'MCB': MCB,
                  'KNOP': KNOP
                  }


def test_algorithms():
    df_iris = load_dataset('iris')
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    for algorithm in all_algorithms.keys():
        seed = 42
        ds_split = .2

        # julearn
        np.random.seed(seed)
        ensemble_model = RandomForestClassifier(random_state=6)
        dynamic_model = DynamicSelection(
            ensemble=ensemble_model, algorithm=algorithm,
            random_state=seed, random_state_algorithm=seed, ds_split=ds_split)
        dynamic_model.fit(df_iris[X], df_iris[y])
        score_julearn = dynamic_model.score(df_iris[X], df_iris[y])
        pred_julearn = dynamic_model.predict(df_iris[X])

        # deslib
        np.random.seed(seed)
        X_train, X_dsel, y_train, y_dsel = train_test_split(
            df_iris[X], df_iris[y], test_size=ds_split, random_state=seed)

        pool_classifiers = RandomForestClassifier(random_state=6)
        pool_classifiers.fit(X_train, y_train)

        cls = all_algorithms[algorithm]
        model_deslib = cls(pool_classifiers, random_state=seed)
        model_deslib.fit(X_dsel, y_dsel)
        score_deslib = model_deslib.score(df_iris[X], df_iris[y])
        pred_deslib = model_deslib.predict(df_iris[X])

        assert score_deslib == score_julearn
        assert (pred_deslib == pred_julearn).all()

        if hasattr(model_deslib, 'predict_proba'):
            pred_proba_julearn = dynamic_model.predict_proba(df_iris[X])
            pred_proba_deslib = model_deslib.predict_proba(df_iris[X])
            assert (pred_proba_deslib == pred_proba_julearn).all()


def test_wrong_algo():
    df_iris = load_dataset('iris')
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    ensemble_model = RandomForestClassifier()

    with pytest.raises(ValueError, match='wrong is not a valid or supported'):
        dynamic_model = DynamicSelection(
            ensemble=ensemble_model, algorithm='wrong')
        dynamic_model.fit(df_iris[X], df_iris[y])


def test_ds_split_no_error():

    df_iris = load_dataset('iris')
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]
    df_iris = df_iris.sample(n=len(df_iris))
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    ensemble_model = RandomForestClassifier()

    train, test = train_test_split(
        np.arange(len(df_iris)), test_size=.4, shuffle=True)

    for ds_split in [.2, .3,
                     [(train, test)],
                     ShuffleSplit(n_splits=1)]:
        dynamic_model = DynamicSelection(
            ensemble=ensemble_model, algorithm='METADES',
            ds_split=ds_split
        )
        dynamic_model.fit(df_iris[X], df_iris[y])


def test_ds_split_error():

    df_iris = load_dataset('iris')
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]
    df_iris = df_iris.sample(n=len(df_iris))
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    ensemble_model = RandomForestClassifier()

    train, test = train_test_split(
        np.arange(len(df_iris)), test_size=.4, shuffle=True)

    for ds_split in [4, [(train, test), (train, test)],
                     ShuffleSplit(n_splits=2)]:
        with pytest.raises(ValueError, match='ds_split only allows'):
            dynamic_model = DynamicSelection(
                ensemble=ensemble_model, algorithm='METADES',
                ds_split=ds_split
            )
            dynamic_model.fit(df_iris[X], df_iris[y])
