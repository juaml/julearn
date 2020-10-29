import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score,
                                     RepeatedKFold,
                                     GridSearchCV)
from sklearn.preprocessing import LabelBinarizer
from seaborn import load_dataset
from julearn import run_cross_validation


def _test_scoring(X, y, data, api_params, sklearn_model, scorers, sk_y=None):
    sk_X = data[X].values
    if sk_y is None:
        sk_y = data[y].values

    for scoring in scorers:
        actual = run_cross_validation(
            X=X, y=y, data=data, seed=42, scoring=scoring, **api_params)

        np.random.seed(42)
        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        expected = cross_val_score(sklearn_model, sk_X, sk_y, cv=cv,
                                   scoring=scoring)

        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])


def test_simple_binary():
    """Test simple binary classification"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    scorers = ['accuracy', 'balanced_accuracy']
    api_params = {'model': 'svm'}
    clf = make_pipeline(StandardScaler(), svm.SVC())
    _test_scoring(X, y, data=df_iris, api_params=api_params, sklearn_model=clf,
                  scorers=scorers)

    # now let's try target-dependent scores
    scorers = ['recall', 'precision', 'f1']
    sk_y = (df_iris[y].values == 'setosa').astype(np.int)
    api_params = {'model': 'svm', 'pos_labels': 'setosa'}
    clf = make_pipeline(StandardScaler(), svm.SVC())
    _test_scoring(X, y, data=df_iris, api_params=api_params, sklearn_model=clf,
                  scorers=scorers, sk_y=sk_y)

    # now let's try proba-dependent scores
    scorers = ['roc_auc']
    sk_y = (df_iris[y].values == 'setosa').astype(np.int)
    model = svm.SVC(probability=True)
    api_params = {'model': model, 'pos_labels': 'setosa'}
    clf = make_pipeline(StandardScaler(), svm.SVC())
    _test_scoring(X, y, data=df_iris, api_params=api_params, sklearn_model=clf,
                  scorers=scorers, sk_y=sk_y)


def test_scoring_y_transformer():
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scorers = ['accuracy', 'balanced_accuracy']
    for scoring in scorers:
        y_transformer = LabelBinarizer()
        actual = run_cross_validation(
            X=X, y=y, data=df_iris, model='svm', preprocess_y=y_transformer,
            seed=42, scoring=scoring)

        # Now do the same with scikit-learn
        clf = make_pipeline(StandardScaler(), svm.SVC(probability=True))

        np.random.seed(42)
        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        expected = cross_val_score(clf, sk_X, sk_y, cv=cv, scoring=scoring)

        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])


def test_set_hyperparam():
    """Test setting one hyperparmeter"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scoring = 'roc_auc'
    t_sk_y = (sk_y == 'setosa').astype(np.int)
    hyperparameters = {'svm__probability': True}

    actual = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', hyperparameters=hyperparameters,
        seed=42, scoring=scoring, pos_labels='setosa')

    # Now do the same with scikit-learn
    clf = make_pipeline(StandardScaler(), svm.SVC(probability=True))

    np.random.seed(42)
    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    expected = cross_val_score(clf, sk_X, t_sk_y, cv=cv, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_tune_hyperparam():
    """Test tunning one hyperparmeter"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scoring = 'accuracy'
    hyperparameters = {'svm__C': [0.01, 0.001]}

    actual = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', hyperparameters=hyperparameters,
        seed=42, scoring=scoring)

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_inner = RepeatedKFold(n_splits=5, n_repeats=5)
    cv_outer = RepeatedKFold(n_splits=5, n_repeats=5)

    clf = make_pipeline(StandardScaler(), svm.SVC())
    gs = GridSearchCV(clf, {'svc__C': [0.01, 0.001]}, cv=cv_inner)

    expected = cross_val_score(gs, sk_X, sk_y, cv=cv_outer, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])
