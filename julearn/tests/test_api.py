import numpy as np
from numpy.testing import assert_array_equal
from sklearn import svm
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score,
                                     RepeatedKFold,
                                     GridSearchCV)
from sklearn.preprocessing import LabelBinarizer
from seaborn import load_dataset
from julearn import run_cross_validation


def _compare_models(clf1, clf2):
    if isinstance(clf1, svm.SVC):
        idx1 = np.argsort(clf1.support_)
        v1 = clf1.support_vectors_[idx1]
        idx2 = np.argsort(clf2.support_)
        v2 = clf2.support_vectors_[idx2]
    else:
        raise NotImplementedError(
            f'Model comparison for {clf1} not yet implemented.')
    assert_array_equal(v1, v2)


def _test_scoring(X, y, data, api_params, sklearn_model, scorers, sk_y=None):
    sk_X = data[X].values
    if sk_y is None:
        sk_y = data[y].values

    for scoring in scorers:
        actual, actual_estimator = run_cross_validation(
            X=X, y=y, data=data, seed=42, scoring=scoring,
            return_estimator=True, **api_params)

        np.random.seed(42)
        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        expected = cross_val_score(sklearn_model, sk_X, sk_y, cv=cv,
                                   scoring=scoring)

        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])

        # Compare the models
        clf1 = actual_estimator.dataframe_pipeline.steps[-1][1]
        clf2 = clone(sklearn_model).fit(sk_X, sk_y).steps[-1][1]
        _compare_models(clf1, clf2)


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
    """Test scoring with y transformer"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    # sk_X = df_iris[X].values
    sk_y = df_iris[y].values
    clf = make_pipeline(StandardScaler(), svm.SVC(probability=True))
    y_transformer = LabelBinarizer()

    scorers = ['accuracy', 'balanced_accuracy']
    api_params = {'model': 'svm', 'preprocess_y': y_transformer}
    _test_scoring(X, y, data=df_iris, api_params=api_params, sklearn_model=clf,
                  scorers=scorers, sk_y=sk_y)


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
    model_selection = {'hyperparameters': hyperparameters}

    actual, actual_estimator = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_selection=model_selection,
        seed=42, scoring=scoring, pos_labels='setosa', return_estimator=True)

    # Now do the same with scikit-learn
    clf = make_pipeline(StandardScaler(), svm.SVC(probability=True))

    np.random.seed(42)
    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    expected = cross_val_score(clf, sk_X, t_sk_y, cv=cv, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    # Compare the models
    clf1 = actual_estimator.dataframe_pipeline.steps[-1][1]
    clf2 = clone(clf).fit(sk_X, sk_y).steps[-1][1]
    _compare_models(clf1, clf2)


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
    model_selection = {'hyperparameters': hyperparameters}

    actual, actual_estimator  = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_selection=model_selection,
        seed=42, scoring=scoring, return_estimator=True)

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=5, n_repeats=5)
    cv_inner = RepeatedKFold(n_splits=5, n_repeats=5)

    clf = make_pipeline(StandardScaler(), svm.SVC())
    gs = GridSearchCV(clf, {'svc__C': [0.01, 0.001]}, cv=cv_inner)

    expected = cross_val_score(gs, sk_X, sk_y, cv=cv_outer, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    # Compare the models
    clf1 = actual_estimator.best_estimator_.dataframe_pipeline.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    _compare_models(clf1, clf2)

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=3, n_repeats=1)

    scoring = 'accuracy'
    gs_scoring = 'f1'
    hyperparameters = {'svm__C': [0.01, 0.001]}
    model_selection = {'hyperparameters': hyperparameters,
                       'scoring': gs_scoring,
                       'cv': cv_inner}

    actual, actual_estimator  = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_selection=model_selection,
        seed=42, scoring=scoring, return_estimator=True, pos_labels=['setosa'],
        cv=cv_outer)

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=3, n_repeats=1)

    clf = make_pipeline(StandardScaler(), svm.SVC())
    gs = GridSearchCV(clf, {'svc__C': [0.01, 0.001]}, cv=cv_inner,
                      scoring=gs_scoring)
    sk_y = (sk_y == 'setosa').astype(np.int)
    expected = cross_val_score(gs, sk_X, sk_y, cv=cv_outer, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    # Compare the models
    clf1 = actual_estimator.best_estimator_.dataframe_pipeline.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    _compare_models(clf1, clf2)
