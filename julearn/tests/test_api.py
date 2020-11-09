# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np

from sklearn import svm
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold,
                                     GroupKFold,
                                     RepeatedKFold,
                                     GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import LabelBinarizer
from seaborn import load_dataset
import pytest

from julearn import run_cross_validation
from julearn.utils.testing import do_scoring_test, compare_models


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
    do_scoring_test(X, y, data=df_iris, api_params=api_params,
                    sklearn_model=clf, scorers=scorers)

    # now let's try target-dependent scores
    scorers = ['recall', 'precision', 'f1']
    sk_y = (df_iris[y].values == 'setosa').astype(np.int)
    api_params = {'model': 'svm', 'pos_labels': 'setosa'}
    clf = make_pipeline(StandardScaler(), svm.SVC())
    do_scoring_test(X, y, data=df_iris, api_params=api_params,
                    sklearn_model=clf, scorers=scorers, sk_y=sk_y)

    # now let's try proba-dependent scores
    scorers = ['roc_auc']
    sk_y = (df_iris[y].values == 'setosa').astype(np.int)
    model = svm.SVC(probability=True)
    api_params = {'model': model, 'pos_labels': 'setosa'}
    clf = make_pipeline(StandardScaler(), svm.SVC())
    do_scoring_test(X, y, data=df_iris, api_params=api_params,
                    sklearn_model=clf, scorers=scorers, sk_y=sk_y)


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
    do_scoring_test(X, y, data=df_iris, api_params=api_params,
                    sklearn_model=clf, scorers=scorers, sk_y=sk_y)


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

    with pytest.warns(RuntimeWarning,
                      match=r"Hyperparameter search CV"):
        model_params = {'cv': 5}
        _, _ = run_cross_validation(
            X=X, y=y, data=df_iris, model='svm',
            model_params=model_params,
            seed=42, scoring='accuracy', pos_labels='setosa',
            return_estimator=True)
    with pytest.warns(RuntimeWarning,
                      match=r"Hyperparameter search method"):
        model_params = {'search': 'grid'}
        _, _ = run_cross_validation(
            X=X, y=y, data=df_iris, model='svm',
            model_params=model_params,
            seed=42, scoring='accuracy', pos_labels='setosa',
            return_estimator=True)

    with pytest.warns(RuntimeWarning,
                      match=r"Hyperparameter search scoring"):
        model_params = {'scoring': 'accuracy'}
        _, _ = run_cross_validation(
            X=X, y=y, data=df_iris, model='svm',
            model_params=model_params,
            seed=42, scoring='accuracy', pos_labels='setosa',
            return_estimator=True)

    model_params = {'svm__probability': True}
    actual, actual_estimator = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm',
        model_params=model_params,
        seed=42, scoring=scoring, pos_labels='setosa',
        return_estimator=True)

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
    compare_models(clf1, clf2)


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

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    model_params = {'svm__C': [0.01, 0.001], 'cv': cv_inner}
    actual, actual_estimator  = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_params=model_params,
        cv=cv_outer, scoring=scoring, return_estimator=True)

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    clf = make_pipeline(StandardScaler(), svm.SVC())
    gs = GridSearchCV(clf, {'svc__C': [0.01, 0.001]}, cv=cv_inner)

    expected = cross_val_score(gs, sk_X, sk_y, cv=cv_outer, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    # Compare the models
    clf1 = actual_estimator.best_estimator_.dataframe_pipeline.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    compare_models(clf1, clf2)

    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    # Now randomized serach
    model_params = {'svm__C': [0.01, 0.001], 'cv': cv_inner,
                    'search': 'random', 'search_params': {'n_iter': 2}}
    actual, actual_estimator  = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_params=model_params,
        cv=cv_outer, scoring=scoring, return_estimator=True)

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    clf = make_pipeline(StandardScaler(), svm.SVC())
    gs = RandomizedSearchCV(clf, {'svc__C': [0.01, 0.001]}, cv=cv_inner,
                            n_iter=2)

    expected = cross_val_score(gs, sk_X, sk_y, cv=cv_outer, scoring=scoring)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    # Compare the models
    clf1 = actual_estimator.best_estimator_.dataframe_pipeline.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    compare_models(clf1, clf2)

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=3, n_repeats=1)

    scoring = 'accuracy'
    gs_scoring = 'f1'
    model_params = {'svm__C': [0.01, 0.001],
                    'scoring': gs_scoring,
                    'cv': cv_inner}

    actual, actual_estimator  = run_cross_validation(
        X=X, y=y, data=df_iris, model='svm', model_params=model_params,
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
    compare_models(clf1, clf2)


def test_consistency():
    """Test for consistency in the parameters"""
    df_iris = load_dataset('iris')
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    cv = StratifiedKFold(2)

    # Example 1: 3 classes, as strings

    # No error for multiclass
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             problem_type='multiclass_classification')

    # Error for binary
    with pytest.raises(ValueError, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv)

    # no error with pos_labels
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             pos_labels='setosa')

    # Warn with target transformer
    with pytest.warns(RuntimeWarning, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 preprocess_y='zscore')

    # Error for regression
    with pytest.raises(ValueError, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression')

    # Warn for regression with pos_labels
    match = 'but only 2 distinct values are defined'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression',
                                 pos_labels='setosa')

    # Warn for regression with y_transformer
    match = 'owever, a y transformer'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression',
                                 preprocess_y='zscore')

    # Example 2: 2 classes, as strings
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]

    # no error for binary
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv)

    # Warning for multiclass
    match = 'multiclass classification will be performed but only 2'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='multiclass_classification')

    # Error for regression
    with pytest.raises(ValueError, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression')

    # Warn for regression with pos_labels
    match = 'but only 2 distinct values are defined'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression',
                                 pos_labels='setosa')

    # Exampe 3: 3 classes, as integers
    df_iris = load_dataset('iris')
    le = LabelEncoder()
    df_iris['species'] = le.fit_transform(df_iris['species'].values)

    # No error for multiclass
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             problem_type='multiclass_classification')

    # Error for binary
    with pytest.raises(ValueError, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv)

    # no error with pos_labels
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             pos_labels=2)

    # Warn with target transformer
    with pytest.warns(RuntimeWarning, match='not suitable for'):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 preprocess_y='zscore')

    # no error for regression
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             problem_type='regression')

    # Warn for regression with pos_labels
    match = 'but only 2 distinct values are defined'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 problem_type='regression',
                                 pos_labels=2)

    # Groups parameters
    df_iris = load_dataset('iris')
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    df_iris['groups'] = np.random.randint(0, 3, len(df_iris))
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    groups = 'groups'
    match = 'groups was specified but the CV strategy'
    with pytest.warns(RuntimeWarning, match=match):
        _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                                 groups=groups)

    # No warning:
    cv = GroupKFold(2)
    _ = run_cross_validation(X=X, y=y, data=df_iris, model='svm', cv=cv,
                             groups=groups)
