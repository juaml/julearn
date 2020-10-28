import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, RepeatedKFold

from seaborn import load_dataset
from julearn import run_cross_validation


def test_simple_binary():
    """Test simple binary classification"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    actual = run_cross_validation(X=X, y=y, data=df_iris, model='svm', seed=42)

    # Now do the same with scikit-learn
    X = df_iris[X].values
    y = df_iris[y].values

    clf = make_pipeline(StandardScaler(), svm.SVC())

    np.random.seed(42)
    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    expected = cross_val_score(clf, X, y, cv=cv)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])
