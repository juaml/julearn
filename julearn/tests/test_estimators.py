# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from seaborn import load_dataset

from julearn.utils.testing import do_scoring_test


_binary_estimators = {
    'svm': SVC,
    'rf': RandomForestClassifier,
    'et': ExtraTreesClassifier,
    'dummy': DummyClassifier
}

_binary_params = {
    'rf': {'n_estimators': 10},
    'et': {'n_estimators': 10},
    'dummy': {'strategy': 'prior'}
}


def test_binary_estimators():
    """Test all estimators"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    for t_mname, t_model_class in _binary_estimators.items():
        m_params = _binary_params.get(t_mname, {})
        model_params = None
        if len(m_params) > 0:
            model_params = {
                f'{t_mname}__{t_param}': t_value
                for t_param, t_value in m_params.items()
            }
            t_model = t_model_class(**m_params)
        else:
            t_model = t_model_class()
        scorers = ['accuracy']
        api_params = {'model': t_mname, 'model_params': model_params}
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)
        if t_mname != 'dummy':
            # now let's try target-dependent scores
            scorers = ['recall', 'precision', 'f1']
            sk_y = (df_iris[y].values == 'setosa').astype(np.int)
            api_params = {'model': t_mname, 'pos_labels': 'setosa',
                          'model_params': model_params}
            clf = make_pipeline(StandardScaler(), clone(t_model))
            do_scoring_test(X, y, data=df_iris, api_params=api_params,
                            sklearn_model=clf,
                            scorers=scorers, sk_y=sk_y)
