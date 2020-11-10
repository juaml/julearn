# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)

from seaborn import load_dataset

from julearn.utils.testing import do_scoring_test


_features_transformers = {
    'zscore': StandardScaler,
    'scaler_robust': RobustScaler,
    'scaler_minmax': MinMaxScaler,
    'scaler_maxabs': MaxAbsScaler,
    'scaler_normalizer': Normalizer,
    'scaler_quantile': QuantileTransformer,
    'scaler_power': PowerTransformer,
}

_transformer_params = {
    'scaler_quantile': {'n_quantiles': 10}
}


def test_feature_transformers():
    """Test transform X"""
    """Test simple binary classification"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    scorers = ['accuracy']

    for tr_name, tr_klass in _features_transformers.items():
        api_params = {'model': 'svm', 'preprocess_X': tr_name}
        if tr_name in _transformer_params:
            model_params = {f'{tr_name}__{k}': v
                            for k, v in _transformer_params[tr_name].items()}
            api_params['model_params'] = model_params
            tr = tr_klass(**_transformer_params[tr_name])
        else:
            tr = tr_klass()
        clf = make_pipeline(tr, svm.SVC())
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)
