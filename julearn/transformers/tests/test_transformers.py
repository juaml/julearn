# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectPercentile,
    SelectKBest,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    VarianceThreshold,
)
from sklearn.base import BaseEstimator, TransformerMixin
from seaborn import load_dataset

import pytest

from julearn.utils.testing import (
    do_scoring_test,
    PassThroughTransformer,
)
from julearn.transformers import (
    get_transformer,
    reset_transformer_register,
    register_transformer,
)


class fish(BaseEstimator, TransformerMixin):
    def __init__(self, can_it_fly):
        self.can_it_fly = can_it_fly

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


reset_transformer_register()


@pytest.mark.parametrize(
    "name, klass, params",
    [
        ("zscore", StandardScaler, {}),
        ("scaler_robust", RobustScaler, {}),
        ("scaler_minmax", MinMaxScaler, {}),
        ("scaler_maxabs", MaxAbsScaler, {}),
        ("scaler_normalizer", Normalizer, {}),
        ("scaler_quantile", QuantileTransformer, {"n_quantiles": 10}),
        ("scaler_power", PowerTransformer, {}),
        ("select_univariate", GenericUnivariateSelect, {}),
        ("select_percentile", SelectPercentile, {}),
        ("select_k", SelectKBest, {"k": 2}),
        ("select_fdr", SelectFdr, {}),
        ("select_fpr", SelectFpr, {}),
        ("select_fwe", SelectFwe, {}),
        ("select_variance", VarianceThreshold, {}),
    ],
)
def test_feature_transformers(name, klass, params):
    """Test transform X"""
    """Test simple binary classification"""
    df_iris = load_dataset("iris")

    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["setosa", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    X_types = dict(continuous=X)
    y = "species"

    scorers = ["accuracy"]

    api_params = {"model": "svm", "preprocess": name}
    model_params = {f"{name}__{k}": v for k, v in params.items()}
    api_params["model_params"] = model_params
    tr = klass(**params)
    clf = make_pipeline(tr, svm.SVC())
    df_test = df_iris.copy()
    do_scoring_test(
        X,
        y,
        X_types=X_types,
        data=df_test,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )


def test_register_reset():
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer("passthrough", PassThroughTransformer)
    assert get_transformer("passthrough").__class__ == PassThroughTransformer

    with pytest.warns(RuntimeWarning, match="Transformer named"):
        register_transformer(
            "passthrough", PassThroughTransformer
        )
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer(
        "passthrough", PassThroughTransformer, "continuous"
    )
    assert get_transformer("passthrough").__class__ == PassThroughTransformer


def test_register_class_no_default_params():

    reset_transformer_register()
    register_transformer("fish", fish)
    get_transformer("fish", can_it_fly="dont_be_stupid")


def test_register_warning():
    with pytest.warns(RuntimeWarning, match="Transformer name"):
        register_transformer("zscore", fish)
    reset_transformer_register()

    with pytest.raises(ValueError, match="Transformer name"):
        register_transformer("zscore", fish,  overwrite=False)
    reset_transformer_register()

    with pytest.warns(None) as record:
        register_transformer("zscore", fish,   overwrite=True)
    reset_transformer_register()
    assert len(record) == 0
